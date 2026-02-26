"""
Smart phrase detection and resolution service.

Detects [smartXXX] tags in transcripts, queries the dotPhrases MySQL table,
and returns resolved templates with normalized placeholders.
"""
import re
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import SmartPhraseMatch, ResolvedTemplate

logger = structlog.get_logger()

# Maps spoken smart phrase categories -> dotPhrases.templateType values in DB
CATEGORY_TO_TEMPLATE_TYPE: dict[str, str] = {
    "exam": "exam",
    "medication": "currentmedication",
    "procedure": "procedure",
    "hpi": "complaint",
    "ros": "ros",
    "assessment": "assesment",      # matches DB spelling (legacy typo)
    "pmh": "pmh",
    "pasthistory": "pmh",           # [smartpasthistory] -> pmh
    "allergies": "allergies",
}

# Regex to detect [smartXXX] followed by an optional dot and the phrase name
SMART_PHRASE_REGEX = re.compile(
    r"\[smart(\w+)\]\s*\.?([\w.]+)", re.IGNORECASE
)


def detect_smart_phrases(transcript: str) -> tuple[str, list[SmartPhraseMatch]]:
    """
    Find all [smartXXX] tags in the transcript and return:
    - The cleaned transcript (tags removed)
    - List of detected SmartPhraseMatch objects
    """
    matches: list[SmartPhraseMatch] = []
    cleaned = transcript

    for m in SMART_PHRASE_REGEX.finditer(transcript):
        full_match = m.group(0)
        category = m.group(1).lower()
        keyword = m.group(2).lstrip(".")

        matches.append(SmartPhraseMatch(
            full_match=full_match,
            category=category,
            keyword=keyword,
        ))
        # Remove the tag from transcript
        cleaned = cleaned.replace(full_match, "", 1)

    # Collapse leftover whitespace
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    return cleaned, matches


async def resolve_smart_phrases(
    db: AsyncSession,
    matches: list[SmartPhraseMatch],
) -> list[ResolvedTemplate]:
    """
    For each detected smart phrase, query the dotPhrases table and return
    the resolved template with normalized placeholders.
    """
    resolved: list[ResolvedTemplate] = []

    for match in matches:
        template_type = CATEGORY_TO_TEMPLATE_TYPE.get(match.category)

        if not template_type:
            logger.warning(
                "unknown_smart_phrase_category",
                category=match.category,
                keyword=match.keyword,
            )
            continue

        # Parameterized query — no SQL injection
        result = await db.execute(
            text(
                "SELECT name, description, templateType "
                "FROM dotPhrases "
                "WHERE name LIKE :keyword "
                "AND templateType = :template_type "
                "AND deleted = 0 "
                "ORDER BY CASE WHEN name = :exact_keyword THEN 0 ELSE 1 END, name "
                "LIMIT 5"
            ),
            {
                "keyword": f"%{match.keyword}%",
                "template_type": template_type,
                "exact_keyword": match.keyword,
            },
        )
        rows = result.mappings().all()

        if not rows:
            logger.info(
                "no_dot_phrase_found",
                keyword=match.keyword,
                template_type=template_type,
            )
            continue

        # First row is the best match (exact match sorted first)
        best = rows[0]
        raw_html = best["description"] or ""
        normalized = normalize_template_placeholders(raw_html)

        resolved.append(ResolvedTemplate(
            category=match.category,
            keyword=match.keyword,
            template_name=best["name"],
            template_html=raw_html,
            normalized_html=normalized,
        ))

        logger.info(
            "dot_phrase_resolved",
            keyword=match.keyword,
            template_name=best["name"],
            category=match.category,
        )

    return resolved


def normalize_template_placeholders(template_html: str) -> str:
    """
    Convert template labels like "Clinical Diagnosis: <br>" into
    "Clinical Diagnosis: {{ClinicalDiagnosis}}<br>" so the LLM knows
    where to fill values.

    Only matches labels that are standalone (start of string or after <br>)
    to avoid false positives from normal prose containing colons.
    """
    if not template_html:
        return ""

    # Strip <strong>/<b> tags so we can detect label patterns
    cleaned = re.sub(r"</?(?:strong|b)>", "", template_html, flags=re.IGNORECASE)

    def replace_label(m: re.Match) -> str:
        label = m.group(1).strip()
        token = re.sub(r"[^a-zA-Z0-9]", "", label)
        return f"{label}: {{{{{token}}}}}"

    # Match labels at line start or after <br>, followed by empty value
    cleaned = re.sub(
        r"(?:^|(?:<br\s*/?>))\s*([A-Za-z][A-Za-z0-9\s/\-()&]{1,40}):\s*(?=<br\s*/?>|\n|$)",
        replace_label,
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned


async def get_provider_note_style(db: AsyncSession, provider_id: int) -> str:
    """Fetch the provider's noteFormattingStyle from the providers table."""
    result = await db.execute(
        text(
            "SELECT noteFormattingStyle FROM providers "
            "WHERE providerId = :provider_id LIMIT 1"
        ),
        {"provider_id": provider_id},
    )
    row = result.mappings().first()
    if row and row.get("noteFormattingStyle"):
        return row["noteFormattingStyle"]
    return "Focused"
