"""
LangGraph node functions for the clinical note pipeline.

Each node is a pure function: (state) -> partial state update.
DB and LLM dependencies are injected via closures.
"""
from __future__ import annotations
import structlog
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.state import PipelineState
from app.services.transcript import format_transcript
from app.services.smart_phrases import (
    detect_smart_phrases,
    resolve_smart_phrases,
    get_provider_note_style,
)
from app.prompts.clinical_note import (
    get_system_prompt,
    get_user_prompt,
    build_template_context,
)
from app.models.schemas import ClinicalNoteOutput

logger = structlog.get_logger()


# ─── Node 1: Format Transcript ──────────────────────────────────────────────

async def format_transcript_node(state: PipelineState) -> dict:
    """Clean and normalize the raw dictation text."""
    raw = state["raw_transcript"]
    formatted = format_transcript(raw)

    logger.info("transcript_formatted", raw_len=len(raw), formatted_len=len(formatted))

    return {"formatted_transcript": formatted}


# ─── Node 2: Detect Smart Phrases ───────────────────────────────────────────

async def detect_smart_phrases_node(state: PipelineState) -> dict:
    """Find [smartXXX] tags in the formatted transcript."""
    transcript = state["formatted_transcript"]
    cleaned, matches = detect_smart_phrases(transcript)

    logger.info(
        "smart_phrases_detected",
        count=len(matches),
        categories=[m.category for m in matches],
    )

    return {
        "cleaned_transcript": cleaned,
        "smart_phrase_matches": [m.model_dump() for m in matches],
    }


# ─── Node 3: Resolve Smart Phrases (DB) ─────────────────────────────────────

def make_resolve_node(db: AsyncSession):
    """Factory: creates a node with the DB session injected."""

    async def resolve_smart_phrases_node(state: PipelineState) -> dict:
        """Query dotPhrases table and resolve templates + fetch provider style."""
        from app.models.schemas import SmartPhraseMatch

        matches = [SmartPhraseMatch(**m) for m in state["smart_phrase_matches"]]

        # Parallel: resolve templates + fetch provider style
        resolved = await resolve_smart_phrases(db, matches)
        note_style = await get_provider_note_style(db, state["provider_id"])

        # Build template context string for LLM
        template_dicts = [
            {"category": r.category, "normalized_html": r.normalized_html}
            for r in resolved
        ]
        context = build_template_context(template_dicts)

        logger.info(
            "smart_phrases_resolved",
            resolved_count=len(resolved),
            note_style=note_style,
        )

        return {
            "resolved_templates": [r.model_dump() for r in resolved],
            "template_context": context,
            "note_style": note_style,
        }

    return resolve_smart_phrases_node


# ─── Node 4: Generate Clinical Note (LLM) ───────────────────────────────────

def make_generate_node(openai_client: AsyncOpenAI):
    """Factory: creates a node with the OpenAI client injected."""

    async def generate_clinical_note_node(state: PipelineState) -> dict:
        """Call GPT-4o with structured output to generate the clinical note."""
        note_style = state.get("note_style", "Focused")
        transcript = state["cleaned_transcript"]
        template_context = state.get("template_context", "")

        system_prompt = get_system_prompt(note_style)
        user_prompt = get_user_prompt(transcript, template_context)

        logger.info(
            "llm_call_start",
            transcript_len=len(transcript),
            has_templates=bool(template_context),
            note_style=note_style,
        )

        # Use OpenAI structured output — guarantees valid JSON matching our Pydantic model
        completion = await openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=ClinicalNoteOutput,
            temperature=0.2,
        )

        parsed: ClinicalNoteOutput = completion.choices[0].message.parsed #type: ignore

        if parsed is None:
            # Refusal or content filter triggered
            refusal = completion.choices[0].message.refusal
            logger.error("llm_refusal", refusal=refusal)
            return {
                "error": refusal or "LLM refused to generate note",
                "clinical_note": {},
            }

        logger.info(
            "llm_call_complete",
            icd_count=len(parsed.icdCodes),
            cpt_count=len(parsed.cptCodes),
            has_hpi=bool(parsed.history_of_present_illness),
            has_exam=bool(parsed.examination),
            has_ap=bool(parsed.assessment_and_plan),
        )

        return {
            "clinical_note": parsed.model_dump(),
            "error": None,
        }

    return generate_clinical_note_node


# ─── Node 5: Validate Output ────────────────────────────────────────────────

async def validate_output_node(state: PipelineState) -> dict:
    """
    Post-processing validation of the generated note.
    
    Checks:
    - At least one section has content (not all empty)
    - ICD codes have valid format (letter + digits)
    - No section headings leaked into content
    """
    note = state.get("clinical_note", {})

    if not note:
        return {"error": "No clinical note generated"}

    # Check if at least one content field is non-empty
    content_fields = [
        "past_medical_history", "allergies", "current_medication",
        "review_of_system", "history_of_present_illness", "examination",
        "assessment_and_plan", "procedure",
    ]
    has_content = any(note.get(f, "").strip() for f in content_fields)

    if not has_content:
        # Check if the "no content" marker was set
        icd_codes = note.get("icdCodes", [])
        if icd_codes and icd_codes[0].get("Code") == "NONE":
            return {"error": "Insufficient or unrelated content"}
        return {"error": "No clinical content could be extracted from the dictation"}

    # Clean up: strip section headings that may have leaked into content
    cleaned_note = dict(note)
    heading_patterns = {
        "past_medical_history": ["Past Medical History:", "PMH:"],
        "allergies": ["Allergies:"],
        "current_medication": ["Current Medication:", "Medications:"],
        "review_of_system": ["Review of Systems:", "ROS:"],
        "history_of_present_illness": ["History of Present Illness:", "HPI:"],
        "examination": ["Examination:", "Physical Examination:", "Exam:"],
        "assessment_and_plan": ["Assessment and Plan:", "A&P:", "Assessment:"],
        "procedure": ["Procedure:", "Procedures:"],
    }

    for field, headings in heading_patterns.items():
        val = cleaned_note.get(field, "")
        if val:
            for heading in headings:
                # Remove heading from start of content (case-insensitive)
                if val.strip().lower().startswith(heading.lower()):
                    val = val.strip()[len(heading):].strip()
                # Also check for HTML-wrapped headings
                for tag in ["<strong>", "<b>", "<u>"]:
                    wrapped = f"{tag}{heading}"
                    if val.strip().lower().startswith(wrapped.lower()):
                        close_tag = tag.replace("<", "</")
                        val = val.strip()[len(wrapped):].strip()
                        if val.startswith(close_tag):
                            val = val[len(close_tag):].strip()
            cleaned_note[field] = val

    logger.info("output_validated", content_fields_filled=sum(1 for f in content_fields if cleaned_note.get(f, "").strip()))

    return {"clinical_note": cleaned_note, "error": None}
