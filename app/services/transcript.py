"""
Transcript formatting service.

Cleans raw voice dictation text:
- Normalizes smart phrase tags
- Handles [period] markers
- Fixes punctuation and casing
"""
import re

# All recognized smart phrase categories
SMART_PHRASE_TAGS = [
    "smartassessment",
    "smartprocedure",
    "smartmedication",
    "smartpasthistory",
    "smartros",
    "smartexam",
    "smarthpi",
    "smartpmh",
    "smartallergies",
]


def format_transcript(raw_text: str) -> str:
    """
    Clean and normalize raw dictation text before smart phrase detection.

    Steps:
    1. Merge broken sentence-ending periods (e.g. "a. b." -> "a.b.")
    2. Clean smart phrase tags (remove stray punctuation around brackets)
    3. Normalize whitespace
    4. Handle [period] markers and sentence breaks
    5. Capitalize first letter of sentences
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text

    # Step 1: Merge broken periods between lowercase letters
    text = re.sub(r"([a-z])\.\s*\n*\s*([a-z])\.", r"\1.\2.", text, flags=re.IGNORECASE)

    # Step 2: Clean smart phrase tags — remove trailing punctuation after [tag]
    for phrase in SMART_PHRASE_TAGS:
        pattern = re.compile(rf"\[\s*{phrase}\s*\]\s*[,.;:]*\s*", re.IGNORECASE)
        text = pattern.sub(f"[{phrase}] ", text)

    # Step 3: Normalize whitespace
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Step 4: Handle [period] markers and natural sentence breaks
    text = re.sub(r"\[period\]", "[period]\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\. ", ".\n\n", text)

    # Step 5: Capitalize first letter after period or [period]
    def capitalize_after_break(m: re.Match) -> str:
        return m.group(1) + m.group(2).upper()

    text = re.sub(r"(^|\.\s+|\[period\]\s+)([a-z])", capitalize_after_break, text)

    return text
