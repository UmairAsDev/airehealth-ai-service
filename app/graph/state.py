"""
LangGraph state definition for the clinical note generation pipeline.
"""
from __future__ import annotations
from typing import Annotated, Any
from typing_extensions import TypedDict

from app.models.schemas import (
    SmartPhraseMatch,
    ResolvedTemplate,
    ClinicalNoteOutput,
)


class PipelineState(TypedDict):
    """
    Immutable state passed through the LangGraph pipeline.
    Each node reads from and writes to this state.
    """

    # ── Inputs (set once at entry) ──────────────────────────────
    raw_transcript: str
    patient_id: int
    note_id: int
    provider_id: int

    # ── Derived from DB ─────────────────────────────────────────
    note_style: str  # Focused / Comprehensive / Categorized

    # ── Processing stages ───────────────────────────────────────
    formatted_transcript: str
    cleaned_transcript: str  # after smart phrase tags removed
    smart_phrase_matches: list[dict]  # serialized SmartPhraseMatch
    resolved_templates: list[dict]  # serialized ResolvedTemplate
    template_context: str  # formatted for LLM prompt

    # ── LLM output ──────────────────────────────────────────────
    clinical_note: dict  # serialized ClinicalNoteOutput
    
    # ── Pipeline metadata ───────────────────────────────────────
    error: str | None
    retries: int
