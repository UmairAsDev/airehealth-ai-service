"""
LangGraph pipeline definition for clinical note generation.

Pipeline flow:
  format_transcript → detect_smart_phrases → resolve_smart_phrases → generate_clinical_note → validate_output

With retry logic on LLM generation failures.
"""
from __future__ import annotations
import structlog
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.state import PipelineState
from app.graph.nodes import (
    format_transcript_node,
    detect_smart_phrases_node,
    make_resolve_node,
    make_generate_node,
    validate_output_node,
)

logger = structlog.get_logger()

MAX_RETRIES = 2


def should_retry(state: PipelineState) -> str:
    """Conditional edge: retry LLM generation or proceed to validation."""
    error = state.get("error")
    retries = state.get("retries", 0)

    if error and retries < MAX_RETRIES:
        logger.warning("llm_retry", error=error, attempt=retries + 1)
        return "retry"
    return "validate"


def build_pipeline(db: AsyncSession, openai_client: AsyncOpenAI) -> StateGraph:
    """
    Construct the LangGraph pipeline with DB and OpenAI dependencies injected.

    Returns a compiled graph ready to invoke.
    """
    # Create node functions with dependencies
    resolve_node = make_resolve_node(db)
    generate_node = make_generate_node(openai_client)

    # Build the graph
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("format_transcript", format_transcript_node)
    graph.add_node("detect_smart_phrases", detect_smart_phrases_node)
    graph.add_node("resolve_smart_phrases", resolve_node)
    graph.add_node("generate_clinical_note", generate_node)
    graph.add_node("validate_output", validate_output_node)

    # Define edges (linear flow)
    graph.set_entry_point("format_transcript")
    graph.add_edge("format_transcript", "detect_smart_phrases")
    graph.add_edge("detect_smart_phrases", "resolve_smart_phrases")
    graph.add_edge("resolve_smart_phrases", "generate_clinical_note")

    # Conditional: retry or validate
    graph.add_conditional_edges(
        "generate_clinical_note",
        should_retry,
        {
            "retry": "generate_clinical_note",
            "validate": "validate_output",
        },
    )

    graph.add_edge("validate_output", END)

    return graph.compile() #type: ignore


async def run_pipeline(
    db: AsyncSession,
    openai_client: AsyncOpenAI,
    raw_transcript: str,
    patient_id: int,
    note_id: int,
    provider_id: int,
) -> dict:
    """
    Execute the full clinical note generation pipeline.

    Returns the final state dict containing the clinical note or error.
    """
    pipeline = build_pipeline(db, openai_client)

    initial_state: PipelineState = {
        "raw_transcript": raw_transcript,
        "patient_id": patient_id,
        "note_id": note_id,
        "provider_id": provider_id,
        "note_style": "Focused",
        "formatted_transcript": "",
        "cleaned_transcript": "",
        "smart_phrase_matches": [],
        "resolved_templates": [],
        "template_context": "",
        "clinical_note": {},
        "error": None,
        "retries": 0,
    }

    logger.info(
        "pipeline_start",
        patient_id=patient_id,
        note_id=note_id,
        provider_id=provider_id,
        transcript_len=len(raw_transcript),
    )

    result = await pipeline.ainvoke(initial_state) #type: ignore
 
    logger.info(
        "pipeline_complete",
        has_error=bool(result.get("error")),
        note_fields_filled=sum(
            1
            for f in [
                "past_medical_history", "allergies", "current_medication",
                "review_of_system", "history_of_present_illness", "examination",
                "assessment_and_plan", "procedure",
            ]
            if result.get("clinical_note", {}).get(f, "").strip()
        ),
    )

    return result
