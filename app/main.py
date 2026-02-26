"""
FastAPI application — AI Clinical Note Generation Service.

Provides the /generate-note endpoint that the Node.js backend calls
to replace the inline LegendAI function.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import get_db, startup_db, shutdown_db
from app.models.schemas import GenerateNoteRequest, GenerateNoteResponse, ClinicalNoteOutput
from app.graph.pipeline import run_pipeline

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
)

logger = structlog.get_logger()

# ─── OpenAI Client (singleton) ──────────────────────────────────────────────

_openai_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        settings = get_settings()
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


# ─── App Lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger.info("startup", message="Connecting to database...")
    await startup_db()
    logger.info("startup", message="Database connected. Service ready.")
    yield
    logger.info("shutdown", message="Closing database connections...")
    await shutdown_db()
    logger.info("shutdown", message="Service stopped.")


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AireHealth AI Service",
    description="Clinical note generation using LangGraph + GPT-4o",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Health Check ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "airehealth-ai-service"}


# ─── Main Endpoint ──────────────────────────────────────────────────────────

@app.post("/generate-note", response_model=GenerateNoteResponse)
async def generate_note(
    request: GenerateNoteRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a structured clinical note from a voice dictation transcript.

    This endpoint replaces the Node.js LegendAI function.
    Called by the Express backend at POST /emr/progress-note/common/LegendAI.

    Input:
    - text: Raw dictation transcript
    - patientId: Patient ID
    - noteId: Progress note ID
    - mappedProvider: Provider ID (used to fetch noteFormattingStyle)

    Output:
    - Structured clinical note with HTML sections + ICD/CPT codes

    Pipeline:
    1. Format transcript (clean punctuation, normalize tags)
    2. Detect [smartXXX] tags
    3. Resolve smart phrases from dotPhrases DB table
    4. Generate clinical note via GPT-4o (structured output)
    5. Validate and clean output
    """
    try:
        openai_client = get_openai_client()

        result = await run_pipeline(
            db=db,
            openai_client=openai_client,
            raw_transcript=request.text,
            patient_id=request.patientId,
            note_id=request.noteId,
            provider_id=request.mappedProvider,
        )

        error = result.get("error")
        if error:
            logger.warning("pipeline_error", error=error)
            return GenerateNoteResponse(
                success=False,
                error=error,
            )

        note_data = result.get("clinical_note", {})
        clinical_note = ClinicalNoteOutput(**note_data)

        return GenerateNoteResponse(
            success=True,
            data=clinical_note,
        )

    except Exception as e:
        logger.exception("unhandled_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"AI service error: {str(e)}",
        )


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=settings.environment == "development",
    )
