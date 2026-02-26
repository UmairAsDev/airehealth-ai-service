"""
Pydantic models for API request/response validation and LLM structured output.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ─── Enums ───────────────────────────────────────────────────────────────────

class NoteStyle(str, Enum):
    FOCUSED = "Focused"
    COMPREHENSIVE = "Comprehensive"
    CATEGORIZED = "Categorized"


# ─── API Request ─────────────────────────────────────────────────────────────

class GenerateNoteRequest(BaseModel):
    """Incoming request from the Node.js backend."""
    text: str = Field(..., description="Raw dictation transcript")
    patientId: int = Field(..., description="Patient ID")
    noteId: int = Field(..., description="Progress note ID")
    mappedProvider: int = Field(..., description="Provider ID for note formatting style")


# ─── LLM Structured Output ──────────────────────────────────────────────────

class ICDCode(BaseModel):
    Code: str = Field(default="", description="ICD-10 code")
    Description: str = Field(default="", description="ICD-10 description")


class CPTCode(BaseModel):
    Code: str = Field(default="", description="CPT code")
    Description: str = Field(default="", description="CPT description")


class ClinicalNoteOutput(BaseModel):
    """
    The exact JSON schema the LLM must produce.
    Using OpenAI structured outputs, this is guaranteed to always be valid.
    """
    past_medical_history: str = Field(default="", description="Past medical history (HTML)")
    allergies: str = Field(default="", description="Allergies (HTML)")
    current_medication: str = Field(default="", description="Current medications (HTML)")
    review_of_system: str = Field(default="", description="Review of systems (HTML)")
    history_of_present_illness: str = Field(default="", description="HPI (HTML)")
    examination: str = Field(default="", description="Physical examination (HTML)")
    assessment_and_plan: str = Field(default="", description="Assessment and plan (HTML)")
    procedure: str = Field(default="", description="Procedures performed (HTML)")
    icdCodes: list[ICDCode] = Field(default_factory=list, description="ICD-10 codes")
    cptCodes: list[CPTCode] = Field(default_factory=list, description="CPT codes")


# ─── Internal Pipeline Models ────────────────────────────────────────────────

class SmartPhraseMatch(BaseModel):
    """A detected [smartXXX] tag in the transcript."""
    full_match: str
    category: str
    keyword: str


class ResolvedTemplate(BaseModel):
    """A smart phrase resolved from the dotPhrases DB table."""
    category: str
    keyword: str
    template_name: str
    template_html: str
    normalized_html: str


# ─── API Response ────────────────────────────────────────────────────────────

class GenerateNoteResponse(BaseModel):
    """Response returned to the Node.js backend."""
    success: bool = True
    data: Optional[ClinicalNoteOutput] = None
    error: Optional[str] = None
