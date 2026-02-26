# AireHealth AI Service

Clinical note generation microservice using **LangGraph + GPT-4o** with structured outputs.

## Architecture

```
Node.js Backend                    Python AI Service
┌─────────────────┐   HTTP POST    ┌──────────────────────────────┐
│ Express API     │───────────────▶│ FastAPI /generate-note       │
│ LegendAI route  │                │                              │
│                 │◀───────────────│ LangGraph Pipeline:          │
│ Returns JSON    │   JSON         │  1. Format Transcript        │
└─────────────────┘                │  2. Detect Smart Phrases     │
                                   │  3. Resolve from DB          │
                                   │  4. GPT-4o Structured Output │
                                   │  5. Validate & Clean         │
                                   └──────────────────────────────┘
```

## Why LangGraph?

| Feature | Before (Node.js inline) | After (LangGraph) |
|---------|------------------------|-------------------|
| JSON reliability | ~70% (regex parsing) | 100% (Pydantic structured output) |
| Error handling | Single try/catch | Per-node + auto-retry on LLM failures |
| Smart phrases | SQL injection vulnerable | Parameterized queries |
| Prompt management | Hardcoded in 8700-line file | Separate, versionable prompt module |
| Testing | Untestable (coupled to Express) | Each node unit-testable |
| Observability | `console.log` | Structured logging (structlog) |

## Setup

### 1. Install Python 3.12+

```bash
python3 --version  # should be 3.12+
```

### 2. Create virtual environment

```bash
cd /home/umair/projects/airehealth-ai-service
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your database credentials and OpenAI API key
```

### 5. Run the service

```bash
# Development (with auto-reload)
python -m app.main

# Production (with PM2)
pm2 start ecosystem.config.js
```

### 6. Test

```bash
curl -X POST http://localhost:8000/generate-note \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient is a 45-year-old male presenting with acne vulgaris. [smartprocedure] chemicalpeel. Currently on isotretinoin 40mg daily.",
    "patientId": 12345,
    "noteId": 67890,
    "mappedProvider": 61079
  }'
```

## Pipeline Nodes

| Node | Purpose |
|------|---------|
| `format_transcript` | Clean raw dictation (normalize tags, fix punctuation) |
| `detect_smart_phrases` | Find `[smartXXX]` tags via regex |
| `resolve_smart_phrases` | Query `dotPhrases` MySQL table, normalize templates |
| `generate_clinical_note` | GPT-4o structured output → `ClinicalNoteOutput` |
| `validate_output` | Strip leaked headings, verify content exists |

## Docker

```bash
docker build -t airehealth-ai-service .
docker run -p 8000:8000 --env-file .env airehealth-ai-service
```
