# Backend Architecture

This document describes the ASL Sign Language Communication Platform backend:
its structure, request flow, configuration, logging, error handling, and
operational endpoints. It is intended for contributors and production
maintenance.

---

## Overview

The backend is a **FastAPI** application that exposes a REST API for:

- Real-time ASL letter/word prediction from webcam frames
- Session-scoped text assembly (words and sentences)
- Text-to-speech generation
- Optional dynamic gesture data collection

Inference runs **entirely on-device** using pre-trained models. The API layer
is responsible for HTTP concerns, validation, observability, and session
management — not for ML algorithm logic.

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI (main.py)                       │
│  Middleware → Routers → Services → Inference → ML artefacts     │
└─────────────────────────────────────────────────────────────────┘
         │                    │                │
         ▼                    ▼                ▼
   Request logging      ASLPredictionService   XGBoost + MediaPipe
   CORS / exceptions    Session isolation    DynamicPredictor (ONNX/fallback)
```

### Directory layout

```
backend/
├── main.py                     # App entry, lifespan, health/metrics routes
├── core/
│   ├── config.py               # Centralized Settings (env-driven)
│   ├── logging_config.py       # Structured logging setup
│   ├── middleware.py           # Request ID + latency middleware
│   ├── exceptions.py           # Global exception handlers
│   ├── dependencies.py         # FastAPI Depends() providers
│   └── inference/              # Prediction pipeline (unchanged behaviour)
├── routers/
│   ├── prediction.py           # /api/v1/predict, /tts, /state, …
│   └── data_collection.py      # /api/v1/collect/*
├── services/
│   └── prediction_service.py   # Orchestrates inference + sessions
├── schemas/                    # Pydantic request/response models
└── tests/                      # Unit + HTTP integration tests
```

---

## Request lifecycle

Every HTTP request passes through the following stages:

```
Client
  │
  ▼
CORSMiddleware
  │
  ▼
RequestLoggingMiddleware
  │  • Assign X-Request-ID (or reuse incoming header)
  │  • Record start time
  ▼
Router endpoint
  │  • Pydantic validation (422 on failure)
  │  • Depends(get_prediction_service)
  ▼
ASLPredictionService
  │  • Session lookup / creation
  │  • Image decode + pipeline
  ▼
Inference modules (predict_frame, DynamicPredictor, TextBuilder)
  │
  ▼
JSON response (+ X-Request-ID header)
  │
  ▼
Middleware logs: method, path, status, duration_ms, request_id
```

### Traceability

Each request receives a **`X-Request-ID`** header in the response. This ID
is included in error payloads and server logs, making it possible to correlate
client errors with server-side log lines.

Health probe paths (`/health`, `/ready`, `/metrics`, `/api/v1/health`) are
excluded from INFO-level request completion logs to reduce noise.

---

## Prediction lifecycle

A single `POST /api/v1/predict` request follows this path:

1. **Router validation** — MIME type check (`image/jpeg`, `image/png`,
   `image/webp`); upload size limit enforced via `settings.max_upload_bytes`.
2. **Image decode** — OpenCV `imdecode`. Invalid/empty payloads return HTTP
   **200** with a degraded `PredictionResponse` (preserved frontend contract).
3. **Session resolution** — `session_id` query param is normalised; invalid
   values fall back to `"default"` without rejecting the request.
4. **Static prediction** — `predict_frame()` runs MediaPipe + XGBoost with
   three-layer stabilisation (confidence gate → majority vote → hysteresis).
5. **Dynamic prediction** — If the hand is moving, static letter output is
   suppressed and `DynamicPredictor` may return a whole word (ONNX or geometric
   fallback).
6. **Text assembly** — `TextBuilder` updates word/sentence state and optional
   spell-check suggestions.
7. **Response** — `PredictionResponse` JSON with latency in milliseconds.

Prediction-specific logs include: `prediction_ms`, `latency_ms`, `letter`,
`word`, `confidence`, `model` (`static` | `dynamic`), and `hand_detected`.

---

## Service architecture

### ASLPredictionService

A **singleton** created during application lifespan and stored on
`app.state.prediction_service`. Routers access it via
`Depends(get_prediction_service)`.

Responsibilities:

| Concern | Implementation |
|---------|----------------|
| Session isolation | `dict[str, PredictionSession]` keyed by normalised session ID |
| Idle cleanup | Sessions inactive > `session_idle_seconds` removed on each request |
| Capacity limit | Evicts oldest non-default session when `max_sessions` reached |
| Metrics | In-process counters: total/static/dynamic predictions |
| Readiness | Delegates to `is_static_predictor_ready()` |

### Expensive singletons (import-time)

| Component | Initialisation | Notes |
|-----------|----------------|-------|
| XGBoost model + LabelEncoder | `realtime_asl_predictor` import | Loaded once; shared across sessions |
| MediaPipe Hands | Same module | Closed on shutdown via `shutdown_static_predictor()` |
| DynamicPredictor | Service `__post_init__` | ONNX if artefacts present; else geometric fallback |
| SymSpell dictionary | Per `TextBuilder` instance | Loaded per session |

Models are **not** reloaded per request.

---

## Configuration flow

All runtime settings live in `core/config.py` as a **`Settings`** object
(pydantic-settings). Access via:

```python
from core.config import settings
```

Or for dependency injection:

```python
from core.dependencies import get_app_settings
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LOG_FORMAT` | `text` | `text` or `json` |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `PROJECT_ROOT` | auto-detected | Repository root (set `/app` in Docker) |
| `MAX_UPLOAD_BYTES` | `10485760` | Max image upload size (10 MB) |
| `MAX_BASE64_CHARS` | `14000000` | Max base64 payload length |
| `MAX_TTS_CHARS` | `5000` | Max TTS input length |
| `MAX_SESSION_ID_LENGTH` | `128` | Session ID truncation limit |
| `MAX_COLLECT_FRAMES` | `300` | Max frames per collection request |
| `SESSION_IDLE_SECONDS` | `600` | Session TTL |
| `MAX_SESSIONS` | `1000` | In-memory session cap |

### CORS in production

Development uses `CORS_ORIGINS=*` by default. For production, set explicit
origins:

```bash
CORS_ORIGINS=https://app.example.com,https://www.example.com
```

### Path resolution

Derived paths (not set directly):

- `{project_root}/models/` — XGBoost and ONNX artefacts
- `{project_root}/dataset/dynamic_gestures.jsonl` — collection dataset

---

## Logging strategy

Logging is configured once at startup via `configure_logging()`:

- **Development** — human-readable text format with timestamp, level, logger
  name, and message.
- **Production** — set `LOG_FORMAT=json` for one JSON object per line.

### What gets logged

| Event | Level | Key fields |
|-------|-------|------------|
| Request completed | INFO | `request_id`, `method`, `path`, `status_code`, `duration_ms` |
| Prediction completed | INFO | `prediction_ms`, `letter`, `confidence`, `model` |
| Degraded prediction | WARNING | `latency_ms`, reason message |
| Unhandled exception | ERROR | `request_id`, stack trace (server-side only) |
| Startup / shutdown | INFO | paths, readiness status |

Third-party noise (`uvicorn.access`, `multipart`) is suppressed to WARNING.

---

## Error handling strategy

The backend uses a layered error model that **preserves existing API contracts**:

| Situation | HTTP status | Response shape |
|-----------|-------------|----------------|
| Pydantic validation failure | 422 | `{ detail: [...], request_id }` |
| Invalid MIME / upload too large | 400 | `{ detail: "...", request_id }` |
| Empty TTS text | 400 | `{ detail: "Empty text", request_id }` |
| Invalid/corrupt image | **200** | Degraded `PredictionResponse` |
| Unhandled server error | 500 | `{ detail: "Internal server error", request_id }` |

Stack traces are **never** returned in API responses. They are logged server-side
via the global exception handler in `core/exceptions.py`.

---

## Startup and shutdown lifecycle

FastAPI **`lifespan`** context manager (replaces deprecated `@app.on_event`):

### Startup

1. Configure logging from `settings.log_level`
2. Log resolved paths and CORS configuration
3. Instantiate `ASLPredictionService` → `app.state.prediction_service`
4. Log readiness check results

### Shutdown

1. `prediction_service.shutdown()` — clear all sessions
2. `shutdown_static_predictor()` — close MediaPipe Hands context
3. `flush_logging()` — flush all log handlers

---

## Health and metrics endpoints

| Endpoint | Purpose | Status codes |
|----------|---------|--------------|
| `GET /health` | **Liveness** — process is running | Always 200 |
| `GET /ready` | **Readiness** — models + MediaPipe ready | 200 or 503 |
| `GET /metrics` | Operational counters | 200 |
| `GET /api/v1/health` | **Legacy liveness** (frontend) | Always `{ "status": "ok" }` |

### Readiness checks

```json
{
  "status": "ready",
  "checks": {
    "static_model": true,
    "mediapipe": true,
    "dynamic_predictor": true,
    "prediction_service": true
  }
}
```

`dynamic_predictor` is always `true` because geometric fallback is always
available. `static_model` and `mediapipe` require XGBoost artefacts and a
successful MediaPipe initialisation.

### Metrics

```json
{
  "uptime_seconds": 3600.5,
  "active_sessions": 3,
  "total_predictions": 1200,
  "static_predictions": 1100,
  "dynamic_predictions": 45
}
```

No external metrics dependency (Prometheus, etc.) — counters are in-process.

---

## Dependency relationships

```
main.py
 ├── core/config.py          (settings singleton)
 ├── core/logging_config.py  (startup)
 ├── core/middleware.py      (RequestLoggingMiddleware)
 ├── core/exceptions.py      (error handlers)
 ├── routers/prediction.py
 │    └── services/prediction_service.py
 │         ├── core/inference/realtime_asl_predictor.py  [import-time model load]
 │         ├── core/inference/dynamic_predictor.py
 │         ├── core/inference/prediction_session.py
 │         └── core/inference/text_builder.py
 └── routers/data_collection.py
      └── core/config.py     (dataset path)
```

### API endpoints summary

| Method | Path | Tag |
|--------|------|-----|
| GET | `/` | root |
| GET | `/health` | health |
| GET | `/ready` | health |
| GET | `/metrics` | health |
| POST | `/api/v1/predict` | prediction |
| POST | `/api/v1/predict-base64` | prediction |
| POST | `/api/v1/reset` | prediction |
| GET | `/api/v1/state` | prediction |
| GET | `/api/v1/health` | prediction |
| GET | `/api/v1/info` | prediction |
| POST | `/api/v1/tts` | prediction |
| POST | `/api/v1/collect/sequence` | data-collection |
| GET | `/api/v1/collect/stats` | data-collection |
| DELETE | `/api/v1/collect/clear` | data-collection |

---

## Security notes

This milestone focuses on production **baselines** without authentication:

- Upload and payload size limits are configurable
- Session IDs are normalised to prevent unbounded arbitrary keys
- CORS is permissive by default for development; restrict via `CORS_ORIGINS`
- Data collection endpoints remain **unauthenticated** (deferred to a future
  deployment milestone)
- Internal errors do not leak stack traces

---

## Running locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Ensure model artefacts exist at `{project_root}/models/`:

- `asl_xgboost.pkl`
- `label_encoder.pkl`

Optional dynamic models:

- `asl_dynamic.onnx`
- `dynamic_label_encoder.pkl`
- `dynamic_scaler.pkl`

## Running tests

```bash
cd backend
pytest tests/ -v
```
