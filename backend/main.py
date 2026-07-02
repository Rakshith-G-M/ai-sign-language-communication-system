"""
ASL Recognition Backend — FastAPI Application
──────────────────────────────────────────────

Main application entry point.
Handles app setup, middleware, routing, and lifespan management.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.dependencies import get_prediction_service
from core.exceptions import register_exception_handlers
from core.inference.realtime_asl_predictor import shutdown_static_predictor
from core.logging_config import configure_logging, flush_logging
from core.middleware import RequestLoggingMiddleware
from routers.prediction import router as prediction_router
from schemas.health import LivenessResponse, MetricsResponse, ReadinessResponse
from services.prediction_service import ASLPredictionService

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    configure_logging()

    log.info("─" * 60)
    log.info("%s v%s starting", settings.app_name, settings.app_version)
    log.info("Environment: %s", settings.environment)
    log.info("Project root: %s", settings.project_root)
    log.info("Models dir:   %s", settings.models_dir)
    log.info("CORS origins: %s", settings.cors_origin_list)
    log.info("API Docs: http://localhost:%s/docs", settings.port)
    log.info("Health:   http://localhost:%s/health", settings.port)
    log.info("Ready:    http://localhost:%s/ready", settings.port)
    log.info("─" * 60)

    prediction_service = ASLPredictionService()
    app.state.prediction_service = prediction_service

    checks = prediction_service.readiness_checks()
    if prediction_service.is_ready():
        log.info("Readiness: all components ready")
    else:
        log.warning("Readiness: not all components ready — %s", checks)

    yield

    log.info("Shutting down ASL Recognition Engine")
    prediction_service.shutdown()
    shutdown_static_predictor()
    flush_logging()
    log.info("Shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description=(
        "Real-time ASL alphabet recognition via MediaPipe + XGBoost. "
        "Configure production CORS via the CORS_ORIGINS environment variable."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)


@app.get("/", summary="Service root")
async def root():
    return JSONResponse(
        {
            "service": "ASL Recognition Engine",
            "version": settings.api_version,
            "status": "running",
            "docs": "/docs",
            "api_base": "/api/v1",
        }
    )


@app.get(
    "/health",
    response_model=LivenessResponse,
    tags=["health"],
    summary="Process liveness probe",
)
async def health_liveness() -> dict:
    """Return 200 when the process is alive. Does not verify model readiness."""
    return {"status": "ok"}


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["health"],
    summary="Readiness probe",
    responses={503: {"description": "One or more components are not ready."}},
)
async def health_readiness(
    service: ASLPredictionService = Depends(get_prediction_service),
):
    """
    Return 200 when prediction dependencies are ready to serve traffic.
    Returns 503 when static model or MediaPipe failed to initialise.
    """
    checks = service.readiness_checks()
    ready = all(checks.values())
    body = {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }
    status_code = 200 if ready else 503
    return JSONResponse(status_code=status_code, content=body)


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["health"],
    summary="Operational metrics",
)
async def metrics(
    service: ASLPredictionService = Depends(get_prediction_service),
) -> dict:
    """Lightweight in-process operational statistics."""
    return service.get_metrics()


app.include_router(prediction_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
