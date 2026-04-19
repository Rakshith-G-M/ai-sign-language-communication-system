"""
ASL Recognition Backend — FastAPI Application
──────────────────────────────────────────────

Main application entry point.
Handles app setup, middleware, and routing.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# Import router
from routers.prediction import router as prediction_router

# ─────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ASL Recognition Engine",
    description="Real-time sign language recognition via MediaPipe + XGBoost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ─────────────────────────────────────────────────────────────────────────────
# CORS Middleware
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Root Endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return JSONResponse({
        "service": "ASL Recognition Engine",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "api_base": "/api/v1",
    })

# ─────────────────────────────────────────────────────────────────────────────
# Include Routers
# ─────────────────────────────────────────────────────────────────────────────
app.include_router(prediction_router)

# ─────────────────────────────────────────────────────────────────────────────
# Startup & Shutdown Events
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    log.info("─" * 60)
    log.info("ASL Recognition Engine Starting")
    log.info("API Docs: http://localhost:8000/docs")
    log.info("Health: http://localhost:8000/api/v1/health")
    log.info("─" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    log.info("ASL Recognition Engine Shutdown")

# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,   # ✅ Enable for development
        log_level="info",
    )