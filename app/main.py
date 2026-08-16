from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    description=(
        "Local-first Multi-Agent Healthcare Intelligence Platform "
        "powered by open-source AI."
    ),
    version=settings.app_version,
)


app.include_router(chat_router)


@app.get("/")
async def root():
    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "ai_model": settings.ollama_model,
        "architecture": "local-first",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "clinicalops-ai",
        "model": settings.ollama_model,
    }