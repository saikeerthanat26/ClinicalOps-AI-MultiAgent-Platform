from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.fhir import router as fhir_router
from app.api.clinical import router as clinical_router
from app.api.rag import router as rag_router
from app.api.risk import router as risk_router
from app.api.nlp import router as nlp_router
from app.api.agent import router as agent_router
from app.core.config import settings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from app.mcp_server import mcp


# ---------------------------------------------------------
# MCP Streamable HTTP application
# ---------------------------------------------------------

mcp_http_app = mcp.streamable_http_app(
    streamable_http_path="/"
)


# ---------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:

    async with (
        mcp.session_manager.run()
    ):

        yield

app = FastAPI(
    title=settings.app_name,
    description=(
        "Local-first Multi-Agent Healthcare Intelligence Platform "
        "powered by open-source AI."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)


app.include_router(chat_router)
app.include_router(fhir_router)
app.include_router(clinical_router)
app.include_router(rag_router)
app.include_router(risk_router)
app.include_router(nlp_router)
app.include_router(agent_router)


# ---------------------------------------------------------
# MCP Streamable HTTP endpoint
# ---------------------------------------------------------

app.mount(
    "/mcp",
    mcp_http_app,
)

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