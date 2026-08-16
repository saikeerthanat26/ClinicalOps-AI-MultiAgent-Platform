from fastapi import APIRouter

from app.rag.vector_store import get_vector_store
from app.core.config import settings
from app.rag.vector_store import get_vector_store
from app.schemas.rag import (
    RAGAskRequest,
    RAGAskResponse,
    RAGSearchRequest,
    RAGSearchResponse,
)
from app.services.rag_service import rag_service


router = APIRouter(
    prefix="/api/v1/rag",
    tags=["Healthcare RAG"],
)


@router.post(
    "/search",
    response_model=RAGSearchResponse,
)
@router.post(
    "/ask",
    response_model=RAGAskResponse,
)
async def rag_ask(
    request: RAGAskRequest,
) -> RAGAskResponse:

    result = await rag_service.ask(
        question=request.question,
        top_k=request.top_k,
    )

    return RAGAskResponse(
        answer=result["answer"],
        model=settings.ollama_model,
        grounded=True,
        sources=result["sources"],
    )
async def semantic_search(
    request: RAGSearchRequest,
) -> RAGSearchResponse:
    vector_store = get_vector_store()

    results = vector_store.search(
        query=request.query,
        top_k=request.top_k,
    )

    return RAGSearchResponse(
        query=request.query,
        result_count=len(results),
        results=results,
    )