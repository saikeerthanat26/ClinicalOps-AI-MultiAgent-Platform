from fastapi import APIRouter

from app.core.config import settings
from app.rag.hybrid_retriever import get_hybrid_retriever
from app.rag.vector_store import get_vector_store
from app.schemas.rag import (
    HybridSearchResponse,
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


# ---------------------------------------------------------
# Basic FAISS Semantic Search
# ---------------------------------------------------------

@router.post(
    "/search",
    response_model=RAGSearchResponse,
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


# ---------------------------------------------------------
# Hybrid Search
# BM25 + FAISS + RRF + CrossEncoder
# ---------------------------------------------------------

@router.post(
    "/hybrid-search",
    response_model=HybridSearchResponse,
)
async def hybrid_search(
    request: RAGSearchRequest,
) -> HybridSearchResponse:

    retriever = get_hybrid_retriever()

    result = retriever.search(
        query=request.query,
        top_k=request.top_k,
    )

    return HybridSearchResponse(
        query=request.query,
        relevant=result["relevant"],
        max_dense_score=result["max_dense_score"],
        result_count=len(result["results"]),
        results=result["results"],
    )


# ---------------------------------------------------------
# Full Grounded RAG Generation
# ---------------------------------------------------------

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

    generation_used = result["generation_used"]

    return RAGAskResponse(
        answer=result["answer"],
        model=(
            settings.ollama_model
            if generation_used
            else None
        ),
        grounded=True,
        retrieval_method="BM25+FAISS+RRF+CrossEncoder",
        retrieval_relevant=result["retrieval_relevant"],
        generation_used=generation_used,
        max_dense_score=result["max_dense_score"],
        sources=result["sources"],
    )