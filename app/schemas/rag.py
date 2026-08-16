from pydantic import BaseModel, Field


class RAGSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
    )


class RAGSearchResult(BaseModel):
    id: str
    title: str
    source: str
    content: str
    score: float


class RAGSearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[RAGSearchResult]


class RAGAskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
    )


class RAGSource(BaseModel):
    id: str
    title: str
    source: str
    dense_score: float
    bm25_score: float
    rrf_score: float
    reranker_score: float


class RAGAskResponse(BaseModel):
    answer: str
    model: str | None
    grounded: bool

    retrieval_method: str
    retrieval_relevant: bool
    generation_used: bool
    max_dense_score: float

    sources: list[RAGSource]


class HybridSearchResult(BaseModel):
    id: str
    title: str
    source: str
    content: str

    dense_score: float
    bm25_score: float
    rrf_score: float
    reranker_score: float


class HybridSearchResponse(BaseModel):
    query: str
    relevant: bool
    max_dense_score: float
    result_count: int
    results: list[HybridSearchResult]