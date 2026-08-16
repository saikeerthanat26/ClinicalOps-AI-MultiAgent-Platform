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
    score: float


class RAGAskResponse(BaseModel):
    answer: str
    model: str
    grounded: bool
    sources: list[RAGSource]