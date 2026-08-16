from pydantic import BaseModel, Field


class ClinicalNoteRequest(BaseModel):
    note: str = Field(
        ...,
        min_length=10,
        max_length=20000,
        description=(
            "Synthetic or de-identified "
            "clinical text."
        ),
    )


class ClinicalNoteEncodingResponse(BaseModel):
    model: str

    token_count: int

    embedding_dimension: int

    embedding_norm: float

    embedding_preview: list[float]

    synthetic_demo: bool

    disclaimer: str


class ClinicalPatternRequest(BaseModel):
    note: str = Field(
        ...,
        min_length=10,
        max_length=20000,
        description=(
            "Synthetic or de-identified "
            "clinical text."
        ),
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=6,
    )


class ClinicalPatternMatch(BaseModel):
    id: str

    label: str

    title: str

    clinicalbert_similarity: float

    semantic_score: float

    keyword_score: float

    hybrid_score: float

    matched_keywords: list[str]


class ClinicalPatternResponse(BaseModel):
    model: str

    matching_method: str

    semantic_weight: float

    keyword_weight: float

    matches: list[
        ClinicalPatternMatch
    ]

    synthetic_demo: bool

    disclaimer: str