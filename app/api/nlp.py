from fastapi import (
    APIRouter,
    HTTPException,
)

from app.nlp.clinicalbert_service import (
    clinicalbert_service,
)

from app.nlp.clinical_pattern_service import (
    clinical_pattern_service,
)

from app.schemas.nlp import (
    ClinicalNoteEncodingResponse,
    ClinicalNoteRequest,
    ClinicalPatternRequest,
    ClinicalPatternResponse,
)


router = APIRouter(
    prefix="/api/v1/nlp",
    tags=[
        "Clinical NLP"
    ],
)


DISCLAIMER = (
    "Educational ClinicalOps AI demo. "
    "Use only synthetic or appropriately "
    "de-identified text. ClinicalBERT "
    "similarity results are semantic pattern "
    "matches and must not be interpreted as "
    "diagnoses or treatment recommendations."
)


# ---------------------------------------------------------
# ClinicalBERT encoder
# ---------------------------------------------------------

@router.post(
    "/clinicalbert/encode",
    response_model=(
        ClinicalNoteEncodingResponse
    ),
)
async def encode_clinical_note(
    request: ClinicalNoteRequest,
) -> ClinicalNoteEncodingResponse:

    try:

        result = (
            clinicalbert_service
            .encode_note(
                request.note
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "ClinicalBERT encoding failed: "
                f"{error}"
            ),
        )

    return ClinicalNoteEncodingResponse(
        model=result["model"],
        token_count=result[
            "token_count"
        ],
        embedding_dimension=result[
            "embedding_dimension"
        ],
        embedding_norm=result[
            "embedding_norm"
        ],
        embedding_preview=result[
            "embedding_preview"
        ],
        synthetic_demo=True,
        disclaimer=DISCLAIMER,
    )


# ---------------------------------------------------------
# Clinical semantic pattern matching
# ---------------------------------------------------------

@router.post(
    "/clinicalbert/pattern-match",
    response_model=(
        ClinicalPatternResponse
    ),
)
async def match_clinical_patterns(
    request: ClinicalPatternRequest,
) -> ClinicalPatternResponse:

    try:

        result = (
            clinical_pattern_service
            .match_patterns(
                note=request.note,
                top_k=request.top_k,
            )
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Clinical pattern matching "
                f"failed: {error}"
            ),
        )

    return ClinicalPatternResponse(
        model=result["model"],
        matching_method=(
            "Hybrid ClinicalBERT "
            "+ clinical concept matching"
        ),
        semantic_weight=result[
            "semantic_weight"
        ],
        keyword_weight=result[
            "keyword_weight"
        ],
        matches=result["matches"],
        synthetic_demo=True,
        disclaimer=DISCLAIMER,
    )