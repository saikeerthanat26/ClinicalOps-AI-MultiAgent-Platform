from typing import (
    Annotated,
    Any,
)

from mcp.server import MCPServer
from pydantic import Field

from app.nlp.clinical_pattern_service import (
    clinical_pattern_service,
)

from app.rag.hybrid_retriever import (
    get_hybrid_retriever,
)

from app.schemas.risk import (
    RiskPredictionRequest,
)

from app.services.fhir_service import (
    fhir_service,
)

from app.services.risk_service import (
    risk_service,
)


# ---------------------------------------------------------
# ClinicalOps MCP Server
# ---------------------------------------------------------

mcp = MCPServer(
    "ClinicalOps AI MCP"
)


# ---------------------------------------------------------
# Tool 1
# FHIR patient context
# ---------------------------------------------------------

@mcp.tool()
def get_patient_context(
    patient_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=50,
            description=(
                "Synthetic ClinicalOps patient ID, "
                "for example P001."
            ),
        ),
    ],
) -> dict[str, Any]:
    """
    Retrieve structured synthetic FHIR context for a
    ClinicalOps patient.

    Returns patient demographics, conditions,
    observations, medications, and encounters.

    Synthetic educational data only.
    """

    context = (
        fhir_service
        .get_patient_context(
            patient_id
        )
    )

    if context is None:

        return {
            "found": False,
            "patient_id": patient_id,
            "message": (
                f"Patient {patient_id} "
                "was not found."
            ),
        }

    return {
        "found": True,
        "patient_id": patient_id,
        "context": context,
        "synthetic_demo": True,
    }


# ---------------------------------------------------------
# Tool 2
# Hybrid healthcare retrieval
# ---------------------------------------------------------

@mcp.tool()
def search_healthcare_knowledge(
    query: Annotated[
        str,
        Field(
            min_length=3,
            max_length=2000,
            description=(
                "Healthcare knowledge question "
                "or search query."
            ),
        ),
    ],
    top_k: Annotated[
        int,
        Field(
            ge=1,
            le=5,
        ),
    ] = 3,
) -> dict[str, Any]:
    """
    Search the ClinicalOps healthcare knowledge base.

    Uses hybrid retrieval:
    BM25 + FAISS + reciprocal rank fusion +
    CrossEncoder reranking.

    Returns only sufficiently relevant evidence.
    """

    retriever = (
        get_hybrid_retriever()
    )

    result = retriever.search(
        query=query,
        top_k=top_k,
    )

    relevant = result[
        "relevant"
    ]

    matches = (
        result["results"]
        if relevant
        else []
    )

    return {
        "query": query,
        "relevant": relevant,
        "max_dense_score": result[
            "max_dense_score"
        ],
        "retrieval_method": (
            "BM25+FAISS+RRF+CrossEncoder"
        ),
        "matches": matches,
    }


# ---------------------------------------------------------
# Tool 3
# XGBoost readmission-risk prediction
# ---------------------------------------------------------

@mcp.tool()
def predict_readmission_risk(
    features: RiskPredictionRequest,
) -> dict[str, Any]:
    """
    Run the ClinicalOps synthetic XGBoost
    30-day readmission-risk model.

    Returns probability, tuned-threshold
    classification, risk level, and SHAP
    feature explanations.

    This model is synthetic and is not
    clinically validated.
    """

    result = (
        risk_service
        .predict_with_explanation(
            features=(
                features.model_dump()
            ),
            top_n=5,
        )
    )

    return {
        **result,
        "model_name": (
            "ClinicalOps Synthetic "
            "XGBoost Readmission Model"
        ),
        "synthetic_demo": True,
        "disclaimer": (
            "Synthetic educational model only. "
            "Not clinically validated and not "
            "intended for patient-care decisions."
        ),
    }


# ---------------------------------------------------------
# Tool 4
# ClinicalBERT note analysis
# ---------------------------------------------------------

@mcp.tool()
def analyze_clinical_note(
    note: Annotated[
        str,
        Field(
            min_length=10,
            max_length=20000,
            description=(
                "Synthetic or appropriately "
                "de-identified clinical note."
            ),
        ),
    ],
    top_k: Annotated[
        int,
        Field(
            ge=1,
            le=6,
        ),
    ] = 3,
) -> dict[str, Any]:
    """
    Analyze synthetic or de-identified clinical text
    using Bio_ClinicalBERT plus clinical-concept
    evidence.

    Returns semantic clinical pattern matches.

    Pattern matches are not diagnoses.
    """

    result = (
        clinical_pattern_service
        .match_patterns(
            note=note,
            top_k=top_k,
        )
    )

    return {
        **result,
        "matching_method": (
            "Hybrid ClinicalBERT "
            "+ clinical concept matching"
        ),
        "synthetic_demo": True,
        "disclaimer": (
            "Semantic pattern matching only. "
            "Not a diagnosis or treatment "
            "recommendation."
        ),
    }