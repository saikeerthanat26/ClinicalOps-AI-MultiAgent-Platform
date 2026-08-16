from fastapi import (
    APIRouter,
    HTTPException,
)

from app.schemas.risk import (
    RiskExplainResponse,
    RiskPredictionRequest,
    RiskPredictionResponse,
)

from app.services.risk_service import (
    risk_service,
)


router = APIRouter(
    prefix="/api/v1/risk",
    tags=[
        "Predictive Risk Modeling"
    ],
)


MODEL_NAME = (
    "ClinicalOps Synthetic "
    "XGBoost Readmission Model"
)

DISCLAIMER = (
    "Synthetic educational model only. "
    "Not clinically validated and not "
    "intended for diagnosis, treatment, "
    "or patient-care decisions."
)


# ---------------------------------------------------------
# Prediction only
# ---------------------------------------------------------

@router.post(
    "/predict",
    response_model=RiskPredictionResponse,
)
async def predict_readmission_risk(
    request: RiskPredictionRequest,
) -> RiskPredictionResponse:

    try:
        result = risk_service.predict(
            request.model_dump()
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
                "Risk prediction failed: "
                f"{error}"
            ),
        )

    return RiskPredictionResponse(
        probability=result[
            "probability"
        ],
        probability_percent=result[
            "probability_percent"
        ],
        decision_threshold=result[
            "decision_threshold"
        ],
        predicted_readmission=result[
            "predicted_readmission"
        ],
        risk_level=result[
            "risk_level"
        ],
        model_name=MODEL_NAME,
        synthetic_demo=True,
        disclaimer=DISCLAIMER,
    )


# ---------------------------------------------------------
# Prediction + SHAP explanation
# ---------------------------------------------------------

@router.post(
    "/predict-explain",
    response_model=RiskExplainResponse,
)
async def predict_readmission_risk_with_explanation(
    request: RiskPredictionRequest,
) -> RiskExplainResponse:

    try:
        result = (
            risk_service.predict_with_explanation(
                request.model_dump(),
                top_n=5,
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
                "Risk explanation failed: "
                f"{error}"
            ),
        )

    return RiskExplainResponse(
        probability=result[
            "probability"
        ],
        probability_percent=result[
            "probability_percent"
        ],
        decision_threshold=result[
            "decision_threshold"
        ],
        predicted_readmission=result[
            "predicted_readmission"
        ],
        risk_level=result[
            "risk_level"
        ],
        model_name=MODEL_NAME,
        synthetic_demo=True,
        disclaimer=DISCLAIMER,
        explanation_method=result[
            "explanation_method"
        ],
        shap_output_space=result[
            "shap_output_space"
        ],
        top_factors=result[
            "top_factors"
        ],
    )