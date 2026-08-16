from pydantic import BaseModel, Field


class RiskPredictionRequest(BaseModel):
    age: int = Field(
        ...,
        ge=18,
        le=120,
    )

    prior_admissions_12m: int = Field(
        ...,
        ge=0,
        le=20,
    )

    length_of_stay_days: float = Field(
        ...,
        ge=0,
        le=365,
    )

    chronic_condition_count: int = Field(
        ...,
        ge=0,
        le=30,
    )

    medication_count: int = Field(
        ...,
        ge=0,
        le=50,
    )

    recent_ed_visit: int = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "1 if a recent ED visit is present, "
            "otherwise 0"
        ),
    )

    hba1c: float = Field(
        ...,
        ge=3.0,
        le=20.0,
    )

    systolic_bp: int = Field(
        ...,
        ge=60,
        le=260,
    )

    egfr: float = Field(
        ...,
        ge=0,
        le=200,
    )

    followup_days: int = Field(
        ...,
        ge=0,
        le=90,
    )


class RiskPredictionResponse(BaseModel):
    probability: float
    probability_percent: float
    decision_threshold: float

    predicted_readmission: bool
    risk_level: str

    model_name: str
    synthetic_demo: bool
    disclaimer: str


class RiskFactor(BaseModel):
    feature: str
    value: float

    shap_value: float

    impact: str


class RiskExplainResponse(
    RiskPredictionResponse
):
    explanation_method: str

    shap_output_space: str

    top_factors: list[RiskFactor]