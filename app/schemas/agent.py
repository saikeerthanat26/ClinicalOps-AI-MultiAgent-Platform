from typing import Any

from pydantic import (
    BaseModel,
    Field,
)

from app.schemas.risk import (
    RiskPredictionRequest,
)


class MultiAgentRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=3,
        max_length=4000,
    )

    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description=(
            "Reuse the same thread ID "
            "for multi-turn memory."
        ),
    )

    patient_id: str | None = (
        None
    )

    note: str | None = Field(
        default=None,
        max_length=20000,
    )

    risk_features: (
        RiskPredictionRequest
        | None
    ) = None


class MultiAgentResponse(BaseModel):

    request_id: str

    thread_id: str

    memory_turns: int

    active_patient_id: str | None

    route: str

    route_reason: str

    agent_used: str

    verified: bool

    verification_notes: list[str]

    answer: str

    data: dict[
        str,
        Any,
    ]