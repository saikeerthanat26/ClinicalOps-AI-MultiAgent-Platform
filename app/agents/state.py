from operator import add
from typing import (
    Annotated,
    Any,
    Literal,
    TypedDict,
)


AgentRoute = Literal[
    "fhir",
    "rag",
    "risk",
    "nlp",
]


class ClinicalOpsAgentState(
    TypedDict,
    total=False,
):
    request_id: str

    question: str

    # Current-turn inputs
    patient_id: str | None

    note: str | None

    risk_features: dict[
        str,
        Any,
    ] | None

    # Persistent thread context
    active_patient_id: str | None

    conversation_history: Annotated[
        list[dict[str, Any]],
        add,
    ]

    # Supervisor state
    route: AgentRoute

    route_reason: str

    # Specialist state
    agent_used: str

    agent_result: dict[
        str,
        Any,
    ]

    # Verification state
    verified: bool

    verification_notes: list[str]

    final_answer: str