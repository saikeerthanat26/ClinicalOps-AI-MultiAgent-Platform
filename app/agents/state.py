from operator import add

from typing import (
    Annotated,
    Any,
    TypedDict,
)


class ClinicalOpsAgentState(
    TypedDict,
    total=False,
):

    # -----------------------------------------------------
    # Request
    # -----------------------------------------------------

    request_id: str

    question: str

    patient_id: str | None

    note: str | None

    risk_features: dict[
        str,
        Any,
    ] | None


    # -----------------------------------------------------
    # Thread memory
    # -----------------------------------------------------

    active_patient_id: str | None

    conversation_history: Annotated[
        list[dict[str, Any]],
        add,
    ]


    # -----------------------------------------------------
    # Input guardrails
    # -----------------------------------------------------

    input_guardrail_passed: bool

    input_guardrail_flags: list[str]

    input_guardrail_reason: str

    guardrail_blocked: bool


    # -----------------------------------------------------
    # Supervisor
    # -----------------------------------------------------

    route: str

    route_reason: str


    # -----------------------------------------------------
    # Specialist
    # -----------------------------------------------------

    agent_used: str

    agent_result: dict[
        str,
        Any,
    ]


    # -----------------------------------------------------
    # Verification
    # -----------------------------------------------------

    verified: bool

    verification_notes: list[str]

    final_answer: str


    # -----------------------------------------------------
    # Output guardrails
    # -----------------------------------------------------

    output_guardrail_passed: (
        bool | None
    )

    output_guardrail_flags: (
        list[str]
    )