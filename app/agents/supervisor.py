from app.agents.state import (
    ClinicalOpsAgentState,
)

from app.services.ollama_service import (
    ollama_service,
)


async def supervisor_node(
    state: ClinicalOpsAgentState,
) -> dict:

    # --------------------------------------------------
    # Resolve patient context
    # --------------------------------------------------

    effective_patient_id = (
        state.get(
            "patient_id"
        )
        or state.get(
            "active_patient_id"
        )
    )

    routing_result = (
        await ollama_service.route_agent(
            question=state[
                "question"
            ],
            patient_id=(
                effective_patient_id
            ),
            note=state.get(
                "note"
            ),
            risk_features=state.get(
                "risk_features"
            ),
        )
    )

    route = routing_result[
        "route"
    ]

    reason = routing_result[
        "reason"
    ]

    # --------------------------------------------------
    # Deterministic safety fallbacks
    # --------------------------------------------------

    if (
        route == "fhir"
        and not effective_patient_id
    ):

        route = "rag"

        reason = (
            "FHIR routing required a patient "
            "context, so the request was safely "
            "routed to general healthcare RAG."
        )

    if (
        route == "risk"
        and not state.get(
            "risk_features"
        )
    ):

        route = "rag"

        reason = (
            "Risk routing required structured "
            "risk features, so the request was "
            "safely routed to RAG."
        )

    if (
        route == "nlp"
        and not state.get(
            "note"
        )
    ):

        route = "rag"

        reason = (
            "Clinical NLP routing required a "
            "clinical note, so the request was "
            "safely routed to RAG."
        )

    return {
        "route": route,
        "route_reason": reason,
    }