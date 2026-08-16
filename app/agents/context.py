from app.agents.state import (
    ClinicalOpsAgentState,
)


def context_node(
    state: ClinicalOpsAgentState,
) -> dict:

    current_patient_id = (
        state.get(
            "patient_id"
        )
    )

    # A newly supplied patient ID becomes
    # the active patient for this thread.

    if current_patient_id:

        return {
            "active_patient_id": (
                current_patient_id
            )
        }

    # If no patient ID was supplied in this turn,
    # LangGraph keeps the previously checkpointed
    # active_patient_id.

    return {}