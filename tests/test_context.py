from app.agents.context import (
    context_node,
)


def test_new_patient_becomes_active_patient():

    state = {
        "patient_id": "P001"
    }

    result = context_node(
        state
    )

    assert (
        result[
            "active_patient_id"
        ]
        == "P001"
    )


def test_missing_patient_keeps_existing_memory():

    state = {
        "patient_id": None,

        "active_patient_id": (
            "P001"
        ),
    }

    result = context_node(
        state
    )

    assert result == {}