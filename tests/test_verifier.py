from app.agents.verifier import (
    verifier_node,
)


def test_fhir_grounded_response_is_verified():

    state = {
        "request_id": (
            "test-fhir"
        ),

        "question": (
            "What medications are documented?"
        ),

        "route": "fhir",

        "agent_used": (
            "fhir_agent"
        ),

        "patient_id": "P001",

        "agent_result": {
            "answer": (
                "Metformin, Lisinopril, "
                "and Furosemide."
            ),

            "grounded": True,
        },
    }

    result = verifier_node(
        state
    )

    assert (
        result[
            "verified"
        ]
        is True
    )

    assert (
        result[
            "final_answer"
        ]
        is not None
    )


def test_rag_response_with_sources_is_verified():

    state = {
        "request_id": (
            "test-rag"
        ),

        "question": (
            "What is HCC risk adjustment?"
        ),

        "route": "rag",

        "agent_used": (
            "rag_agent"
        ),

        "agent_result": {
            "answer": (
                "HCC risk adjustment..."
            ),

            "retrieval_relevant": True,

            "sources": [
                {
                    "id": "KB001"
                }
            ],
        },
    }

    result = verifier_node(
        state
    )

    assert (
        result[
            "verified"
        ]
        is True
    )


def test_rag_irrelevant_query_can_safely_refuse():

    state = {
        "request_id": (
            "test-rag-refusal"
        ),

        "question": (
            "How do I repair a "
            "car transmission?"
        ),

        "route": "rag",

        "agent_used": (
            "rag_agent"
        ),

        "agent_result": {
            "answer": (
                "The knowledge base does "
                "not contain sufficient evidence."
            ),

            "retrieval_relevant": False,

            "generation_used": False,

            "sources": [],
        },
    }

    result = verifier_node(
        state
    )

    assert (
        result[
            "verified"
        ]
        is True
    )


def test_synthetic_risk_response_is_verified():

    state = {
        "request_id": (
            "test-risk"
        ),

        "question": (
            "Estimate synthetic "
            "readmission risk."
        ),

        "route": "risk",

        "agent_used": (
            "risk_agent"
        ),

        "agent_result": {
            "answer": (
                "Synthetic probability: 46.59%."
            ),

            "synthetic_demo": True,
        },
    }

    result = verifier_node(
        state
    )

    assert (
        result[
            "verified"
        ]
        is True
    )


def test_nlp_matches_are_verified():

    state = {
        "request_id": (
            "test-nlp"
        ),

        "question": (
            "Analyze this clinical note."
        ),

        "route": "nlp",

        "agent_used": (
            "clinical_nlp_agent"
        ),

        "agent_result": {
            "answer": (
                "Top pattern: copd_pattern."
            ),

            "matches": [
                {
                    "id": "CP004",
                    "label": (
                        "copd_pattern"
                    ),
                }
            ],
        },
    }

    result = verifier_node(
        state
    )

    assert (
        result[
            "verified"
        ]
        is True
    )