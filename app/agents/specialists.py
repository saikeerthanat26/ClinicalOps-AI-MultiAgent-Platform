from app.agents.state import (
    ClinicalOpsAgentState,
)

from app.nlp.clinical_pattern_service import (
    clinical_pattern_service,
)

from app.services.clinical_service import (
    clinical_service,
)

from app.services.rag_service import (
    rag_service,
)

from app.services.risk_service import (
    risk_service,
)


# ---------------------------------------------------------
# FHIR Agent
# ---------------------------------------------------------

async def fhir_agent_node(
    state: ClinicalOpsAgentState,
) -> dict:

    patient_id = (
        state.get(
            "patient_id"
        )
        or state.get(
            "active_patient_id"
        )
    )

    question = state[
        "question"
    ]

    if patient_id is None:

        return {
            "agent_used": (
                "fhir_agent"
            ),
            "agent_result": {
                "answer": (
                    "A patient ID is required "
                    "for FHIR analysis."
                ),
                "grounded": False,
            },
        }

    answer = (
        await clinical_service.ask_patient(
            patient_id=patient_id,
            question=question,
        )
    )

    if answer is None:

        answer = (
            f"Patient {patient_id} "
            "was not found."
        )

        grounded = False

    else:

        grounded = True

    return {
        "agent_used": (
            "fhir_agent"
        ),
        "agent_result": {
            "answer": answer,
            "patient_id": patient_id,
            "grounded": grounded,
        },
    }

# ---------------------------------------------------------
# RAG Agent
# ---------------------------------------------------------

async def rag_agent_node(
    state: ClinicalOpsAgentState,
) -> dict:

    result = await rag_service.ask(
        question=state[
            "question"
        ],
        top_k=3,
    )

    return {
        "agent_used": (
            "rag_agent"
        ),
        "agent_result": {
            **result,
            "grounded": True,
        },
    }


# ---------------------------------------------------------
# Risk Agent
# ---------------------------------------------------------

def risk_agent_node(
    state: ClinicalOpsAgentState,
) -> dict:

    risk_features = state.get(
        "risk_features"
    )

    if not risk_features:

        return {
            "agent_used": (
                "risk_agent"
            ),
            "agent_result": {
                "answer": (
                    "Structured risk features "
                    "were not provided."
                ),
                "synthetic_demo": True,
            },
        }

    result = (
        risk_service
        .predict_with_explanation(
            features=risk_features,
            top_n=5,
        )
    )

    probability = result[
        "probability_percent"
    ]

    classification = result[
        "predicted_readmission"
    ]

    risk_level = result[
        "risk_level"
    ]

    answer = (
        "Synthetic readmission-risk model "
        f"probability: {probability}%. "
        f"Risk level: {risk_level}. "
        "The model classification is "
        f"{'positive' if classification else 'negative'} "
        "using the stored tuned threshold."
    )

    return {
        "agent_used": (
            "risk_agent"
        ),
        "agent_result": {
            "answer": answer,
            **result,
            "synthetic_demo": True,
            "disclaimer": (
                "Synthetic educational model only. "
                "Not clinically validated."
            ),
        },
    }


# ---------------------------------------------------------
# Clinical NLP Agent
# ---------------------------------------------------------

def nlp_agent_node(
    state: ClinicalOpsAgentState,
) -> dict:

    note = state.get(
        "note"
    )

    if not note:

        return {
            "agent_used": (
                "clinical_nlp_agent"
            ),
            "agent_result": {
                "answer": (
                    "A clinical note is required "
                    "for ClinicalBERT analysis."
                ),
                "synthetic_demo": True,
            },
        }

    result = (
        clinical_pattern_service
        .match_patterns(
            note=note,
            top_k=3,
        )
    )

    matches = result[
        "matches"
    ]

    if matches:

        match_summary = ", ".join(
            (
                f"{match['label']} "
                f"({match['hybrid_score']:.4f})"
            )
            for match in matches
        )

        answer = (
            "Top semantic clinical pattern "
            f"matches: {match_summary}. "
            "These are semantic pattern matches, "
            "not diagnoses."
        )

    else:

        answer = (
            "No clinical semantic pattern "
            "matches were returned."
        )

    return {
        "agent_used": (
            "clinical_nlp_agent"
        ),
        "agent_result": {
            "answer": answer,
            **result,
            "synthetic_demo": True,
            "disclaimer": (
                "Pattern similarity is not a "
                "diagnosis or treatment recommendation."
            ),
        },
    }