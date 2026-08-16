from app.agents.state import (
    ClinicalOpsAgentState,
)


def verifier_node(
    state: ClinicalOpsAgentState,
) -> dict:

    route = state.get(
        "route"
    )

    result = state.get(
        "agent_result",
        {},
    )

    notes = []

    verified = True

    answer = result.get(
        "answer"
    )

    # --------------------------------------------------
    # Basic output validation
    # --------------------------------------------------

    if not answer:

        verified = False

        notes.append(
            "Agent returned no answer."
        )

    # --------------------------------------------------
    # FHIR validation
    # --------------------------------------------------

    if route == "fhir":

        if not result.get(
            "grounded",
            False,
        ):

            verified = False

            notes.append(
                "FHIR result was not grounded "
                "to a valid patient record."
            )

        else:

            notes.append(
                "FHIR response is grounded to "
                "synthetic patient context."
            )

    # --------------------------------------------------
    # RAG validation
    # --------------------------------------------------

    elif route == "rag":

        if (
            result.get(
                "retrieval_relevant"
            )
            is False
        ):

            notes.append(
                "No sufficiently relevant "
                "RAG evidence was found; "
                "generation was safely skipped."
            )

        else:

            sources = result.get(
                "sources",
                [],
            )

            if not sources:

                verified = False

                notes.append(
                    "Relevant RAG response "
                    "did not contain sources."
                )

            else:

                notes.append(
                    "RAG response contains "
                    "retrieved evidence sources."
                )

    # --------------------------------------------------
    # Risk validation
    # --------------------------------------------------

    elif route == "risk":

        if not result.get(
            "synthetic_demo",
            False,
        ):

            verified = False

            notes.append(
                "Risk result is missing "
                "synthetic-demo labeling."
            )

        else:

            notes.append(
                "Risk output is labeled as a "
                "synthetic, non-clinically-validated "
                "prediction."
            )

    # --------------------------------------------------
    # NLP validation
    # --------------------------------------------------

    elif route == "nlp":

        matches = result.get(
            "matches",
            [],
        )

        if not matches:

            verified = False

            notes.append(
                "Clinical NLP agent returned "
                "no pattern matches."
            )

        else:

            notes.append(
                "Clinical NLP output contains "
                "interpretable pattern matches."
            )

    # --------------------------------------------------
    # Final answer
    # --------------------------------------------------

    if verified:

        final_answer = answer

    else:

        final_answer = (
            answer
            or (
                "ClinicalOps could not verify "
                "the specialist agent output."
            )
        )

    # --------------------------------------------------
    # Store one compact memory record
    # --------------------------------------------------

    effective_patient_id = (
        state.get(
            "patient_id"
        )
        or state.get(
            "active_patient_id"
        )
    )

    history_entry = {
        "request_id": state.get(
            "request_id"
        ),
        "question": state.get(
            "question"
        ),
        "route": route,
        "agent_used": state.get(
            "agent_used"
        ),
        "patient_id": (
            effective_patient_id
        ),
        "answer": final_answer,
        "verified": verified,
    }

    return {
        "verified": verified,
        "verification_notes": notes,
        "final_answer": final_answer,
        "conversation_history": [
            history_entry
        ],
    }