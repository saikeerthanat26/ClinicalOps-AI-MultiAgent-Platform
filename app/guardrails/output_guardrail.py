import re

from app.agents.state import (
    ClinicalOpsAgentState,
)


# ---------------------------------------------------------
# Unsafe directive patterns
#
# These target generated recommendations rather than
# neutral record summaries.
# ---------------------------------------------------------

UNSAFE_OUTPUT_PATTERNS = [
    r"\byou\s+should\s+take\b",
    r"\byou\s+should\s+stop\b",
    r"\byou\s+should\s+start\b",
    r"\bincrease\s+your\s+dose\b",
    r"\bdecrease\s+your\s+dose\b",
    r"\bstop\s+taking\s+your\b",
    r"\bstart\s+taking\s+\d",
    r"\bi\s+recommend\s+that\s+you\s+take\b",
]


def _contains_unsafe_directive(
    text: str,
) -> bool:

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in UNSAFE_OUTPUT_PATTERNS
    )


def output_guardrail_node(
    state: ClinicalOpsAgentState,
) -> dict:

    route = state.get(
        "route"
    )

    result = state.get(
        "agent_result",
        {},
    )

    answer = (
        state.get(
            "final_answer"
        )
        or result.get(
            "answer"
        )
        or ""
    )

    flags: list[str] = []

    critical_failure = False


    # -----------------------------------------------------
    # Unsafe generated recommendation
    # -----------------------------------------------------

    if _contains_unsafe_directive(
        answer
    ):

        flags.append(
            "unsafe_clinical_directive"
        )

        critical_failure = True


    # -----------------------------------------------------
    # FHIR grounding
    # -----------------------------------------------------

    if route == "fhir":

        if not result.get(
            "grounded",
            False,
        ):

            flags.append(
                "ungrounded_fhir_output"
            )

            critical_failure = True


    # -----------------------------------------------------
    # RAG grounding
    # -----------------------------------------------------

    elif route == "rag":

        retrieval_relevant = (
            result.get(
                "retrieval_relevant"
            )
        )

        generation_used = (
            result.get(
                "generation_used"
            )
        )

        sources = result.get(
            "sources",
            [],
        )

        if (
            retrieval_relevant is False
            and generation_used is True
        ):

            flags.append(
                "generation_without_relevant_evidence"
            )

            critical_failure = True

        if (
            retrieval_relevant is True
            and generation_used is True
            and not sources
        ):

            flags.append(
                "rag_generation_without_sources"
            )

            critical_failure = True


    # -----------------------------------------------------
    # Risk model requirements
    # -----------------------------------------------------

    elif route == "risk":

        if not result.get(
            "synthetic_demo",
            False,
        ):

            flags.append(
                "missing_synthetic_risk_label"
            )

            critical_failure = True

        disclaimer = (
            "Synthetic educational model only. "
            "Not clinically validated and not "
            "intended for patient-care decisions."
        )

        if (
            "not clinically validated"
            not in answer.lower()
        ):

            answer = (
                f"{answer}\n\n"
                f"{disclaimer}"
            )

            flags.append(
                "risk_disclaimer_appended"
            )


    # -----------------------------------------------------
    # Clinical NLP requirements
    # -----------------------------------------------------

    elif route == "nlp":

        if not result.get(
            "synthetic_demo",
            False,
        ):

            flags.append(
                "missing_synthetic_nlp_label"
            )

            critical_failure = True

        if (
            "not diagnoses"
            not in answer.lower()
            and
            "not a diagnosis"
            not in answer.lower()
        ):

            answer = (
                f"{answer}\n\n"
                "These semantic pattern matches "
                "are not diagnoses or treatment "
                "recommendations."
            )

            flags.append(
                "nlp_disclaimer_appended"
            )


    # -----------------------------------------------------
    # Final guardrail decision
    # -----------------------------------------------------

    if critical_failure:

        safe_answer = (
            "The generated response was blocked "
            "by ClinicalOps output guardrails "
            "because it did not satisfy the "
            "platform's grounding or safety "
            "requirements."
        )

        return {
            "output_guardrail_passed": (
                False
            ),

            "output_guardrail_flags": (
                flags
            ),

            "guardrail_blocked": True,

            "final_answer": (
                safe_answer
            ),
        }


    return {
        "output_guardrail_passed": (
            True
        ),

        "output_guardrail_flags": (
            flags
        ),

        "guardrail_blocked": False,

        "final_answer": answer,
    }