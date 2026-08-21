import re

from app.agents.state import (
    ClinicalOpsAgentState,
)


# ---------------------------------------------------------
# Prompt injection indicators
# ---------------------------------------------------------

PROMPT_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\bignore\s+(all\s+)?prior\s+instructions\b",
    r"\bignore\s+the\s+system\s+prompt\b",
    r"\breveal\s+(the\s+)?system\s+prompt\b",
    r"\bshow\s+(me\s+)?the\s+system\s+prompt\b",
    r"\bdeveloper\s+message\b",
    r"\bbypass\s+(the\s+)?guardrails?\b",
    r"\bdisable\s+(the\s+)?guardrails?\b",
    r"\bjailbreak\b",
    r"\bact\s+as\s+an?\s+unrestricted\b",
]


# ---------------------------------------------------------
# Direct clinical advice indicators
#
# ClinicalOps is an educational/synthetic platform.
# It can summarize records and demonstrate models,
# but should not prescribe or diagnose.
# ---------------------------------------------------------

CLINICAL_ADVICE_PATTERNS = [
    r"\bdiagnose\s+me\b",
    r"\bdiagnose\s+this\s+patient\b",

    r"\bwhat\s+(medication|medicine|drug)\s+should\s+i\s+take\b",

    r"\bwhat\s+(medication|medicine|drug)\s+should\s+the\s+patient\s+take\b",

    r"\bwhat\s+dose\s+should\s+i\s+take\b",

    r"\bhow\s+much\s+.+\s+should\s+i\s+take\b",

    r"\bshould\s+i\s+(start|stop|increase|decrease|change)\b",

    r"\bshould\s+the\s+patient\s+(start|stop|increase|decrease|change)\b",

    r"\bprescribe\s+(me|this\s+patient)\b",

    r"\bwhat\s+treatment\s+should\s+i\s+use\b",

    r"\bwhat\s+treatment\s+should\s+the\s+patient\s+receive\b",
]


# ---------------------------------------------------------
# Obvious direct identifiers
#
# This is only a lightweight demo heuristic.
# It is NOT a complete PHI/HIPAA de-identification system.
# ---------------------------------------------------------

SENSITIVE_IDENTIFIER_PATTERNS = {
    "email_address": (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    "ssn_like": (
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),

    "phone_number": (
        r"\b(?:\+?1[-.\s]?)?"
        r"\(?\d{3}\)?[-.\s]?"
        r"\d{3}[-.\s]?\d{4}\b"
    ),

    "mrn_like": (
        r"\bMRN\s*[:#-]?\s*"
        r"[A-Za-z0-9-]{5,}\b"
    ),
}


def _matches_any(
    text: str,
    patterns: list[str],
) -> bool:

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


def input_guardrail_node(
    state: ClinicalOpsAgentState,
) -> dict:

    question = (
        state.get(
            "question",
            "",
        )
        or ""
    )

    note = (
        state.get(
            "note",
            "",
        )
        or ""
    )

    combined_text = (
        f"{question}\n{note}"
    )

    flags: list[str] = []

    reasons: list[str] = []


    # -----------------------------------------------------
    # Prompt injection
    # -----------------------------------------------------

    if _matches_any(
        combined_text,
        PROMPT_INJECTION_PATTERNS,
    ):

        flags.append(
            "prompt_injection"
        )

        reasons.append(
            "Potential prompt-injection or "
            "instruction-bypass attempt detected."
        )


    # -----------------------------------------------------
    # Direct diagnosis / treatment request
    # -----------------------------------------------------

    if _matches_any(
        combined_text,
        CLINICAL_ADVICE_PATTERNS,
    ):

        flags.append(
            "direct_clinical_advice"
        )

        reasons.append(
            "ClinicalOps does not provide direct "
            "diagnosis, prescribing, dosing, or "
            "treatment recommendations."
        )


    # -----------------------------------------------------
    # Obvious identifier heuristic
    # -----------------------------------------------------

    for (
        identifier_name,
        pattern,
    ) in SENSITIVE_IDENTIFIER_PATTERNS.items():

        if re.search(
            pattern,
            combined_text,
            flags=re.IGNORECASE,
        ):

            flags.append(
                f"sensitive_identifier:"
                f"{identifier_name}"
            )

    if any(
        flag.startswith(
            "sensitive_identifier:"
        )
        for flag in flags
    ):

        reasons.append(
            "Potential direct identifier detected. "
            "Use only synthetic or appropriately "
            "de-identified clinical text."
        )


    # -----------------------------------------------------
    # Decision
    # -----------------------------------------------------

    passed = (
        len(flags) == 0
    )

    if passed:

        reason = (
            "Input passed ClinicalOps "
            "guardrail checks."
        )

    else:

        reason = " ".join(
            reasons
        )


    return {
        "input_guardrail_passed": (
            passed
        ),
        "input_guardrail_flags": (
            flags
        ),
        "input_guardrail_reason": (
            reason
        ),
        "guardrail_blocked": (
            not passed
        ),
    }


def route_after_input_guardrail(
    state: ClinicalOpsAgentState,
) -> str:

    if state.get(
        "input_guardrail_passed",
        False,
    ):

        return "continue"

    return "blocked"


def blocked_request_node(
    state: ClinicalOpsAgentState,
) -> dict:

    reason = state.get(
        "input_guardrail_reason",
        (
            "The request was blocked "
            "by ClinicalOps guardrails."
        ),
    )

    flags = state.get(
        "input_guardrail_flags",
        [],
    )

    answer = (
        "Request blocked by ClinicalOps "
        f"input guardrails. {reason}"
    )

    return {
        "route": "blocked",

        "route_reason": (
            "Input guardrail rejected "
            "the request before agent routing."
        ),

        "agent_used": (
            "input_guardrail"
        ),

        "agent_result": {
            "answer": answer,
            "guardrail_blocked": True,
            "guardrail_flags": flags,
        },

        "verified": False,

        "verification_notes": [
            (
                "Request was stopped before "
                "LLM routing or MCP tool execution."
            )
        ],

        "final_answer": answer,

        "output_guardrail_passed": None,

        "output_guardrail_flags": [],
    }