import json

from app.agents.state import (
    ClinicalOpsAgentState,
)

from app.services.mcp_client_service import (
    clinicalops_mcp_client,
)

from app.services.ollama_service import (
    ollama_service,
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
                "tool_transport": (
                    "MCP"
                ),
            },
        }

    # -----------------------------------------------------
    # Retrieve patient data THROUGH MCP
    # -----------------------------------------------------

    mcp_result = (
        await clinicalops_mcp_client
        .get_patient_context(
            patient_id
        )
    )

    if not mcp_result.get(
        "found",
        False,
    ):

        return {
            "agent_used": (
                "fhir_agent"
            ),
            "agent_result": {
                "answer": (
                    f"Patient {patient_id} "
                    "was not found."
                ),
                "patient_id": (
                    patient_id
                ),
                "grounded": False,
                "tool_transport": (
                    "MCP"
                ),
                "mcp_tool": (
                    "get_patient_context"
                ),
            },
        }

    patient_context = (
        mcp_result[
            "context"
        ]
    )

    # -----------------------------------------------------
    # Qwen receives only MCP-retrieved context
    # -----------------------------------------------------

    context_text = (
        json.dumps(
            patient_context,
            indent=2,
        )
    )

    answer = (
        await ollama_service
        .rag_chat(
            question=question,
            context=context_text,
        )
    )

    return {
        "agent_used": (
            "fhir_agent"
        ),
        "agent_result": {
            "answer": answer,
            "patient_id": (
                patient_id
            ),
            "grounded": True,
            "tool_transport": (
                "MCP"
            ),
            "mcp_tool": (
                "get_patient_context"
            ),
        },
    }


# ---------------------------------------------------------
# RAG Agent
# ---------------------------------------------------------

async def rag_agent_node(
    state: ClinicalOpsAgentState,
) -> dict:

    question = state[
        "question"
    ]

    # -----------------------------------------------------
    # Retrieval THROUGH MCP
    # -----------------------------------------------------

    mcp_result = (
        await clinicalops_mcp_client
        .search_healthcare_knowledge(
            query=question,
            top_k=3,
        )
    )

    relevant = mcp_result.get(
        "relevant",
        False,
    )

    max_dense_score = (
        mcp_result.get(
            "max_dense_score",
            0.0,
        )
    )

    # -----------------------------------------------------
    # Retrieval gate
    # -----------------------------------------------------

    if not relevant:

        answer = (
            "The ClinicalOps knowledge base "
            "does not contain sufficient "
            "relevant evidence to answer "
            "this question."
        )

        return {
            "agent_used": (
                "rag_agent"
            ),
            "agent_result": {
                "answer": answer,
                "retrieval_relevant": (
                    False
                ),
                "generation_used": (
                    False
                ),
                "max_dense_score": (
                    max_dense_score
                ),
                "sources": [],
                "grounded": True,
                "tool_transport": (
                    "MCP"
                ),
                "mcp_tool": (
                    "search_healthcare_knowledge"
                ),
            },
        }

    matches = (
        mcp_result.get(
            "matches",
            [],
        )
    )

    # -----------------------------------------------------
    # Build evidence context for local Qwen
    # -----------------------------------------------------

    context_parts = []

    for match in matches:

        context_parts.append(
            (
                f"Source ID: "
                f"{match.get('id')}\n"
                f"Title: "
                f"{match.get('title')}\n"
                f"Content: "
                f"{match.get('content')}"
            )
        )

    context_text = (
        "\n\n".join(
            context_parts
        )
    )

    # -----------------------------------------------------
    # Grounded generation
    # -----------------------------------------------------

    answer = (
        await ollama_service
        .rag_chat(
            question=question,
            context=context_text,
        )
    )

    # -----------------------------------------------------
    # Return compact source metadata
    # -----------------------------------------------------

    sources = []

    for match in matches:

        sources.append(
            {
                "id": (
                    match.get(
                        "id"
                    )
                ),
                "title": (
                    match.get(
                        "title"
                    )
                ),
                "source": (
                    match.get(
                        "source"
                    )
                ),
                "dense_score": (
                    match.get(
                        "dense_score"
                    )
                ),
                "bm25_score": (
                    match.get(
                        "bm25_score"
                    )
                ),
                "rrf_score": (
                    match.get(
                        "rrf_score"
                    )
                ),
                "reranker_score": (
                    match.get(
                        "reranker_score"
                    )
                ),
            }
        )

    return {
        "agent_used": (
            "rag_agent"
        ),
        "agent_result": {
            "answer": answer,
            "retrieval_relevant": (
                True
            ),
            "generation_used": (
                True
            ),
            "max_dense_score": (
                max_dense_score
            ),
            "sources": sources,
            "grounded": True,
            "tool_transport": (
                "MCP"
            ),
            "mcp_tool": (
                "search_healthcare_knowledge"
            ),
        },
    }


# ---------------------------------------------------------
# Risk Agent
# ---------------------------------------------------------

async def risk_agent_node(
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
                "tool_transport": (
                    "MCP"
                ),
            },
        }

    # -----------------------------------------------------
    # XGBoost + SHAP THROUGH MCP
    # -----------------------------------------------------

    result = (
        await clinicalops_mcp_client
        .predict_readmission_risk(
            risk_features
        )
    )

    probability = (
        result[
            "probability_percent"
        ]
    )

    classification = (
        result[
            "predicted_readmission"
        ]
    )

    risk_level = (
        result[
            "risk_level"
        ]
    )

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
            "tool_transport": (
                "MCP"
            ),
            "mcp_tool": (
                "predict_readmission_risk"
            ),
        },
    }


# ---------------------------------------------------------
# Clinical NLP Agent
# ---------------------------------------------------------

async def nlp_agent_node(
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
                "tool_transport": (
                    "MCP"
                ),
            },
        }

    # -----------------------------------------------------
    # ClinicalBERT THROUGH MCP
    # -----------------------------------------------------

    result = (
        await clinicalops_mcp_client
        .analyze_clinical_note(
            note=note,
            top_k=3,
        )
    )

    matches = (
        result.get(
            "matches",
            [],
        )
    )

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
            "These are semantic pattern "
            "matches, not diagnoses."
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
            "tool_transport": (
                "MCP"
            ),
            "mcp_tool": (
                "analyze_clinical_note"
            ),
        },
    }