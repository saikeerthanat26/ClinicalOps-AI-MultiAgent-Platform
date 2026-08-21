import time

from uuid import uuid4

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from app.agents.graph import (
    clinicalops_graph,
)

from app.schemas.agent import (
    MultiAgentRequest,
    MultiAgentResponse,
)

from app.services.observability_service import (
    observability_service,
)


router = APIRouter(
    prefix="/api/v1/agents",
    tags=[
        "LangGraph Multi-Agent"
    ],
)


# ---------------------------------------------------------
# Platform status
# ---------------------------------------------------------

@router.get(
    "/status"
)
async def agent_status() -> dict:

    return {
        "status": "ready",

        "orchestrator": (
            "LangGraph"
        ),

        "supervisor_model": (
            "qwen3:4b"
        ),

        "agents": [
            "fhir_agent",
            "rag_agent",
            "risk_agent",
            "clinical_nlp_agent",
        ],

        "verifier": True,

        "thread_memory": True,

        "checkpointer": (
            "InMemorySaver"
        ),

        "mcp_transport": (
            "Streamable HTTP"
        ),

        "guardrails": True,

        "observability": {
            "enabled": True,
            "format": "JSONL",
            "clinical_payload_logging": False,
        },
    }


# ---------------------------------------------------------
# Observability summary
# ---------------------------------------------------------

@router.get(
    "/metrics"
)
async def agent_metrics() -> dict:

    return (
        observability_service
        .summary()
    )


# ---------------------------------------------------------
# Recent traces
# ---------------------------------------------------------

@router.get(
    "/traces/recent"
)
async def recent_agent_traces(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
) -> dict:

    traces = (
        observability_service
        .recent(
            limit=limit
        )
    )

    return {
        "count": len(
            traces
        ),
        "traces": traces,
    }


# ---------------------------------------------------------
# Multi-agent execution
# ---------------------------------------------------------

@router.post(
    "/run",
    response_model=MultiAgentResponse,
)
async def run_multi_agent(
    request: MultiAgentRequest,
) -> MultiAgentResponse:

    request_started = (
        time.perf_counter()
    )

    request_id = str(
        uuid4()
    )

    thread_id = (
        request.thread_id
        or str(
            uuid4()
        )
    )


    # -----------------------------------------------------
    # Structured risk payload
    # -----------------------------------------------------

    risk_features = None

    if (
        request.risk_features
        is not None
    ):

        risk_features = (
            request.risk_features
            .model_dump()
        )


    # -----------------------------------------------------
    # Current-turn state
    # -----------------------------------------------------

    initial_state = {
        "request_id": (
            request_id
        ),

        "question": (
            request.question
        ),

        "patient_id": (
            request.patient_id
        ),

        "note": (
            request.note
        ),

        "risk_features": (
            risk_features
        ),

        "conversation_history": [],
    }


    # -----------------------------------------------------
    # LangGraph thread configuration
    # -----------------------------------------------------

    config = {
        "configurable": {
            "thread_id": (
                thread_id
            ),
        }
    }


    # -----------------------------------------------------
    # Execute graph
    # -----------------------------------------------------

    try:

        result = (
            await clinicalops_graph
            .ainvoke(
                initial_state,
                config=config,
            )
        )

    except Exception as error:

        latency_ms = round(
            (
                time.perf_counter()
                - request_started
            )
            * 1000,
            2,
        )

        # -----------------------------------------------
        # Privacy-conscious failure trace
        # -----------------------------------------------

        observability_service.record(
            {
                "request_id": (
                    request_id
                ),

                "thread_id": (
                    thread_id
                ),

                "execution_status": (
                    "error"
                ),

                "route": None,

                "agent_used": None,

                "mcp_tool": None,

                "tool_transport": None,

                "input_guardrail_passed": (
                    None
                ),

                "output_guardrail_passed": (
                    None
                ),

                "guardrail_blocked": (
                    False
                ),

                "verified": None,

                "memory_turns": 0,

                "latency_ms": (
                    latency_ms
                ),

                "has_patient_id": (
                    request.patient_id
                    is not None
                ),

                "has_clinical_note": (
                    request.note
                    is not None
                ),

                "has_risk_features": (
                    request.risk_features
                    is not None
                ),

                "error_type": (
                    type(
                        error
                    )
                    .__name__
                ),
            }
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Multi-agent execution "
                f"failed: {error}"
            ),
        )


    # -----------------------------------------------------
    # Memory
    # -----------------------------------------------------

    conversation_history = (
        result.get(
            "conversation_history",
            [],
        )
    )

    memory_turns = len(
        conversation_history
    )


    # -----------------------------------------------------
    # Agent data
    # -----------------------------------------------------

    agent_data = (
        result.get(
            "agent_result",
            {},
        )
    )


    # -----------------------------------------------------
    # Total end-to-end latency
    # -----------------------------------------------------

    latency_ms = round(
        (
            time.perf_counter()
            - request_started
        )
        * 1000,
        2,
    )


    # -----------------------------------------------------
    # Execution status
    #
    # A guardrail block is not a system failure.
    # It is a successfully enforced policy decision.
    # -----------------------------------------------------

    if result.get(
        "guardrail_blocked",
        False,
    ):

        execution_status = (
            "blocked"
        )

    else:

        execution_status = (
            "success"
        )


    # -----------------------------------------------------
    # Structured operational trace
    #
    # IMPORTANT:
    # No question text, clinical note, FHIR payload,
    # or risk-feature values are written here.
    # -----------------------------------------------------

    observability_service.record(
        {
            "request_id": (
                request_id
            ),

            "thread_id": (
                thread_id
            ),

            "execution_status": (
                execution_status
            ),

            "route": result.get(
                "route"
            ),

            "agent_used": (
                result.get(
                    "agent_used"
                )
            ),

            "mcp_tool": (
                agent_data.get(
                    "mcp_tool"
                )
            ),

            "tool_transport": (
                agent_data.get(
                    "tool_transport"
                )
            ),

            "input_guardrail_passed": (
                result.get(
                    "input_guardrail_passed"
                )
            ),

            "input_guardrail_flags": (
                result.get(
                    "input_guardrail_flags",
                    [],
                )
            ),

            "output_guardrail_passed": (
                result.get(
                    "output_guardrail_passed"
                )
            ),

            "output_guardrail_flags": (
                result.get(
                    "output_guardrail_flags",
                    [],
                )
            ),

            "guardrail_blocked": (
                result.get(
                    "guardrail_blocked",
                    False,
                )
            ),

            "verified": (
                result.get(
                    "verified"
                )
            ),

            "memory_turns": (
                memory_turns
            ),

            "latency_ms": (
                latency_ms
            ),

            "has_patient_id": (
                request.patient_id
                is not None
            ),

            "has_clinical_note": (
                request.note
                is not None
            ),

            "has_risk_features": (
                request.risk_features
                is not None
            ),
        }
    )


    # -----------------------------------------------------
    # API response
    # -----------------------------------------------------

    return MultiAgentResponse(
        request_id=result[
            "request_id"
        ],

        thread_id=(
            thread_id
        ),

        memory_turns=(
            memory_turns
        ),

        active_patient_id=(
            result.get(
                "active_patient_id"
            )
        ),

        route=result[
            "route"
        ],

        route_reason=result[
            "route_reason"
        ],

        agent_used=result[
            "agent_used"
        ],

        verified=result[
            "verified"
        ],

        verification_notes=result[
            "verification_notes"
        ],

        answer=result[
            "final_answer"
        ],

        data=(
            agent_data
        ),

        input_guardrail_passed=(
            result.get(
                "input_guardrail_passed",
                True,
            )
        ),

        input_guardrail_flags=(
            result.get(
                "input_guardrail_flags",
                [],
            )
        ),

        output_guardrail_passed=(
            result.get(
                "output_guardrail_passed"
            )
        ),

        output_guardrail_flags=(
            result.get(
                "output_guardrail_flags",
                [],
            )
        ),

        guardrail_blocked=(
            result.get(
                "guardrail_blocked",
                False,
            )
        ),

        latency_ms=(
            latency_ms
        ),
    )