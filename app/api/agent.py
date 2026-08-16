from uuid import uuid4

from fastapi import (
    APIRouter,
    HTTPException,
)

from app.agents.graph import (
    clinicalops_graph,
)

from app.schemas.agent import (
    MultiAgentRequest,
    MultiAgentResponse,
)


router = APIRouter(
    prefix="/api/v1/agents",
    tags=[
        "LangGraph Multi-Agent"
    ],
)


# ---------------------------------------------------------
# Status
# ---------------------------------------------------------

@router.get(
    "/status"
)
async def agent_status() -> dict:

    return {
        "status": "ready",
        "orchestrator": "LangGraph",
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

    request_id = str(
        uuid4()
    )

    thread_id = (
        request.thread_id
        or str(
            uuid4()
        )
    )

    risk_features = None

    if (
        request.risk_features
        is not None
    ):

        risk_features = (
            request.risk_features
            .model_dump()
        )

    # --------------------------------------------------
    # Current-turn input
    # --------------------------------------------------

    initial_state = {
        "request_id": request_id,
        "question": request.question,

        # Explicit None values intentionally clear
        # stale per-turn payloads while persistent
        # memory such as active_patient_id remains.

        "patient_id": (
            request.patient_id
        ),
        "note": (
            request.note
        ),
        "risk_features": (
            risk_features
        ),

        # With the list reducer, [] does not erase
        # checkpointed conversation history.

        "conversation_history": [],
    }

    # --------------------------------------------------
    # LangGraph thread configuration
    # --------------------------------------------------

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    try:

        result = (
            await clinicalops_graph
            .ainvoke(
                initial_state,
                config=config,
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Multi-agent execution "
                f"failed: {error}"
            ),
        )

    conversation_history = (
        result.get(
            "conversation_history",
            [],
        )
    )

    return MultiAgentResponse(
        request_id=result[
            "request_id"
        ],
        thread_id=thread_id,
        memory_turns=len(
            conversation_history
        ),
        active_patient_id=result.get(
            "active_patient_id"
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
        data=result[
            "agent_result"
        ],
    )