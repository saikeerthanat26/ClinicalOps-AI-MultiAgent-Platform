from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.agents.context import (
    context_node,
)

from app.agents.specialists import (
    fhir_agent_node,
    nlp_agent_node,
    rag_agent_node,
    risk_agent_node,
)

from app.agents.state import (
    ClinicalOpsAgentState,
)

from app.agents.supervisor import (
    supervisor_node,
)

from app.agents.verifier import (
    verifier_node,
)

from app.guardrails.input_guardrail import (
    blocked_request_node,
    input_guardrail_node,
    route_after_input_guardrail,
)

from app.guardrails.output_guardrail import (
    output_guardrail_node,
)


# ---------------------------------------------------------
# Supervisor routing
# ---------------------------------------------------------

def route_after_supervisor(
    state: ClinicalOpsAgentState,
) -> str:

    return state[
        "route"
    ]


# ---------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------

checkpointer = (
    InMemorySaver()
)


# ---------------------------------------------------------
# Graph
# ---------------------------------------------------------

builder = StateGraph(
    ClinicalOpsAgentState
)


# ---------------------------------------------------------
# Nodes
# ---------------------------------------------------------

builder.add_node(
    "input_guardrail",
    input_guardrail_node,
)

builder.add_node(
    "blocked_request",
    blocked_request_node,
)

builder.add_node(
    "context",
    context_node,
)

builder.add_node(
    "supervisor",
    supervisor_node,
)

builder.add_node(
    "fhir_agent",
    fhir_agent_node,
)

builder.add_node(
    "rag_agent",
    rag_agent_node,
)

builder.add_node(
    "risk_agent",
    risk_agent_node,
)

builder.add_node(
    "nlp_agent",
    nlp_agent_node,
)

builder.add_node(
    "verifier",
    verifier_node,
)

builder.add_node(
    "output_guardrail",
    output_guardrail_node,
)


# ---------------------------------------------------------
# START → Input Guardrail
# ---------------------------------------------------------

builder.add_edge(
    START,
    "input_guardrail",
)


# ---------------------------------------------------------
# Input guardrail routing
# ---------------------------------------------------------

builder.add_conditional_edges(
    "input_guardrail",
    route_after_input_guardrail,
    {
        "continue": "context",
        "blocked": "blocked_request",
    },
)


# ---------------------------------------------------------
# Blocked request → END
# ---------------------------------------------------------

builder.add_edge(
    "blocked_request",
    END,
)


# ---------------------------------------------------------
# Allowed request
# Context → Supervisor
# ---------------------------------------------------------

builder.add_edge(
    "context",
    "supervisor",
)


# ---------------------------------------------------------
# Supervisor → Specialist
# ---------------------------------------------------------

builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "fhir": "fhir_agent",
        "rag": "rag_agent",
        "risk": "risk_agent",
        "nlp": "nlp_agent",
    },
)


# ---------------------------------------------------------
# Specialists → Verifier
# ---------------------------------------------------------

builder.add_edge(
    "fhir_agent",
    "verifier",
)

builder.add_edge(
    "rag_agent",
    "verifier",
)

builder.add_edge(
    "risk_agent",
    "verifier",
)

builder.add_edge(
    "nlp_agent",
    "verifier",
)


# ---------------------------------------------------------
# Verifier → Output Guardrail → END
# ---------------------------------------------------------

builder.add_edge(
    "verifier",
    "output_guardrail",
)

builder.add_edge(
    "output_guardrail",
    END,
)


# ---------------------------------------------------------
# Compile
# ---------------------------------------------------------

clinicalops_graph = (
    builder.compile(
        checkpointer=checkpointer
    )
)