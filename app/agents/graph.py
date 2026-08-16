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
    AgentRoute,
    ClinicalOpsAgentState,
)

from app.agents.supervisor import (
    supervisor_node,
)

from app.agents.verifier import (
    verifier_node,
)


# ---------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------

def route_after_supervisor(
    state: ClinicalOpsAgentState,
) -> AgentRoute:

    return state[
        "route"
    ]


# ---------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------

checkpointer = InMemorySaver()


# ---------------------------------------------------------
# Graph
# ---------------------------------------------------------

builder = StateGraph(
    ClinicalOpsAgentState
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


# ---------------------------------------------------------
# START → Context → Supervisor
# ---------------------------------------------------------

builder.add_edge(
    START,
    "context",
)

builder.add_edge(
    "context",
    "supervisor",
)


# ---------------------------------------------------------
# Supervisor → specialist
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
# Specialist → Verifier
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
# Verifier → END
# ---------------------------------------------------------

builder.add_edge(
    "verifier",
    END,
)


# ---------------------------------------------------------
# Compile with thread-scoped memory
# ---------------------------------------------------------

clinicalops_graph = (
    builder.compile(
        checkpointer=checkpointer
    )
)