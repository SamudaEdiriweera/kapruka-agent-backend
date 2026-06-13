"""
What this file does
This is the wiring of the entire agent. 
It connects all 5 nodes together and defines the conditional routing logic — which node runs next based on the current state.

Why we need it
Without the graph, the nodes are just isolated functions. LangGraph needs this to know:

. Where to start
. Which node runs after which
. When to loop back vs when to end
"""

from langgraph.graph import StateGraph, END
from agent.state import ShoppingState
from agent.nodes import (
    detector_node,
    planner_node,
    reasoner_node,
    tool_caller_node,
    reflector_node,
    responder_node
)

# ------- ROUTERS -------------------

def route_reasoner(state: ShoppingState) -> str:
    """
    After reasoner decides what to do:
    - call_tool -> go to tool_caller
    - clarify / respond -> got to responder
    """
    action = state.get("_decision", {}).get("action", "respond")  
    if action == "call_tool":
        return "tool_caller"
    return "responder"

def route_reflector(state: ShoppingState) -> str:
    """
    After tool_call + reflection:
    - contine -> back to reasoner for next step
    - replan -> back to planner
    - respond -> straight to responder
    """
    status = state.get("reflection", "respond")
    if status == "continue":
        return "reasoner"
    if status == "replan":
        return "planner"
    return "responder"

# ------- GRAPH -----------------------

def build_graph():
    graph = StateGraph(ShoppingState)

    # Register all nodes
    graph.add_node("detector", detector_node)
    graph.add_node("planner", planner_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("tool_caller", tool_caller_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("responder", responder_node)

    # Entry point
    graph.set_entry_point("detector")

    # Fixed edges
    graph.add_edge("detector", "planner")
    graph.add_edge("planner", "reasoner")
    graph.add_edge("tool_caller", "reflector")
    graph.add_edge("responder", END)

    # Conditional edges
    graph.add_conditional_edges(
        "reasoner",
        route_reasoner,
        {
            "tool_caller": "reasoner",
            "planner": "planner",
            "responder": "responder"
        }
    )

    graph.add_conditional_edges(
        "reflector",
        route_reflector,
        {
            "reasoner": "reasoner",
            "planner": "planner",
            "responder": "responder"
        }
    )

    return graph.compile()

# Complied graph - imported by routes.py
graph = build_graph()

