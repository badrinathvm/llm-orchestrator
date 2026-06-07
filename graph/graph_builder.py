from langgraph.graph import StateGraph, START, END
from nodes.nodes import (
    route_query,
    small_model_responder,
    frontier_model_responder,
    specialist_responder,
)
from state.orchestration_state import OrchestrationState


class GraphBuilder:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.workflow = StateGraph(OrchestrationState)

    def build_graph(self):
        self.workflow.add_node("router", route_query)
        self.workflow.add_node("small_model_responder", small_model_responder)
        self.workflow.add_node("frontier_model_responder", frontier_model_responder)
        self.workflow.add_node("specialist_responder", specialist_responder)

        self.workflow.add_edge(START, "router")
        self.workflow.add_conditional_edges(
            "router",
            lambda state: state["route"],
            {
                "small": "small_model_responder",
                "frontier": "frontier_model_responder",
                "specialist": "specialist_responder",
            },
        )
        self.workflow.add_edge("small_model_responder", END)
        self.workflow.add_edge("frontier_model_responder", END)
        self.workflow.add_edge("specialist_responder", END)
        return self

    def run(self, query: str) -> dict:
        return self.build_graph().workflow.compile().invoke({
            "query": query,
            "context": [],
            "route": "",
            "route_reason": "",
            "result": "",
        })
