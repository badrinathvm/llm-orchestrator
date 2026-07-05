from langgraph.graph import END, START, StateGraph

from nodes.orchestrator_nodes import (
    WORKER_NODE_NAMES,
    analysis_worker,
    code_worker,
    plan_tasks,
    research_worker,
    route_to_workers,
    synthesize,
)
from state.orchestrator_state import OrchestratorState


class OrchestratorGraphBuilder:
    def __init__(self) -> None:
        self.workflow = StateGraph(OrchestratorState)

    def build_graph(self):
        self.workflow.add_node("plan_tasks", plan_tasks)
        self.workflow.add_node("research_worker", research_worker)
        self.workflow.add_node("code_worker", code_worker)
        self.workflow.add_node("analysis_worker", analysis_worker)
        self.workflow.add_node("synthesize", synthesize)

        self.workflow.add_edge(START, "plan_tasks")
        self.workflow.add_conditional_edges("plan_tasks", route_to_workers, WORKER_NODE_NAMES)
        self.workflow.add_edge("research_worker", "synthesize")
        self.workflow.add_edge("code_worker", "synthesize")
        self.workflow.add_edge("analysis_worker", "synthesize")
        self.workflow.add_edge("synthesize", END)
        return self

    def run(self, query: str) -> dict:
        return self.build_graph().workflow.compile().invoke({
            "query": query,
            "plan": {},
            "context": [],
            "worker_results": [],
            "answer": "",
        })
