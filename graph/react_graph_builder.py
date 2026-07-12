from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from nodes.react_nodes import act, reason, should_continue
from state.react_state import ReactState


class ReactGraphBuilder:
    def __init__(self) -> None:
        self.workflow = StateGraph(ReactState)

    def build_graph(self):
        self.workflow.add_node("reason", reason)
        self.workflow.add_node("act", act)

        self.workflow.add_edge(START, "reason")
        self.workflow.add_conditional_edges("reason", should_continue, {"act": "act", END: END})
        self.workflow.add_edge("act", "reason")
        return self

    def run(self, query: str, max_iterations: int = 6) -> dict:
        return self.build_graph().workflow.compile().invoke({
            "query": query,
            "messages": [HumanMessage(content=query)],
            "trace": [],
            "iterations": 0,
            "max_iterations": max_iterations,
            "answer": "",
        })
