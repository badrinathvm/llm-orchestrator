from typing import Literal

from pydantic import BaseModel, Field

from state.orchestration_state import OrchestrationState
from llm.llm import OpenAILLM
from llm.embedding_router import get_embedding_router
from prompts.prompt import router_prompt, specialist_prompt


class RouteDecision(BaseModel):
    route: Literal["small", "frontier", "specialist"] = Field(
        description="Which model tier should handle this query"
    )
    reason: str = Field(description="One-sentence justification for the chosen route")


def route_query(state: OrchestrationState) -> dict:
    query = state["query"]

    try:
        route, score = get_embedding_router().classify(query)
    except Exception:
        route, score = None, None

    if route is not None:
        return {"route": route, "route_reason": f"Embedding similarity match (score={score:.2f})"}

    llm = OpenAILLM().get_llm().with_structured_output(RouteDecision)
    chain = router_prompt | llm
    decision = chain.invoke({"query": query})
    return {"route": decision.route, "route_reason": f"LLM fallback: {decision.reason}"}


def small_model_responder(state: OrchestrationState) -> dict:
    query = state["query"]
    llm = OpenAILLM(model="gpt-4o-mini").get_llm()
    response = llm.invoke(query)
    return {"result": response.content}


def frontier_model_responder(state: OrchestrationState) -> dict:
    query = state["query"]
    llm = OpenAILLM(model="gpt-4o").get_llm()
    response = llm.invoke(query)
    return {"result": response.content}


def specialist_responder(state: OrchestrationState) -> dict:
    query = state["query"]
    llm = OpenAILLM(model="gpt-4o").get_llm()
    chain = specialist_prompt | llm
    response = chain.invoke({"query": query})
    return {"result": response.content}
