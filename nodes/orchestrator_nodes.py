from typing import Optional

from pydantic import BaseModel, Field

from llm.llm import OpenAILLM
from prompts.orchestrator_prompts import (
    analysis_worker_prompt,
    code_worker_prompt,
    orchestrator_prompt,
    research_worker_prompt,
    synthesis_prompt,
)
from state.orchestrator_state import OrchestratorState

WORKER_NODE_NAMES = ["research_worker", "code_worker", "analysis_worker"]


class OrchestrationPlan(BaseModel):
    research: Optional[str] = Field(
        default=None, description="Tailored instruction for the research worker, or null if not needed"
    )
    code: Optional[str] = Field(
        default=None, description="Tailored instruction for the code worker, or null if not needed"
    )
    analysis: Optional[str] = Field(
        default=None, description="Tailored instruction for the analysis worker, or null if not needed"
    )


def plan_tasks(state: OrchestratorState) -> dict:
    query = state["query"]
    llm = OpenAILLM().get_llm().with_structured_output(OrchestrationPlan)
    chain = orchestrator_prompt | llm
    plan = chain.invoke({"query": query})
    plan_dict = plan.model_dump()

    if not any(plan_dict.values()):
        plan_dict["research"] = query

    return {
        "plan": plan_dict,
        "context": [{"node": "plan_tasks", "note": f"Dispatch plan: {plan_dict}"}],
    }


def route_to_workers(state: OrchestratorState) -> list:
    plan = state["plan"]
    dest = []
    if plan.get("research"):
        dest.append("research_worker")
    if plan.get("code"):
        dest.append("code_worker")
    if plan.get("analysis"):
        dest.append("analysis_worker")
    return dest


def _run_worker(worker_name: str, prompt, state: OrchestratorState) -> dict:
    instruction = state["plan"][worker_name]
    llm = OpenAILLM(model="gpt-4o").get_llm()
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "instruction": instruction})
    return {
        "worker_results": [{"worker": worker_name, "instruction": instruction, "output": response.content}],
        "context": [{"node": worker_name, "note": f"Completed: {instruction[:120]}"}],
    }


def research_worker(state: OrchestratorState) -> dict:
    return _run_worker("research", research_worker_prompt, state)


def code_worker(state: OrchestratorState) -> dict:
    return _run_worker("code", code_worker_prompt, state)


def analysis_worker(state: OrchestratorState) -> dict:
    return _run_worker("analysis", analysis_worker_prompt, state)


def synthesize(state: OrchestratorState) -> dict:
    query = state["query"]
    worker_results = state["worker_results"]
    formatted = "\n\n".join(
        f"### {r['worker'].title()} Worker\nInstruction: {r['instruction']}\nOutput:\n{r['output']}"
        for r in worker_results
    )
    llm = OpenAILLM(model="gpt-4o").get_llm()
    chain = synthesis_prompt | llm
    response = chain.invoke({"query": query, "worker_outputs": formatted})
    return {"answer": response.content}
