from typing import List

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    route: str
    route_reason: str


class WorkerResult(BaseModel):
    worker: str
    instruction: str
    output: str


class OrchestrationResponse(BaseModel):
    answer: str
    workers_used: List[str]
    plan: dict
    worker_results: List[WorkerResult]


class AgentStep(BaseModel):
    iteration: int
    thought: str
    action: str
    action_input: dict
    observation: str


class AgentResponse(BaseModel):
    answer: str
    iterations: int
    tools_used: List[str]
    trace: List[AgentStep]
