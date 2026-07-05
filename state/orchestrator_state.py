import operator
from typing import Annotated, List

from typing_extensions import TypedDict


class OrchestratorState(TypedDict):
    query: str
    plan: dict
    context: Annotated[List[dict], operator.add]
    worker_results: Annotated[List[dict], operator.add]
    answer: str
