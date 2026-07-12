import operator
from typing import Annotated, List

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ReactState(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    trace: Annotated[List[dict], operator.add]
    iterations: int
    max_iterations: int
    answer: str
