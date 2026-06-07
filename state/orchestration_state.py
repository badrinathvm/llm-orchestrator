from typing_extensions import TypedDict, List


class OrchestrationState(TypedDict):
    query: str
    context: List[dict]
    route: str          # "small" | "frontier" | "specialist"
    route_reason: str
    result: str
