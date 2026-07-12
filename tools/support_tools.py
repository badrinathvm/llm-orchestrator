import os

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

RETURN_WINDOW_DAYS = 30
HIGH_VALUE_THRESHOLD = 200.00

_MOCK_ORDERS = {
    "A1001": {
        "item": "Wireless Headphones",
        "total": 49.99,
        "status": "delivered",
        "days_since_delivery": 3,
        "damaged_reported": False,
    },
    "A1002": {
        "item": "Ceramic Coffee Mug Set",
        "total": 34.50,
        "status": "delivered",
        "days_since_delivery": 3,
        "damaged_reported": True,
    },
    "A1003": {
        "item": "Running Shoes",
        "total": 89.00,
        "status": "delivered",
        "days_since_delivery": 42,
        "damaged_reported": False,
    },
    "A1004": {
        "item": "Bluetooth Speaker",
        "total": 59.99,
        "status": "in_transit",
        "days_since_delivery": None,
        "damaged_reported": False,
    },
    "A1005": {
        "item": "Laptop Stand",
        "total": 250.00,
        "status": "delivered",
        "days_since_delivery": 10,
        "damaged_reported": False,
    },
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order's status, item, total, and delivery info by order id."""
    order = _MOCK_ORDERS.get(order_id.upper())
    if not order:
        return f"No order found with id '{order_id}'."

    delivery = (
        f"delivered {order['days_since_delivery']} day(s) ago"
        if order["status"] == "delivered"
        else f"status: {order['status']}"
    )
    return (
        f"Order {order_id}: {order['item']}, total ${order['total']:.2f}, "
        f"{delivery}, damage reported: {order['damaged_reported']}."
    )


@tool
def search_policy(query: str) -> str:
    """Search the live web for return, refund, or shipping policy information relevant to the query."""
    if not os.getenv("TAVILY_API_KEY"):
        return "search_policy is unavailable: TAVILY_API_KEY is not configured."

    try:
        search = TavilySearch(max_results=3, include_answer=True)
        result = search.invoke({"query": query})
    except Exception as e:
        return f"search_policy failed: {e}"

    parts = []
    if result.get("answer"):
        parts.append(f"Summary: {result['answer']}")
    for r in result.get("results", []):
        parts.append(f"- {r['title']} ({r['url']}): {r['content'][:300]}")

    return "\n".join(parts) if parts else f"No policy information found for '{query}'."


@tool
def calculate_refund(order_id: str, reason: str) -> str:
    """Calculate the refund amount and rationale for an order given the customer's stated reason."""
    order = _MOCK_ORDERS.get(order_id.upper())
    if not order:
        return f"Cannot calculate a refund: no order found with id '{order_id}'."

    if order["status"] != "delivered":
        return (
            f"Order {order_id} has not been delivered yet (status: {order['status']}); "
            "no refund can be calculated until it arrives."
        )

    if order["total"] > HIGH_VALUE_THRESHOLD:
        return (
            f"Order {order_id} total (${order['total']:.2f}) exceeds the ${HIGH_VALUE_THRESHOLD:.2f} "
            "self-service refund limit; this requires human review — use escalate_to_human."
        )

    damaged = order["damaged_reported"] or any(k in reason.lower() for k in ("damaged", "defective", "broken"))
    within_window = order["days_since_delivery"] is not None and order["days_since_delivery"] <= RETURN_WINDOW_DAYS

    if damaged:
        return f"Approved: full refund of ${order['total']:.2f} for order {order_id} (item reported damaged/defective)."
    if within_window:
        return (
            f"Approved: full refund of ${order['total']:.2f} for order {order_id} "
            f"(within the {RETURN_WINDOW_DAYS}-day return window)."
        )
    return (
        f"Order {order_id} is past the {RETURN_WINDOW_DAYS}-day return window and not reported damaged; "
        f"offer ${order['total'] / 2:.2f} in store credit instead of a cash refund."
    )


@tool
def escalate_to_human(order_id: str, reason: str) -> str:
    """Hand an order off to a human support agent when the case is ambiguous or outside self-service policy."""
    ticket_id = f"TCK-{abs(hash((order_id, reason))) % 100000:05d}"
    return f"Escalated order {order_id} to a human agent (ticket {ticket_id}). Reason: {reason}"


SUPPORT_TOOLS = [lookup_order, search_policy, calculate_refund, escalate_to_human]
TOOLS_BY_NAME = {t.name: t for t in SUPPORT_TOOLS}
