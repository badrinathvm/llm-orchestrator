from langchain_core.prompts import ChatPromptTemplate

router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a routing classifier for an LLM orchestration system. "
     "Given a user query, decide which model tier should handle it:\n"
     "- 'small': simple, short, low-stakes queries (greetings, basic facts, quick lookups) "
     "that a fast/cheap model can answer well.\n"
     "- 'frontier': complex queries that need deep reasoning, multi-step analysis, "
     "or careful judgement.\n"
     "- 'specialist': queries about customer support, triage, troubleshooting, or "
     "account/product issues that benefit from a domain-specific support persona.\n"
     "Pick exactly one route and give a one-sentence reason for your choice."),
    ("human", "{query}"),
])

specialist_system_prompt = (
    "You are a senior customer support triage specialist. You read incoming queries, "
    "diagnose the underlying issue, and respond with clear, empathetic, actionable guidance. "
    "When relevant, note how the issue should be prioritized or escalated."
)

specialist_prompt = ChatPromptTemplate.from_messages([
    ("system", specialist_system_prompt),
    ("human", "{query}"),
])
