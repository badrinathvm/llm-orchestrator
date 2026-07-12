from langchain_core.messages import ToolMessage
from langgraph.graph import END

from llm.llm import OpenAILLM
from prompts.react_prompts import react_system_prompt
from state.react_state import ReactState
from tools.support_tools import SUPPORT_TOOLS, TOOLS_BY_NAME


def reason(state: ReactState) -> dict:
    llm = OpenAILLM(model="gpt-4o").get_llm()
    if state["iterations"] < state["max_iterations"]:
        llm = llm.bind_tools(SUPPORT_TOOLS)

    chain = react_system_prompt | llm
    response = chain.invoke({"messages": state["messages"]})

    update: dict = {"messages": [response]}
    if not response.tool_calls:
        update["answer"] = response.content
    return update


def act(state: ReactState) -> dict:
    last_message = state["messages"][-1]
    iteration = state["iterations"] + 1

    tool_messages = []
    trace_entries = []
    for call in last_message.tool_calls:
        result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        trace_entries.append({
            "iteration": iteration,
            "thought": last_message.content or "",
            "action": call["name"],
            "action_input": call["args"],
            "observation": str(result),
        })

    return {
        "messages": tool_messages,
        "trace": trace_entries,
        "iterations": iteration,
    }


def should_continue(state: ReactState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "act"
    return END
