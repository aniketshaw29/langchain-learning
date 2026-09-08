"""
LESSON 05 -- Agents + Tools with LangGraph
==========================================

WHAT YOU'LL LEARN
  * Tools: python functions the model can *choose* to call.
  * Agents: a loop where the model reasons -> calls a tool -> reads the
    result -> reasons again ... until it decides it's done.
  * The modern way: LangGraph's prebuilt ReAct agent (react = reason+act).
  * Built-in memory: a checkpointer gives the agent recall across turns.

RUN IT
    python -m lessons.05_agents_tools
    (asks for a location + a theme -- our tool calls run LOCALLY, no network
     needed beyond the LLM itself)

WHY DISTINCT FROM LESSON 03?  RAG retrieves YOUR docs. Agents *do* things:
compute, fetch, mutate. Same model wrapper, robots instead of copy-paste.
"""

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from shared.llm_factory import get_chat_model

model = get_chat_model()


# ---------------------------------------------------------------------------
# TOOLS -- @tool turns any function into a callable the model can invoke.
# ---------------------------------------------------------------------------
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a city. Fake data, real behavior:
    the model reads HOW you spoke to derive meaning and echo it back."""
    # NOTE: the docstring IS the tool description the model sees. If the model
    # can't tell when to call you, you wrote a bad docstring.
    temps = {"portland": "12°C drizzle", "austin": "33°C hot pepper", "cork": "15°C windy"}
    return f"{location}: " + temps.get(location.lower().split()[0], "weather unknown, try a real city")


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two integers. Use this for ANY arithmetic multiplication."""
    return a * b


# ---------------------------------------------------------------------------
# THE AGENT -- reasoning + tools + loop, in one line
# ---------------------------------------------------------------------------
# create_react_agent(model, tools) builds a full diagram:
#   [model] <--(needs a tool?)---> [call tool] ---> [model]  loop
# and keeps looping until the model stops asking for tools.
# MemorySaver() = in-memory checkpointer; pass a thread_id per conversation
# and the agent restores that conversation next time.
checkpointer = MemorySaver()
agent = create_react_agent(model, [get_weather, multiply_numbers], checkpointer=checkpointer)


def stream_agent_trace(prompt: str, thread_id: str = "demo-1") -> None:
    """Run one user turn and show the *steps* the agent took.

    agent.stream(..., stream_mode="updates") yields frames like:
        {"agent":  {...thinking...}}
        {"tools":  {tool_name: tool_result}}
        {"agent":  {...final answer...}}
    Watching these frames is how you *see* the reasoning loop happening.
    """
    print("  ---------- agent steps ----------")
    config = {"configurable": {"thread_id": thread_id}}
    for step in agent.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, payload in step.items():
            if node_name == "tools":
                for tool_name, tool_out in payload.items():
                    print(f"    [tool] {tool_name}({tool_out})")
            else:
                # last message of this agent frame = its current speech
                msg = payload["messages"][-1]
                if isinstance(msg.content, str) and msg.content.strip():
                    print(f"    [agent] {msg.content}")
    print("  ---------- end steps ----------")


def show_memory() -> None:
    """Ask a NEW question that only makes sense given the FIRST conversation.
    The checkpointer + thread_id = the agent 'remembers' turn 1."""
    prompt = "What was the last city we checked the weather for? And multiply 6 by 7."
    print("  Follow-up (relies on memory):")
    stream_agent_trace(prompt, thread_id="demo-1")


def main() -> None:
    thread_id = "demo-1"

    print("=> turn 1: knowledge + tool use mixed in one prompt")
    stream_agent_trace(
        "What's the weather in Portland, and what is 8 * 7?",
        thread_id=thread_id,
    )
    print("\n=> turn 2: same thread, agent recalls turn 1")
    show_memory()


if __name__ == "__main__":
    main()