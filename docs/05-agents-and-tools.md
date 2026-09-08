# 05 — Agents and Tools (LangGraph)

Worked source: [`lessons/05_agents_tools.py`](../lessons/05_agents_tools.py)

## Chat vs. chat-with-tools vs. agent

| Kind | Can it call functions? | Precision |
|---|---|---|
| plain model | no | text in, text out |
| `model.bind_tools(tools)` | yes, but **you** must write the loop | you control everything |
| agent (LangGraph prebuilt) | yes, and it loops for you | batteries included |

## Tools: functions the model can choose to call

A `@tool`-decorated function becomes a callable the model discovers automatically.
**Its docstring is its help text** — write it from the model's perspective.

```python
from langchain_core.tools import tool

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two integers. Use this for ANY arithmetic multiplication."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"{city}: 12°C drizzle"
```

Rules of thumb for good tools: descriptive docstring, typed params (the model
reads the signature), one concern per tool.

## Agents: reason → act → observe → repeat

`create_react_agent(model, tools)` builds the whole loop ("react" = reason+act):

```
user ─> model ─(I need a tool)─> call tool ─(ToolMessage)─> model
              └─(answer ready)─────────────────────────────┘ stop
```

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, [get_weather, multiply_numbers])
result = agent.invoke({"messages": [{"role": "user", "content": "8 * 7?"}]})
print(result["messages"][-1].content)   # the final answer
```

## Seeing the agent think (stream)

`stream_mode="updates"` gives you a frame per node (`agent` vs `tools`) — watch
the plan, the tool, and the response form:

```python
for step in agent.stream(
    {"messages": [{"role": "user", "content": "weather in Portland and 8 * 7?"}]},
    stream_mode="updates",
):
    for node, payload in step.items():
        if node == "tools":
            for tool_name, out in payload.items():
                print(f"[tool] {tool_name} -> {out}")
        else:
            print(f"[agent] {payload['messages'][-1].content}")
```

## Built-in memory via checkpoints

One extra argument and a `thread_id` turns the agent into a persistent per-user
conversation:

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_react_agent(model, [tools], checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "user-42"}}

agent.invoke({"messages": [{"role": "user", "content": "weather in Austin?"}]}, config)
agent.invoke({"messages": [{"role": "user", "content": "what's the temp there?"}]}, config)  # remembers
```

Swap `MemorySaver` for `SqliteSaver`/`PostgresSaver` and conversations survive
restarts — that's the production path.

## Why agents matter

RAG retrieves; agents **act**. The same wrapped model can compute, pull from an
API, write a file, then answer. Your first real agent = replace
`get_weather`/`multiply_numbers` with tools that hit a real API or database.

## Practice

1. Add a `fastmath` tool that uses the `datetime` library.
2. Give the agent a tool that writes to disk, then ask it to "save today's date".
3. Stream with `stream_mode="updates"` and identify where tools vs. model act.

---

Back to [README](../README.md) · [concepts](concepts.md) · [troubleshooting](troubleshooting.md)