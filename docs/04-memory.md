# 04 — Memory: how models "remember"

Worked source: [`lessons/04_memory.py`](../lessons/04_memory.py)

## The core truth

Models are **stateless**. Every `.invoke()` is a fresh brain. There is no memory
in the model — memory is *you* keeping the old messages and re-sending them.

```
memory (your app)  ── keeps growing list of messages ──
                            |
                            v
        [sys] [H: "hi"] [A: "hello!"] [H: "what did I say"?] ---> model
```

## The minimal working version

```python
history = []

while True:
    user = input("You: ")
    history.append({"role": "user", "content": user})

    reply = model.invoke(
        [{"role": "system", "content": "You are helpful."}] + history
    )
    print("Bot:", reply.content)

    history.append({"role": "assistant", "content": reply.content})
```

That's the whole mechanism. The same thing with message classes:

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

history: list = [SystemMessage(content="You are helpful.")]
history.append(HumanMessage(content="hi"))
reply = model.invoke(history)
history.append(AIMessage(content=reply.content))
```

## Cost & quality levers (this is where engineers earn their pay)

- **More history = more tokens = slower + costlier.** Trim old turns.
- **Fade old turns**: summarize the early part of a long conversation and keep
  only the last N raw turns (a summary Runnable is easy with lesson 02's pipes):

```python
summary = summary_template | model | StrOutputParser()

if len(history) > 12:
    history = [
        SystemMessage(content=f"Earlier summary: {summary.invoke({'msgs': history[:-8]})}")
    ] + history[-8:]
```

- **Windowed memory**: keep the last N turns, period. Simple, predictable.

## The production answer: LangGraph checkpoints

For real apps you don't hand-roll history — `MemorySaver` + a `thread_id` does
it, including save/restore per conversation:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools, checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "user-42"}}

agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)
agent.invoke({"messages": [{"role": "user", "content": "what did I say?"}]}, config)  # remembers
```

Covered hands-on in [05 — agents and tools](05-agents-and-tools.md).

## Three rules

1. The model only ever sees a message list — memory is where that list lives.
2. Memory is bounded; closing the loop is summarizing, not accumulating forever.
3. Use a checkpointer (LangGraph) for persistence; hand-rolled lists for teaching.

---

**Next:** [05 — agents and tools](05-agents-and-tools.md)