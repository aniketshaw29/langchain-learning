# 01 — Models and Messages

Worked source: [`lessons/01_llm_basics.py`](../lessons/01_llm_basics.py)

## The model is a function

A chat model is best thought of as:

```
  (list of messages)  -->  [chat model]  -->  AIMessage
```

Not stateful, not a database — a pure function of its input. Everything else in
LangChain is about making that function's input smarter.

## The three call shapes

All identical to use, all free with LCEL (no code rewrite when you compose chains):

| Method | Returns | Use when |
|---|---|---|
| `.invoke(x)` | `AIMessage` | you want one answer |
| `.stream(x)` | generator of `AIMessageChunk` | you want tokens as they arrive |
| `.batch([...])` | `list[AIMessage]` | you have many *independent* prompts |

```python
from shared.llm_factory import get_chat_model

model = get_chat_model()

answer = model.invoke("2 + 2?")        # starts... waits... returns
for c in model.stream("2 + 2?"):       # first token within ~1s
    print(c.content, end="", flush=True)
answers = model.batch(["a?", "b?", "c?"])  # parallelized, N in one shot
```

`.invoke()` returns an **`AIMessage`**, not a `str`. Get the text with `.content`,
the role with `.type`.

## Messages: the native data type

A conversation is a **list of messages with roles**:

| Type | Meaning |
|---|---|
| `SystemMessage` | instruction/persona — slot 0 |
| `HumanMessage` | user's turn |
| `AIMessage` | model's prior turn (its *content* is what the model said) |
| `ToolMessage` | result of a tool call (lesson 05) |

```python
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

model.invoke([
    SystemMessage(content="You are a terse pirate."),
    HumanMessage(content="Greet me."),
])
# 'Aye, ahoy there!'

# dict form is equally valid and JSON-friendly:
model.invoke([
    {"role": "system", "content": "You are a terse pirate."},
    {"role": "user", "content": "Greet me."},
])
```

Memory is *just this list growing* — see [04 — memory](04-memory.md).

## Temperature and other dials

Two knobs you'll touch daily:

- `temperature` — `0.0` = greedy/near-reproducible; `~0.9` = creative/hallucination-prone.
  `0.7` is a sane default.
- `max_tokens` — hard cap on reply length (saves money + latency).

```python
from langchain_openai import ChatOpenAI

cold = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=100)
```

## The takeaway to internalize

- `invoke / stream / batch` are just three views of the same Runnable.
- The "conversation" you keep hearing about is a list of messages with roles.
- Choose the provider once (`shared/llm_factory.py`); never import a vendor in
  your app code.

---

**Next:** [02 — prompts, parsers, and LCEL](02-prompts-and-chains.md)