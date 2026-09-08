# 02 — Prompts, Output Parsers, and LCEL

Worked source: [`lessons/02_prompts_and_chains.py`](../lessons/02_prompts_and_chains.py)

This is the heart of the framework. Master the three pieces and 80% of LangChain
is familiar.

## The mental model: everything is a Runnable, chains are pipes

**Runnable** = something with `.invoke() / .stream() / .batch()`. Prompt
templates, models, parsers, retrievers, whole chains — all Runnables.

**LCEL** = connect Runnables with `|`. Data flows left to right:

```python
chain = template | model | parser

chain.invoke({"thing": "a toaster"})   # one call through the whole pipe
chain.stream({...})                    # still streams every pipe in the chain
chain.batch([{...}, {...}])            # still batches. no rewriting.
```

This is why lesson 01's three call shapes "just work" everywhere.

## 1. Prompt templates — live blueprints, not strings

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You write {tone} jingles, max 2 lines."),   # (role, text)
    ("human", "Write a jingle about {thing}."),              # {placeholders}
])
```

- Uses `{placeholders}` and a `dict` of values at call time.
- `.invoke({"tone": "...", "thing": "..."})` returns **filled messages** — no model
  needed. Inspect it to debug what the model actually sees.
- Always use a system message for behavior; the human message for the request.

## 2. Output parsers — shape from noise

`StrOutputParser` (the workhorse) just returns the text:

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
chain = template | model | parser
result: str = chain.invoke({...})       # str, not AIMessage
```

When you need structure, swap in `JsonOutputParser`, `PydanticOutputParser`,
`CommaSeparatedListOutputParser` — same pipe, different right-hand side.

## 3. Fan-out + custom code in the pipe

**`RunnableParallel`** runs several pipes against the same input and merges
their outputs into one dict:

```python
from langchain_core.runnables import RunnableParallel

branch = RunnableParallel(
    fact=fact_template | model | parser,   # returns str
    emojis=emoji_template | model | parser,
)

chain = branch | fuser_template | model | parser
chain.invoke({"topic": "bees"})   # each branch sees {"topic": "bees"}
```

**`RunnableLambda`** wraps any Python function so it can sit mid-pipe:

```python
from langchain_core.runnables import RunnableLambda

def uppercase(text: str) -> str:
    return text.upper()

chain = prompt | model | parser | RunnableLambda(uppercase)
```

## Reusable pattern: build once, expose the chain

Structure lessons like this so a chain can be imported and reused:

```python
def build_chain():
    return prompt | model | parser        # call .invoke/.stream/.batch later
```

## Practice

1. Chain that takes `{name}` and produces "a fun fact about {name}" using a
   *system* message for tone.
2. Two branches in a `RunnableParallel`: `short` and `long` versions — fuse them
   into a final prompt.
3. Pipe a `RunnableLambda` that appends `[generated]` after the parser.

---

**Next:** [03 — RAG](03-rag.md)