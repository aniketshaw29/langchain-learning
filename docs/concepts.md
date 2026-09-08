# Concepts — the LangChain vocabulary you'll actually use

The docs are written assuming you read this file once and then everything else
gets filed against it.

## The three foundational ideas

1. **Everything is a `Runnable`** — something with `.invoke()`, `.stream()`,
   `.batch()`. Prompt templates, models, parsers, retrievers, and whole chains
   are all Runnables. Once you know the interface, every component is familiar.

2. **Chains are pipes (`|`, LCEL)** — output of one Runnable flows into the next:
   `template | model | parser`. Composition without glue code.

3. **Two object types matter most** — `BaseChatModel` (the brain) and
   `Embeddings` (text → vectors, powers search). Integrations are swappable
   because both are interfaces, not concrete classes (see `shared/llm_factory.py`).

## Mini-dictionary

| Term | One-liner | Key params |
|---|---|---|
| Runnable | universal unit with invoke/stream/batch | — |
| LCEL | the `\|` composition syntax | — |
| `ChatPromptTemplate` | `[("role", "text with {ph}")]` → filled messages | placeholder names |
| `StrOutputParser` | strips AIMessage to text | — |
| `RunnableParallel` | run pipes side by side, merge into dict | branch names |
| `RunnablePassthrough` | pass input through unchanged | — |
| `RunnableLambda` | wrap any python fn for the pipe | fn |
| `Document` | `page_content` + `metadata` | metadata is filterable |
| `splitter` | chunks a doc | `chunk_size`, `chunk_overlap` |
| `Embeddings` | text → vector | model name |
| vector store | stores vectors, searches by similarity | persist dir |
| `retriever` | query → top-k `Document`s (a Runnable!) | `k` |
| `@tool` | decorator: fn the model can call | docstring = its description |
| agent | model + tools + loop | tools, checkpointer |
| `MemorySaver` | in-memory checkpointer for agent memory | — |
| `thread_id` | namespaced conversation in checkpointer | — |
| hallucination | confident false output | fight with grounding + RAG |

## The one diagram

```
                    ┌──────────── memory (messages) ────────────┐
                    │                                           v
 question ─> retriever ─> {context} ─> prompt ─> model ─> parser ─> answer
                    │                                           ^
                    └────────────── tools (agents) ─────────────┘
```

RAG feeds the model **your documents** (retrieval). Agents let it **do things**
(tools). Memory persists the **conversation**. All three reduce to: *get more of
the right information into the message list.*

## Choosing your path through the docs

- New to it all → [00 getting started](00-getting-started.md) → 01 → 02 → 03 → 04 → 05.
- Building "chat with my docs" → 01, 02, 03.
- Building a tool-calling assistant → 01, 02, 05.
- Stuck? → [troubleshooting](troubleshooting.md)