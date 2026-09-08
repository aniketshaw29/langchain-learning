# 03 — RAG: answering from YOUR documents

Worked source: [`lessons/03_rag.py`](../lessons/03_rag.py)

## The problem

A base LLM knows only its training data. Ask it about your company handbook and
it improvises — confidently wrong ("hallucination"). RAG fixes this by making the
model *look things up before it answers*.

## The pipeline (the whole story on one line)

```
load → split → embed → store → retrieve(question) → prompt{context+question} → model
```

Each step is a library call; here they are with the imports you'll actually use:

| Step | Code | Notes |
|---|---|---|
| 1. Load | `TextLoader(path)` | PDFs → `PyPDFLoader`; URLs → `WebBaseLoader` |
| 2. Split | `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)` | respect context limits |
| 3. Embed | `get_embeddings()` (factory) | turns text → vector |
| 4. Store | `Chroma.from_documents(...)` | vector DB, persists to disk |
| 5. Retrieve | `vs.as_retriever(search_kwargs={"k": 3})` | *also a Runnable* |
| 6. Generate | an LCEL chain | context + question → model |

## Why chunk?

- Model contexts are finite — you can't paste a 600-page manual.
- A 400-char chunk about "benefits" has sharper meaning than the full doc.
- `chunk_overlap` keeps sentences from being cut mid-thought.

## The final chain — read it like a recipe

```python
retriever = vs.as_retriever(search_kwargs={"k": 3})

def format_docs(docs) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)

chain = (
    {
        "context": retriever | RunnableLambda(format_docs), # find + format
        "question": RunnablePassthrough(),                  # pass input through
    }
    | ChatPromptTemplate.from_messages([
        ("system",
         "Answer ONLY from the context. If absent, say 'Not in the docs.'\n\n{context}"),
        ("human", "Question: {question}"),
    ])
    | model
    | StrOutputParser()
)
```

- `retriever` is invoked inside the pipe because a Retriever *is* a Runnable.
- `RunnablePassthrough` just passes `question` through untouched — needed because
  the prompt wants both `context` and `question`.
- That system prompt is a **grounding instruction**: it is your main tool against
  hallucination. Always say what to do when the answer *isn't* in context.

## Debugging RAG (the skill that separates "works" from "vibes")

Always print what retrieval actually returns before blaming the model:

```python
for doc in retriever.invoke("How many vacation days?")[:3]:
    print(doc.page_content[:200], "\n---")
```

If the chunks are irrelevant, fix *retrieval*, not the prompt: raise `k`, lower
`chunk_size`, or use a better embeddings model.

## Storage note

`Chroma.from_documents(..., persist_directory=...)` writes to disk. Delete that
folder if you change source documents, or stale vectors will keep matching.

## Practice

1. Point lesson 03 at your own files (any `.md`/`.txt`).
2. Print retrieved chunks for a question that has no answer — does the prompt
   correctly say "Not in the docs"?
3. Set `k=1` and ask a multi-fact question — see the answer degrade.

---

**Next:** [04 — memory](04-memory.md)