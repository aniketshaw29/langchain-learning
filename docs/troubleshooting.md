# Troubleshooting

Worked examples can be reproduced by running the matching lesson. General rule
before anything: **make retrieval/plumbing visible** — print what retrieval
returns, print the parsed model output, check the prompt is what you think.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: Missing OPENAI_API_KEY` | no key for a key-hungry provider | add key to `.env`, or `LLM_PROVIDER=ollama` |
| `valueerror unknown provider ...` | typo in `LLM_PROVIDER` / `EMBEDDING_PROVIDER` | check `.env` value; `python -m shared.llm_factory` lists valid names |
| RAG: "no embedding API" | `EMBEDDING_PROVIDER` defaults to a chat-only provider (anthropic/groq) | set `EMBEDDING_PROVIDER=openai` (or google/ollama) |
| `ImportError` / `ModuleNotFoundError` | running `python lessons/01_...py` instead of `-m`, or missing install | run `python -m lessons.01_llm_basics` from project root; `pip install -r requirements.txt` |
| Ollama: connection refused | server not running | `ollama serve`; verify model: `ollama pull llama3.2` |
| Ollama: "model not found" | model not pulled | `ollama pull llama3.2` (embeddings: `ollama pull nomic-embed-text`) |
| RAG answers from stale data | old vectors in `data/chroma/` | delete `data/chroma/` and rerun |
| Model errors "model not found" | default model name changed on vendor side | set `<PROVIDER>_MODEL` in `.env` to your account's model |
| Bad answers despite correct docs | retrieval picks wrong chunks | raise `k`, shrink `chunk_size`, better embeddings, debug via `retriever.invoke(...)` |

## Debugging workflow (in order of leverage)

1. **Inspect the prompt template** — run the template alone:
   `template.invoke({...}).to_messages()` and read exactly what the model gets.
2. **Pry open the RAG pipeline** — does the question find relevant chunks?
   ```python
   for d in retriever.invoke("your question"):
       print(d.page_content[:150], "\n---")
   ```
   Wrong chunks ⇒ fix retrieval, not the prompt.
3. **Check the parser** — is the model replying with JSON and your parser expects
   text (or vice versa)? Swap parsers before touching the prompt.
4. **Reduce randomness** — set `temperature=0.0` to see if answers are
   non-deterministic rather than wrong.
5. **Lower the context** — if output degrades with long histories, trim old turns
   or summarize them ([04 — memory](04-memory.md)).

## FAQ

- **Why must I run `python -m lessons.01_llm_basics`?** The lessons import
  `shared.llm_factory`; running as a module from the project root puts the root
  on `sys.path`, so `shared/` resolves.
- **Why does the model answer "Not in the docs"?** Because the system prompt says
  so — grounding instruction at work. Ask something contained in
  `samples/company_handbook.md`.
- **Chat and embeddings provider must match?** No. Chat with Claude,
  embed with OpenAI. That's the factory's whole design.
- **Do I need an API key for Ollama?** No — that's the point. It runs locally.

See also: [concepts](concepts.md) · [getting started](00-getting-started.md)