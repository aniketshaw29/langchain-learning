# Getting Started with LangChain (Python)

A quick-run reference. The full picture lives in the repo `README.md`; this doc is
the "I just cloned it, walk me through" version.

## 1. What you're installing

| Package | Role |
|---|---|
| `langchain` + `langchain-core` | the framework + the `Runnable`/LCEL core |
| `langchain-<vendor>` | one integration per provider (`openai`, `anthropic`, `google`, `ollama`, `groq`) |
| `langchain-chroma` + `chromadb` | local vector store used for RAG |
| `langgraph` | modern agent + checkpoint framework |
| `python-dotenv` | loads your `.env` file |

## 2. One-time setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 3. Pick a model provider (the only real decision)

Edit `.env` — the module `shared/llm_factory.py` is the single integration point.

```ini
LLM_PROVIDER=openai        # or: anthropic | google | groq | ollama

# you only need the key of your chosen provider:
OPENAI_API_KEY=...
# ANTHROPIC_API_KEY=...    # Claude
# GOOGLE_API_KEY=...       # Gemini
# GROQ_API_KEY=...         # fast open models
# OLLAMA_MODEL=llama3.2    # local free models (run: ollama serve)
```

Verify it without writing code:

```bash
python -m shared.llm_factory
```

## 4. Run a lesson

Lessons must be run as *modules* from the project root so shared imports resolve:

```bash
python -m lessons.01_llm_basics
python -m lessons.02_prompts_and_chains
python -m lessons.03_rag
python -m lessons.04_memory
python -m lessons.05_agents_tools
```

> Yes — `python lessons/01_llm_basics.py` breaks the import. Run as `python -m`.

## 5. If something looks off

1. `data/chroma/` old vectors? delete it and rerun lesson 03.
2. "Missing OPENAI_API_KEY" but you chose Ollama? — `LLM_PROVIDER=ollama`.
3. RAG says "no embedding API" — Anthropic/Groq can't embed; set `EMBEDDING_PROVIDER=openai`.
4. See `docs/troubleshooting.md` for the full table.

---

**Next:** [01 — models and messages](01-models-and-messages.md) · or jump straight to
the [concept map](concepts.md) if you already know the pieces.