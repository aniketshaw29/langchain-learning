# Learning LangChain (Python) — a beginner's project track

This repo is a **hands-on, beginner-first pathway** into modern LangChain.
Each lesson is a runnable Python file, densely commented like a tutorial, and
later lessons build on earlier ones. Read the docstring at the top of each file
first — it's the "cheat sheet" for that lesson.

```
lessons/
  01_llm_basics.py          the model: invoke / stream / batch, Messages
  02_prompts_and_chains.py  prompt templates, output parsers, LCEL pipes, fan-out
  03_rag.py                 Retrieval-Augmented Generation over your own docs
  04_memory.py              making a model remember (message history, by hand)
  05_agents_tools.py        agents + tools with LangGraph (it can *do* things)
samples/company_handbook.md sample document for lesson 03 to answer questions about
shared/llm_factory.py       one helper that swaps OpenAI <-> Ollama (read this first)
```

## The one mental model (read this before anything)

LangChain is a **plumbing kit for LLMs**. Its entire core can be summarized as:

1. **Everything is a `Runnable`** — an object with `.invoke()`, `.stream()`,
   `.batch()`. Prompt templates, models, parsers, retrievers, and whole chains
   are all Runnables.

2. **Chains are built with the pipe `|`** — data flows left to right:
   ```
   user input
     v
   prompt ──> model ──> output_parser ──> final answer
   ```
   Each `|` just feeds one Runnable's output into the next one's input. That's
   **LCEL** (LangChain Expression Language), and it's the 80% of LangChain you
   will use daily.

3. **Two types of objects matter most**:
   - **ChatModel** (`BaseChatModel`) — the "brain". OpenAI, Ollama, Anthropic, etc.
   - **Embeddings** — turns text into vectors, powers "similarity search".

Keep that triangle in your head and every lesson below is just one Runnable
piped after another with a new idea dropped in the middle.

## Setup (5 minutes)

```bash
# 1. make a virtual env (Python 3.10+ recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install deps
pip install -r requirements.txt

# 3. pick a model provider -- either is fine
cp .env.example .env

#   Option A – OpenAI (fastest to get going, costs pennies)
#     put your key in OPENAI_API_KEY=

#   Option B – Ollama (free, runs locally; great for tinkering offline)
#     brew install ollama && ollama pull llama3.2
#     and set LLM_PROVIDER=ollama in .env
```

Run a lesson from the **project root** (that's how the imports work):

```bash
python -m lessons.01_llm_basics
```

## The curriculum

> **Lesson 01 — `invoke` / `stream` / `batch` + Messages**
> You meet the ChatModel and the three call shapes. You learn the core data
> type (`SystemMessage` / `HumanMessage` / `AIMessage`) and why a conversation
> is just a *list of messages*.
>
> *Do first*: `python -m lessons.01_llm_basics`. Then open `shared/llm_factory.py`
> — that's the pattern for "swap providers by config, never by code".

> **Lesson 02 — Prompts, parsers, and LCEL**
> The most important file in the repo. `ChatPromptTemplate` turns variables into
> messages; `StrOutputParser` turns an AIMessage into text; `|` glues them into a
> chain. You also see `RunnableParallel` (run two pipes side-by-side and merge)
> and `RunnableLambda` (wrap any Python function into a pipe).
>
> *Key idea*: `template | model | parser` is *already* a full LLM app, and the
> same object works with `.invoke()`, `.stream()`, and `.batch()`.

> **Lesson 03 — RAG (your own documents)**
> The serverless pattern behind most "chat with your docs" products:
>
> ```
> load (TextLoader) -> split (RecursiveCharacterTextSplitter)
>     -> embed (OpenaiEmbeddings/OllamaEmbeddings) -> store (Chroma)
>     -> retrieve top-k -> pipe context + question into prompt -> answer
> ```
>
> Try: change a fact in `samples/company_handbook.md`, then ask the bot about it.
> With RAG its answer *changes*; that's the whole point.

> **Lesson 04 — Memory**
> Models are stateless — "memory" just means *you* keep the old messages and
> re-send them. Build a tiny REPL that remembers, using messages by hand. This
> cements the mental model before we let a framework do it for us.

> **Lesson 05 — Agents + tools (LangGraph)**
> The modern agent standard. A model + a few `@tool` functions + a loop
> (`create_react_agent`) = a model that can *do* things (compute, fetch status,
> mutate data), not just parrot. `MemorySaver` + a `thread_id` gives it
> conversation memory across turns, for free.
>
> Swap `get_weather` / `multiply_numbers` for tools that hit a real API or your
> database, and you've met-practically production architecture.

## Glossary — the words you'll keep seeing

| Term | Meaning |
|---|---|
| Runnable | Anything with `.invoke()/.stream()/.batch()`. The universal interface. |
| LCEL | LangChain Expression Language; the `\|` composition syntax. |
| ChatPromptTemplate | Blueprint: `("role", "text with {placeholders}")` → filled messages. |
| Output parser | Wrings the text (or JSON, or a list) out of the model's raw reply. |
| Document | LangChain's file wrapper (`page_content` + `metadata`). |
| Embedding | A vector that captures meaning; similar texts → nearby vectors. |
| Vector store | DB that stores embeddings and retrieves by similarity (Chroma here). |
| Retriever | A Runnable whose input is a question and output is the *top-k* docs. |
| Tool | A `@tool`-decorated function the model can call. Its docstring is its "help text". |
| Agent | Loop: reason → maybe call a tool → read result → decide when to stop. |
| Checkpointer | Persists an agent's message history per `thread_id`. |
| Hallucination | Confident-but-false model output — RAG + grounded prompts fight this. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `RuntimeError: Missing OPENAI_API_KEY` | Put a key in `.env`, or set `LLM_PROVIDER=ollama`. |
| Ollama connection refused | `ollama serve`, then `ollama pull llama3.2` / `ollama pull nomic-embed-text`. |
| Wrong/old answers in lesson 03 | Delete `data/chroma/` and rerun (stale chunks). |
| Imports break (`python lessons/01_llm_basics.py`) | Always run as `python -m lessons.01_llm_basics` from the project root. |
| Model answer doesn't match the handbook | Ask something *in* the handbook; retrieval is nearest-similarity, not exact. |

## Practice ideas (do at least two)

1. **Tune lesson 02** — add a FactMap where two parallel pipes disagree, then a
   `RunnableLambda` that "votes".
2. **RAG on real files** — point lesson 03 at your own `.md`/`.pdf` docs
   (`PyPDFLoader` from `langchain-community`). Print the retrieved context and
   check it's actually relevant.
3. **Memory 2.0** — instead of a raw history, run a summary Runnable over old
   turns and inject the summary (lesson 02's lambdas make this a few lines).
4. **Real tools** — add a `datetime` tool and a file-writer tool to lesson 05
   and ask the agent to "write a dated log file".

## Where to go next

- LangChain + LangGraph official docs: `python.langchain.com`, `langchain-ai.github.io/langgraph`
- langchain-core is the best package to read source of — it's tiny and well-documented.
- Build a real app: Gradio/Streamlit around lesson 03 is the classic weekend project.