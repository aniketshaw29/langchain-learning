"""
llm_factory.py -- one helper to rule them all
==============================================

The single most annoying part of learning LangChain is switching between
providers. This module hides that diff so every lesson can just ask for
"a chat model" and "some embeddings" without caring who backs them.

THE BIG IDEA YOU'LL REUSE EVERYWHERE:
    LangChain speaks in TYPES, not vendors. Anything that takes a message
    in and returns a message out can be swapped for `ChatOpenAI` or
    `ChatOllama` or any of ~100 integrations -- your code doesn't change.

    Type              | Interface                         | You saw it in
    -------------------+-----------------------------------+-----------------
    Chat model        | BaseChatModel (invoke/stream/batch)| every lesson
    Embeddings        | Embeddings (embed_query/...)      | lesson 03

CURRENT CONFIG (from your .env file):
    LLM_PROVIDER        "openai" or "ollama"
    EMBEDDING_PROVIDER  "openai" or "ollama"
    OPENAI_API_KEY      required only for "openai"
    OLLAMA_MODEL        default local model, e.g. llama3.2
    OLLAMA_BASE_URL     where your local Ollama server lives
"""

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

# Loads your .env file (if it exists) into os.environ so the rest of the
# app can just read os.getenv(...) everywhere.
load_dotenv()

# ---------------------------------------------------------------------------
# Which model do we use for CHAT? (the "brain")
# ---------------------------------------------------------------------------

def get_chat_model() -> BaseChatModel:
    """Return a chat model selected by LLM_PROVIDER.

    Returns a `BaseChatModel` -- the *interface* every chat integration
    implements. Because both branches return the same type, every lesson can
    import this one function and call `.invoke()`, `.stream()`, `.batch()`.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "openai":
        _require("OPENAI_API_KEY", "set LLM_PROVIDER=ollama if you don't have a key")
        from langchain_openai import ChatOpenAI  # lazy import = faster startup

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),  # cheap + fast
            temperature=0.7,     # 0 = deterministic, ~1 = chaotic. tune for taste
            max_retries=2,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama  # local models, runs mostly offline

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.7,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'openai' or 'ollama'.")


# ---------------------------------------------------------------------------
# Which model do we use for EMBEDDINGS? (turns text into vectors, lesson 03)
# ---------------------------------------------------------------------------

def get_embeddings(provider: str | None = None) -> Embeddings:
    """Return an embeddings model for vector search.

    `provider` defaults to EMBEDDING_PROVIDER, which defaults to the chat
    provider -- but they don't have to match. e.g. chat with OpenAI, embed
    locally with Ollama.
    """
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "")).strip().lower()
    if not provider:
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "openai":
        _require("OPENAI_API_KEY", "set EMBEDDING_PROVIDER=ollama to go offline")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    raise ValueError(f"Unknown EMBEDDING_PROVIDER '{provider}'. Use 'openai' or 'ollama'.")


# ---------------------------------------------------------------------------
# Small validator so errors are friendly instead of a wall of stack traces
# ---------------------------------------------------------------------------

def _require(env_var: str, hint: str) -> None:
    """Raise a clear error when a required env var is missing."""
    if not os.getenv(env_var):
        raise RuntimeError(
            f"Missing {env_var}. Put it in your .env file (see .env.example), "
            f"or {hint}."
        )


if __name__ == "__main__":
    # Quick sanity check:  python -m shared.llm_factory
    model = get_chat_model()
    print(f"Chat model     : {type(model).__name__} ({model.__class__.__module__})")
    print(f"  backs OpenAI key? {bool(os.getenv('OPENAI_API_KEY'))}")