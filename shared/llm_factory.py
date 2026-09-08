"""
llm_factory.py -- one helper to rule them all
==============================================

Swap OpenAI <-> Claude <-> Gemini (etc) by changing ONE line in `.env`.
This file is the *integration layer* of a real app: it hides which vendor
backs your prompts so every lesson stays exactly the same.

THE BIG IDEA YOU'LL REUSE EVERYWHERE:
    LangChain speaks in TYPES, not vendors. Anything that takes a message
    in and returns a message out can be swapped for `ChatOpenAI`, `ChatAnthropic`,
    `ChatGoogleGenerativeAI`, ... -- your code doesn't change.

    Type              | Interface                         | You saw it in
    -------------------+-----------------------------------+-----------------
    Chat model        | BaseChatModel (invoke/stream/batch)| every lesson
    Embeddings        | Embeddings (embed_query/...)      | lesson 03

HOW THE REGISTRY WORKS (the modular part):
    Below there's a dict: provider name -> factory function.
    - to ADD a provider: write one tiny factory fn that returns a ChatModel
      and register it in CHAT_MODEL_FACTORIES.
    - to USE it: set LLM_PROVIDER=<that name> in .env.
    No other code in the whole repo needs to change. That's the payoff.

CURRENT PROVIDERS (see .env.example for the full config):
    openai     feeds the GPT family                 needs OPENAI_API_KEY
    anthropic  Claude                               needs ANTHROPIC_API_KEY
    google     Gemini                               needs GOOGLE_API_KEY
    groq       super-fast open models via Groq      needs GROQ_API_KEY
    ollama     your own local models                needs nothing but `ollama serve`

Run a provider status check with:
    python -m shared.llm_factory
"""

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

# Loads your .env file (if it exists) into os.environ so the rest of the
# app can just read os.getenv(...) everywhere.
load_dotenv()


# ---------------------------------------------------------------------------
# CHAT MODEL FACTORIES -- one small function per vendor (lazy imports = fast)
# ---------------------------------------------------------------------------
# Default model names change over time; if you see "model not found" errors,
# set <PROVIDER>_MODEL in .env to a model available on your account today.

def _openai_chat() -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.7)


def _anthropic_chat() -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"), temperature=0.7)


def _google_chat() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"), temperature=0.7)


def _groq_chat() -> BaseChatModel:
    # Groq lets you run open-weights models (Llama, Qwen...) extremely fast.
    from langchain_groq import ChatGroq
    return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.7)


def _ollama_chat() -> BaseChatModel:
    from langchain_ollama import ChatOllama
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                      base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                      temperature=0.7)


# The registry. Provider name -> factory. Adding a provider = adding a row.
CHAT_MODEL_FACTORIES = {
    "openai": _openai_chat,
    "anthropic": _anthropic_chat,
    "google": _google_chat,
    "groq": _groq_chat,
    "ollama": _ollama_chat,
}

# Which API key each provider needs (for friendly errors + the status check).
PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": None,  # local -- no key needed
}


# ---------------------------------------------------------------------------
# Public entry points used by every lesson
# ---------------------------------------------------------------------------

def get_chat_model(provider: str | None = None) -> BaseChatModel:
    """Return a ChatModel for a provider (default: from LLM_PROVIDER env var).

    Returns a `BaseChatModel` -- the *interface* every chat integration
    implements. Both branches return the same type, so every lesson can just
    import this and call `.invoke()`, `.stream()`, `.batch()`.
    """
    provider = _resolve("LLM_PROVIDER", provider)

    if provider not in CHAT_MODEL_FACTORIES:
        raise ValueError(
            f"Unknown provider '{provider}'. Pick from: {', '.join(CHAT_MODEL_FACTORIES)}"
        )

    _require_key(provider)
    return CHAT_MODEL_FACTORIES[provider]()


def get_embeddings(provider: str | None = None) -> Embeddings:
    """Return an embeddings model for vector search (lesson 03).

    NOTE: not every chat provider *has* embeddings. Anthropic and Groq
    both lack an embedding API, so you must set EMBEDDING_PROVIDER=openai
    (or =google / =ollama) in .env. Chat and embeddings don't have to match:
    e.g. chat with Claude, embed locally.
    """
    provider = _resolve("EMBEDDING_PROVIDER", provider)

    if provider == "openai":
        _require("OPENAI_API_KEY")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    if provider == "google":
        _require("GOOGLE_API_KEY")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text",
                                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. Try 'openai', 'google' or 'ollama' "
        f"(Anthropic/Groq don't offer embeddings)."
    )


# ---------------------------------------------------------------------------
# Helpers (nothing below matters for the lessons -- purely quality of life)
# ---------------------------------------------------------------------------

def _resolve(env_var: str, explicit: str | None) -> str:
    """Pick the provider: explicit arg wins, else the env var, else 'openai'."""
    value = (explicit or os.getenv(env_var, "")).strip().lower()
    return value or os.getenv("LLM_PROVIDER", "openai").strip().lower() or "openai"


def _require_key(provider: str) -> None:
    """Fail with a clear message instead of a wall of stack traces."""
    key_env = PROVIDER_KEY_ENV.get(provider)
    if key_env and not os.getenv(key_env):
        raise RuntimeError(
            f"Missing {key_env} for provider '{provider}'. Put it in your .env "
            f"file (see .env.example), switch providers, or run `ollama serve`."
        )


def _require(env_var: str) -> None:
    if not os.getenv(env_var):
        raise RuntimeError(
            f"Missing {env_var}. Put it in your .env file (see .env.example)."
        )


if __name__ == "__main__":
    # Provider status check:  python -m shared.llm_factory
    print(f"{'provider':<10} {'key set?':<9} status")
    print("-" * 40)
    for name in CHAT_MODEL_FACTORIES:
        key_env = PROVIDER_KEY_ENV[name]
        has_key = (key_env is None) or bool(os.getenv(key_env))
        status = "ready" if has_key else "no key in .env"
        print(f"{name:<10} {'yes' if has_key else 'no':<9} {status}")

    print(f"\nActive chat model : {get_chat_model().__class__.__module__}.{get_chat_model().__class__.__name__}")