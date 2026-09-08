"""
LESSON 01 -- LLM basics: invoke / stream / batch
================================================

WHAT YOU'LL LEARN
  * The single most important object: a Chat Model.
  * The three shapes of a model call: invoke, stream, batch.
  * LangChain's Message types and why they matter.
  * The "mental model": think of the model as a function that maps
      input  ->  AIMessage,  and the message types are just data.

RUN IT
    python -m lessons.01_llm_basics

PREREQ: an .env with a working provider, see .env.example. If you use
Ollama, start it first with `ollama serve`.
"""

# ---------------------------------------------------------------------------
# Import the shared helper -- it hides OpenAI-vs-Ollama behind one function.
# ---------------------------------------------------------------------------
from shared.llm_factory import get_chat_model

# The "brain" of every future lesson. type = BaseChatModel
model = get_chat_model()


# ---------------------------------------------------------------------------
# PART 1 -- The simplest possible call
# ---------------------------------------------------------------------------
def part_1_simple_invoke() -> None:
    """Call the model with a plain string, print the raw .content."""
    # .invoke() returns an AIMessage, NOT a string. Two fields you'll use daily:
    #   message.content -> the actual text answer
    #   message.type    -> the *role* ("human", "ai", "system", "tool"...)
    raw = model.invoke("Hello! In one short sentence, who are you?")
    print("=> PART 1: what comes back")
    print("   object type:", type(raw).__name__)
    print("   .type      :", raw.type)
    print("   .content   :", raw.content, "\n")


# ---------------------------------------------------------------------------
# PART 2 -- Messages: the native language of chat models
# ---------------------------------------------------------------------------
def part_2_messages() -> None:
    """Chat models speak in Message LISTS, not bare strings.

    A conversation is just a list of messages with roles:
        SystemMessage  -- the system/instructor prompt ("you are a pirate chef")
        HumanMessage   -- what the user said (every user turn)
        AIMessage      -- what the model previously said (kept for context)

    Give the model a list and it sees the WHOLE conversation, which is how
    it can remember earlier turns (we exploit this in lesson 04).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content="You are a strict no-emoji editor."),
        HumanMessage(content="Fix this: hey!!! can u help me wit my code :)"),
    ]
    reply = model.invoke(messages)  # same method, but now *conversation-aware*
    print("=> PART 2: prompting with structured messages")
    print("   reply:", reply.content, "\n")


# ---------------------------------------------------------------------------
# PART 3 -- Streaming: watch words arrive instead of waiting for the end
# ---------------------------------------------------------------------------
def part_3_streaming() -> None:
    """For any UX better than a frozen spinner, you stream.

    .stream() yields CHUNKS (AIMessageChunk). Because it's lazy/generator-
    based, you get the FIRST words within a second even on slow prompts.
    Chunk .content shows only the *delta* (new words) -- but the message
    object also accumulates (chunk.accumulated) if you want it.
    """
    print("=> PART 3: streaming tokens as they're generated")
    for chunk in model.stream("Count from 1 to 5, one per line."):
        print(chunk.content, end="", flush=True)  # no newline -> real-time feel
    print("\n")


# ---------------------------------------------------------------------------
# PART 4 -- Batch: many independent prompts in ONE round-trip
# ---------------------------------------------------------------------------
def part_4_batch() -> None:
    """For N independent calls, .batch() is faster than N .invoke()s.

    The provider parallelizes internally where possible. Inputs can be
    strings OR message lists -- anything .invoke accepts.
    """
    prompts = [
        "Name a color. One word.",
        "Name a Python type. One word.",
        "Name a country. One word.",
    ]
    answers = model.batch(prompts)
    print("=> PART 4: batch")
    for i, a in enumerate(answers):
        print(f"   q{i}: {prompts[i]!r:45} -> {a.content}")
    print()


# ---------------------------------------------------------------------------
# PART 5 -- You control the vibes: temperature & friends
# ---------------------------------------------------------------------------
def part_5_tuning() -> None:
    """We made ONE model in llm_factory.py; you can also tune per-use:

        ChatOpenAI(temperature=0.0, max_tokens=50, model=...)

    temperature  : randomness. 0.0 = sampling greedy, near-reproducible;
                   0.8 = "creative" which means *hallucination-prone*.
    max_tokens   : hard cap on reply length (saves money & time).
    """
    from langchain_openai import ChatOpenAI

    model_cold = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    model_hot = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

    question = "Give one word: the best color for a race car is..."
    print("=> PART 5: temperature changes character")
    print("   t=0.0:", model_cold.invoke(question).content)
    print("   t=0.9:", model_hot.invoke(question).content)
    print("   (run it twice -- cold repeats, hot wanders)")


# ---------------------------------------------------------------------------
# main() -- keeps lessons importable for building UIs later (lesson 02 build_*)
# ---------------------------------------------------------------------------
def main() -> None:
    part_1_simple_invoke()
    part_2_messages()
    part_3_streaming()
    part_4_batch()
    # Uncomment if you want to burn a few pennies to see temperature do its thing:
    # part_5_tuning()


if __name__ == "__main__":
    main()