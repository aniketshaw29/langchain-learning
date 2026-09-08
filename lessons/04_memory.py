"""
LESSON 04 -- Memory: making the model remember the conversation
===============================================================

WHAT YOU'LL LEARN
  * Why a chat model can't remember anything by itself (statelessness).
  * The ONLY mechanism that matters: the message LIST you send it.
  * Build a tiny stateful REPL by appending every turn to a history list.
  * The production-grade answer exists (LangGraph checkpoints) -- we touch
    that in lesson 05; here we build the concept with raw parts.

RUN IT
    python -m lessons.04_memory
    (it opens an interactive chat. Type 'quit' to stop.)

THE KEY INSIGHT
    LLMs are functions, not stateful apps. Every .invoke() is a fresh brain.
    "Memory" = *you* keep truck of old messages and re-send them.
    More messages = more tokens = larger context = more "memory".

    Rules of thumb
      * keep the list small (trim old turns / summarize) for cost + latency
      * older turns should be summarized, not poured in verbatim
"""

from shared.llm_factory import get_chat_model

model = get_chat_model()


# ---------------------------------------------------------------------------
# The whole "memory" is this list. That's it.
# ---------------------------------------------------------------------------
def simple_chat() -> None:
    """A REPL loop where history is accumulated on EVERY exchange.

    1. we append the user message
    2. invoke with [system] + all prior turns
    3. append the model's reply
    4. next turn, the model "remembers" because it saw the transcript
    """
    history = []  # our running memory. list[BaseMessage]

    print("cli-chat online (type 'quit' to exit)")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break

        from langchain_core.messages import AIMessage, HumanMessage

        history.append(HumanMessage(content=user_text))

        # system prompt stays at slot 0, our growing transcript after it
        from langchain_core.messages import SystemMessage

        messages = [SystemMessage(content="You are a helpful, terse assistant.")] + history

        reply = model.invoke(messages)
        print("Bot:", reply.content)

        history.append(AIMessage(content=reply.content))

        # Show the memory in action on turn 3+ (subtle "remember that" winners)
        if len(history) >= 4 and user_text.lower().startswith("what did"):
            print("  (note the model could only know that from the transcript)")


# ---------------------------------------------------------------------------
# VARIANT: same thing, but message dicts ({role, content}) instead of classes.
# Chat models accept EITHER. The dict form is what LangGraph uses internally.
# ---------------------------------------------------------------------------
def dict_form_chat() -> None:
    """Same memory trick, but history stored as plain dicts.

    Useful because it's JSON-shaped (easy to save/load/route). Both styles
    are first-class in LangChain.
    """
    history: list[dict] = []
    print("\n--> dict-form chat online (try: remember my favorite number)")
    try:
        while True:
            user_text = input("You: ").strip()
            if user_text.lower() in {"quit", "exit"}:
                break
            history.append({"role": "user", "content": user_text})
            reply = model.invoke(history)  # send the whole transcript
            history.append({"role": "assistant", "content": reply.content})
            print("Bot:", reply.content)
    except KeyboardInterrupt:
        print("\n(bye)")


# ---------------------------------------------------------------------------
# REAL-WORLD NOTE
# ---------------------------------------------------------------------------
def the_future() -> None:
    """Production memory = LangGraph checkpoints, lesson 05.

    Long-term memory lives OUTSIDE the prompt: a StateGraph persists the
    whole message history to disk/sqlite, restores it per thread_id, and
    sends only what the model needs. The mental model above still holds:
    the model only ever sees a message list -- the framework just maintains it.
    """
    print("=> see lesson 05 for checkpoints + agents with built-in memory")


def main() -> None:
    simple_chat()
    # Uncomment to see the dict variant in action:
    # dict_form_chat()
    the_future()


if __name__ == "__main__":
    main()