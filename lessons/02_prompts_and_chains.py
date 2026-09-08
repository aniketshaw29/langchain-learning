"""
LESSON 02 -- Prompts, Output Parsers, and Chains (LCEL)
=======================================================

WHAT YOU'LL LEARN
  * Prompt Templates: reusable blueprints that turn variables into messages.
  * Output parsers: force the model's *text* into a *shape* you want.
  * LCEL (LangChain Expression Language): the `|` pipe. This is THE core
    idea of modern LangChain -- everything is a Runnable.
  * RunnableParallel: fan-out and combine.

RUN IT
    python -m lessons.02_prompts_and_chains

READ ME FIRST -- the mental model of LCEL:
    In LCEL you compose pieces with the pipe `|`. Data flows left to right:

        inputs --> prompt --> model --> parser --> outputs

    A "Runnable" is something with .invoke() / .stream() / .batch().
    Prompt templates, models, parsers, and whole chains are ALL runnables,
    so you can pipe what you built right into something else.

    THIS is why lesson 01's stream/batch worked with zero extra code and
    why lesson 03's RAG chain reuses the same plumbing.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

from shared.llm_factory import get_chat_model

model = get_chat_model()

# ---------------------------------------------------------------------------
# PART 1 -- Prompt Templates: params in, formatted messages out
# ---------------------------------------------------------------------------
def part_1_prompt_template() -> None:
    """A ChatPromptTemplate is a *live* spec, not a string.

    Two positional forms used everywhere:
        ("role", "text with {placeholders}")
        or a Message class directly.

    Inspect it BEFORE running a model to de-mystify what the model actually
    receives (this is how I debug everything).
    """
    jingle_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You write catchy jingles of exactly {tone} tone. "
                "Keep replies under 2 lines.",
            ),
            ("human", "Write a jingle about {thing}."),
        ]
    )

    # .invoke returns the FILLED-IN list of messages -- no model needed.
    filled = jingle_template.invoke({"tone": "very serious like a bank advert", "thing": "a toaster"})
    print("=> PART 1: what a prompt template builds")
    for msg in filled.to_messages():
        print(f"   {msg.type:>5}: {msg.content!r}")
    print()


# ---------------------------------------------------------------------------
# PART 2 -- Output Parser: sift text from billions of tokens
# ---------------------------------------------------------------------------
def part_2_output_parser() -> None:
    """A model returns an AIMessage; a parser extracts what you want.

    StrOutputParser returns just the text (.content). It's the most common
    parser because most pipelines then feed the text to the NEXT step.
    """
    parser = StrOutputParser()

    template = ChatPromptTemplate.from_messages(
        [("human", "Give me a one-sentence pitch for {topic}.")]
    )

    # Build a chain: template | model | parser
    chain = template | model | parser

    # .invoke now returns a STRING, not an AIMessage -- already parsed.
    answer = chain.invoke({"topic": "a flying umbrella"})
    print("=> PART 2: a real pipeline with 3 runnables")
    print("   type:", type(answer).__name__, "| value:", answer)
    print("   chain type:", type(chain).__name__, "\n")


# ---------------------------------------------------------------------------
# PART 3 -- THE BIG ONE: fan-out with RunnableParallel
# ---------------------------------------------------------------------------
def part_3_parallel_and_custom() -> None:
    """RunnableParallel = run several pipes independently, merge results.

       Highly reusable pattern: get context (retriever), get a fresh take
       (another model), then combine into the final prompt.

       We also show a RunnableLambda: turn any python function into a runnable
       so it can sit inside a pipe (lessons 03/04 depend on this trick).
    """
    from langchain_core.runnables import RunnableLambda

    # Two INPUT specs, run against the same user "{topic}".
    #   - the `fact_chain` returns text; `emojis` returns text
    pipeline = RunnableParallel(
        fact_chain=ChatPromptTemplate.from_messages(
            [("human", "State one factual-sounding fact about {topic}.")]
        )
        | model
        | StrOutputParser(),
        emojis=ChatPromptTemplate.from_messages(
            [("human", "Return exactly three emojis for {topic}.")]
        )
        | model
        | StrOutputParser(),
    )

    fuser = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a news editor. Compose an exciting headline from "
                "these inputs.",
            ),
            ("human", "Fact: {fact_chain}\nEmojis: {emojis}"),
        ]
    )

    # A tiny helper: any function becomes a runnable with RunnableLambda.
    def shout(text: str) -> str:
        return text.upper() + "!!1!"  # old-school internet

    chain = pipeline | fuser | model | StrOutputParser() | RunnableLambda(shout)

    out = chain.invoke({"topic": "bees"})
    print("=> PART 3: RunnableParallel + RunnableLambda")
    print("   headline:", out)


# ---------------------------------------------------------------------------
# PART 4 -- Stream the WHOLE chain (yes, LCEL streams through pipes)
# ---------------------------------------------------------------------------
def part_4_streaming_chain() -> None:
    """Any LCEL pipe streams end-to-end automatically because every link is
    a Runnable. Build a chain once, then pick invoke/stream/batch --
    no rewrites, that's the win.
    """
    chain = (
        ChatPromptTemplate.from_messages(
            [("human", "Write a haiku about {subject}.")]
        )
        | model
        | StrOutputParser()
    )

    print("=> PART 4: streaming a full chain")
    for chunk in chain.stream({"subject": "a very large potato"}):
        print(chunk, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# main() -- also the pattern used by later lessons (each defines build_chain)
# ---------------------------------------------------------------------------
def build_chain() -> RunnableParallel:
    """Exported so a later lesson (or REPL) can reuse the full pipeline:
    build_chain().invoke({...})
    """
    return RunnableParallel(
        fact_chain=ChatPromptTemplate.from_messages(
            [("human", "State one factual-sounding fact about {topic}.")]
        )
        | model
        | StrOutputParser(),
        emojis=ChatPromptTemplate.from_messages(
            [("human", "Return exactly three emojis for {topic}.")]
        )
        | model
        | StrOutputParser(),
    )


def main() -> None:
    part_1_prompt_template()
    part_2_output_parser()
    part_3_parallel_and_custom()
    part_4_streaming_chain()


if __name__ == "__main__":
    main()