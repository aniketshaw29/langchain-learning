"""
LESSON 03 -- RAG: give your LLM YOUR OWN documents
==================================================

WHAT YOU'LL LEARN
  * The problem: an LLM only knows its TRAINING data. Ask about your company
    handbook and it will invent (hallucinate) an answer.
  * RAG (Retrieval-Augmented Generation) fixes that:
        1. LOAD      your documents on disk
        2. SPLIT     them into chunks (context windows have size limits)
        3. EMBED     each chunk into a vector (semantic coordinates)
        4. STORE     vectors in a vector DB (Chroma)
        5. RETRIEVE  the top-k chunks most similar to the user question
        6. GENERATE  answer from {retrieved context + question}
  * The full picture as ONE LCEL chain (lesson 02's `|` pipes again).

RUN IT
    python -m lessons.03_rag
    (needs an embeddings backend: openai key OR `ollama pull nomic-embed-text`)

IF YOU WANT PROOF IT WORKS: change a fact in samples/company_handbook.md,
then ask about it. With RAG the answer changes; without it, it won't.
"""

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.llm_factory import get_chat_model, get_embeddings

# ---------------------------------------------------------------------------
# CONFIG -- change these, not the code
# ---------------------------------------------------------------------------
DOC_PATH = "samples/company_handbook.md"      # our pretend company's docs
DB_DIR = "data/chroma/"                        # vector DB persists here
CHUNK_SIZE = 400                               # ~400 chars per chunk
CHUNK_OVERLAP = 50                             # keep 50 chars of overlap
TOP_K = 3                                      # how many chunks to retrieve

model = get_chat_model()


# ---------------------------------------------------------------------------
# STEP 1+2 -- Load & split
# ---------------------------------------------------------------------------
def load_and_split() -> list:
    """TextLoader reads the file; the splitter chops it into chunks.

    Why chunk at all? (a) model context is finite, (b) a chunk about
    "benefits" is more semantically coherent than a whole manual.
    Overlap ensures sentence boundaries don't lose meaning mid-split.
    """
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    docs = loader.load()  # returns a list of langchain Document objects

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)  # -> list[Document]


# ---------------------------------------------------------------------------
# STEP 3+4 -- Embed & store (creates data/chroma on first run)
# ---------------------------------------------------------------------------
def build_vectorstore(chunks: list) -> Chroma:
    """Embed every chunk and save vectors to disk.

    Rerun and it creates the DB fresh; vector DBs are additive but we just
    rebuild for learning. DELETE data/chroma if you change the source doc
    (stale chunks would otherwise linger).
    """
    embeddings = get_embeddings()
    return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)


# ---------------------------------------------------------------------------
# STEP 5 -- The retriever (a Runnable! it fits in your pipes)
# ---------------------------------------------------------------------------
def get_retriever():
    """as_retriever turns the vector store into a searchable object.

    When later invoked it takes a QUERY string and returns the TOP_K most
    similar Documents. 'similar' = cosine distance of embeddings. The whole
    retriever is a Runnable so it plugs into LCEL.
    """
    vs = build_vectorstore(load_and_split())
    return vs.as_retriever(search_kwargs={"k": TOP_K})


# ---------------------------------------------------------------------------
# Glue -- a python function to format retrieved docs into one context blob
# ---------------------------------------------------------------------------
def _format(docs) -> str:
    """One normalized block of text for the final prompt."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# STEP 6 -- The full RAG chain, one readable pipe
# ---------------------------------------------------------------------------
def build_rag_chain():
    """Read this like a recipe, left to right:

        question
          |
          v
     { "context": retriever(question),   <-- fetch top-k docs
       "question": question          }   <-- pass question through untouched
          |
          v
     prompt (fills {context} and {question})
          |
          v
     model  -->  parser  -->  final answer string
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions ONLY from the given context. If the "
                "context doesn't contain the answer, say 'Not in the docs.' "
                "Back up every claim with a fact seen in context. Context:\n{context}",
            ),
            ("human", "Question: {question}"),
        ]
    )

    chain = (
        {
            "context": get_retriever() | RunnableLambda(_format),  # find+format
            "question": RunnablePassthrough(),                    # same input passes
        }
        | prompt
        | model
        | StrOutputParser()
    )
    return chain


def main() -> None:
    # Warm up: build/load the vector store, then query
    chunks = load_and_split()
    print(f"=> loaded {len(chunks)} chunks from {DOC_PATH}")

    retriever = get_retriever()
    # Peek at what retrieval actually finds (great for debugging RAG)
    print("\n=> sample retrieval for: 'What is the best vacation policy?'")
    for doc in retriever.invoke("What is the best vacation policy?"):
        print("   •", doc.page_content.replace("\n", " ")[:110])

    chain = build_rag_chain()
    print("\n=> questions against the handbook:")
    for q in [
        "How many vacation days do employees get?",
        "What does Acme Corp do?",
        "Who founded the company and where?",
        "Why is Portland HQ bigger than Austin?",   # not answered in docs!
    ]:
        print(f"   Q: {q}")
        print(f"   A: {chain.invoke({'question': q})}\n")


if __name__ == "__main__":
    main()