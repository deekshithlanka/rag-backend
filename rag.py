"""
Step 1 of the RAG backend: ask questions about your own PDFs.

Commands:
  python rag.py embed "some text"        # see what an embedding looks like
  python rag.py ingest docs/report.pdf   # PDF -> chunks -> embeddings -> Qdrant
  python rag.py search "your question"   # raw vector search: which chunks match?
  python rag.py ask "your question"      # full RAG: search + LLM answer with pages

The big idea:
  INGEST:  PDF -> split into chunks -> each chunk becomes a vector -> stored in Qdrant
  ASK:     question -> vector -> find the closest chunk vectors -> give those chunks
           plus the question to Gemini -> answer that cites page numbers
"""

import os
import sys
import time
import uuid

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()  # reads GOOGLE_API_KEY (and optional settings) from .env

# ---------- Settings (change these to experiment) ----------
COLLECTION = "docs"                        # like a table name in a normal database
EMBED_MODEL = "gemini-embedding-2-preview" # Gemini embedding model
EMBED_DIM = 1536                           # how many numbers per vector (768, 1536 or 3072)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash")  # LLM that writes the final answer
CHUNK_SIZE = 1000                          # characters per chunk
CHUNK_OVERLAP = 150                        # characters shared between neighbouring chunks
TOP_K = 5                                  # how many chunks to retrieve per question

embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, output_dimensionality=EMBED_DIM)


_client = None


def get_client() -> QdrantClient:
    """Qdrant runs inside this script by default (data saved in ./qdrant_data).
    No Docker needed. Set QDRANT_URL in .env to use a Qdrant server or Qdrant Cloud instead.
    One shared client per process: local Qdrant only allows one program to open the folder."""
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL")
        if url:
            _client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
        else:
            _client = QdrantClient(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_data"))
    return _client


def get_store(client: QdrantClient) -> QdrantVectorStore:
    """Create the collection the first time, then wrap it so LangChain can use it."""
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            # COSINE = compare vectors by angle, the standard for text embeddings
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)


# ---------- Command: embed ----------
def embed(text: str):
    """Shows what an embedding actually is: a long list of numbers.
    Same idea as word_vectors in your Week 3 notebook, but learned and 1536 long."""
    vector = embeddings.embed_query(text)
    print(f"'{text}' -> {len(vector)} numbers")
    print("first 8:", [round(x, 4) for x in vector[:8]], "...")


# ---------- Command: ingest ----------
def ingest(pdf_path: str) -> dict:
    # 1. Load: one Document per page, each remembers its page number in metadata
    reader = PdfReader(pdf_path)
    pages = [
        Document(page_content=page.extract_text() or "", metadata={"page": i})
        for i, page in enumerate(reader.pages)
    ]
    print(f"Loaded {len(pages)} pages from {pdf_path}")

    # 2. Split: pages are too long to embed well, so cut them into overlapping chunks.
    #    Splitting each page separately keeps the page number on every chunk.
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    # 3. IDs: the same PDF chunk always gets the same ID, so ingesting a file twice
    #    overwrites its chunks instead of duplicating them
    source = os.path.basename(pdf_path)
    ids = []
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = source
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}-{i}")))

    # 4. Embed + store: LangChain sends each chunk to Gemini for a vector, then saves
    #    vector + text + metadata in Qdrant
    get_store(get_client()).add_documents(chunks, ids=ids)
    print(f"Stored {len(chunks)} chunks in Qdrant collection '{COLLECTION}'")
    return {"filename": source, "pages": len(pages), "chunks": len(chunks)}


# ---------- Command: search ----------
def search(question: str):
    """Vector search only, no LLM. Useful for seeing WHY an answer is good or bad."""
    results = get_store(get_client()).similarity_search_with_score(question, k=TOP_K)
    for rank, (doc, score) in enumerate(results, 1):
        page = doc.metadata.get("page", 0) + 1  # PyPDF counts pages from 0
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"\n#{rank}  score={score:.3f}  {doc.metadata.get('source')} p.{page}")
        print(f"    {preview}...")


def answer_text(message) -> str:
    """Newer Gemini models return a list of content blocks (text + a hidden 'signature').
    Keep only the text parts so the answer prints cleanly."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


def call_with_retry(fn, retries=3):
    """The Gemini free tier allows only ~5 requests per minute.
    If we hit that limit (error 429 / RESOURCE_EXHAUSTED), wait and try again."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if attempt < retries and ("429" in msg or "RESOURCE_EXHAUSTED" in msg):
                print(f"(Rate limit reached - waiting 35s, then retrying {attempt + 1}/{retries})")
                time.sleep(35)
            else:
                raise


# ---------- Command: ask ----------
PROMPT = """Answer the question using ONLY the context below.
Cite the page for each fact like (p. 12).
If the context does not contain the answer, say "I couldn't find that in the documents."

Context:
{context}

Question: {question}"""


def answer_question(question: str) -> dict:
    """The core of RAG. Used by both the CLI (ask) and the web server (/chat)."""
    results = get_store(get_client()).similarity_search(question, k=TOP_K)
    if not results:
        return {"text": "No documents have been added yet. Upload a PDF first.", "sources": []}

    # Build the context block: each chunk labelled with its page
    context = "\n\n".join(
        f"[p. {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}" for doc in results
    )

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)  # 0 = stick to the facts
    answer = call_with_retry(lambda: llm.invoke(PROMPT.format(context=context, question=question)))

    # Sources in rank order (most relevant first), one entry per page
    sources, seen = [], set()
    for doc in results:
        key = (doc.metadata.get("source"), doc.metadata.get("page", 0) + 1)
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})
    return {"text": answer_text(answer), "sources": sources}


def ask(question: str):
    result = answer_question(question)
    print(result["text"])
    if result["sources"]:
        pages = ", ".join(f"p. {s['page']}" for s in result["sources"])
        print(f"\nSources (most relevant first): {pages}")


COMMANDS = {"embed": embed, "ingest": ingest, "search": search, "ask": ask}

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]](" ".join(sys.argv[2:]))
