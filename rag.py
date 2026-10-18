"""
RAG backend: ask questions about your own PDFs.

Commands:
  python rag.py embed "some text"        # see what an embedding looks like
  python rag.py ingest docs/report.pdf   # PDF -> chunks -> embeddings -> Qdrant
  python rag.py reindex                  # re-ingest every PDF in docs/
  python rag.py search "your question"   # compare: hybrid search order vs reranked order
  python rag.py ask "your question"      # full RAG: search + rerank + Gemini answer

The big idea:
  INGEST:  PDF -> chunks -> two vectors per chunk -> stored in Qdrant
             * dense vector  (Gemini embedding): captures MEANING
             * sparse vector (BM25 keywords):    captures exact WORDS, names, numbers
  ASK:     question -> HYBRID search (meaning + keywords) finds 20 candidate chunks
           -> RERANKER reads each candidate next to the question and scores how well it
              actually ANSWERS it -> keep the best 5 -> Gemini answers with page numbers
"""

import atexit
import glob
import math
import os
import sys
import time
import uuid

# Avoids a Hugging Face download bug ("Byte range not sequential") for the reranker model
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, SparseVectorParams, VectorParams

load_dotenv()  # reads GOOGLE_API_KEY (and optional settings) from .env

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- Settings (change these to experiment) ----------
COLLECTION = "docs_hybrid"                 # like a table name in a normal database
EMBED_MODEL = "gemini-embedding-2-preview" # Gemini embedding model (meaning)
EMBED_DIM = 1536                           # numbers per dense vector (768, 1536 or 3072)
SPARSE_MODEL = "Qdrant/bm25"               # keyword model, runs locally, no key
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-12-v2")  # local, no key; L-6 is smaller
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.6-flash")  # LLM that writes the answer
CHUNK_SIZE = 1000                          # characters per chunk
CHUNK_OVERLAP = 150                        # characters shared between neighbouring chunks
TOP_K = 5                                  # chunks given to Gemini
CANDIDATES = 20                            # chunks fetched before reranking

embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, output_dimensionality=EMBED_DIM)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)

_client = None
_reranker = None


def get_client() -> QdrantClient:
    """Qdrant runs inside this program by default (data saved in ./qdrant_data).
    Set QDRANT_URL in .env to use Qdrant Cloud instead (needed when deployed).
    One shared client per process: local Qdrant only allows one program to open the folder."""
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL")
        if url:
            _client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
        else:
            _client = QdrantClient(path=os.path.join(HERE, "qdrant_data"))
        atexit.register(_client.close)  # close cleanly on exit (avoids a noisy warning)
    return _client


def get_store() -> QdrantVectorStore:
    """Create the collection the first time (with room for BOTH vector types), then wrap it."""
    client = get_client()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            # dense: compare by angle (COSINE), the standard for text embeddings
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            # sparse: keyword weights, stored under LangChain's default name
            sparse_vectors_config={QdrantVectorStore.SPARSE_VECTOR_NAME: SparseVectorParams()},
        )
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,  # meaning + keywords, merged with RRF
    )


def get_reranker():
    """Cross-encoder reranker. Downloaded once (~130 MB) on first use, then cached."""
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


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


def answer_text(message) -> str:
    """Newer Gemini models return a list of content blocks (text + a hidden 'signature').
    Keep only the text parts."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


# ---------- Command: embed ----------
def embed(text: str):
    """Shows what an embedding actually is: a long list of numbers."""
    vector = embeddings.embed_query(text)
    print(f"'{text}' -> {len(vector)} numbers")
    print("first 8:", [round(x, 4) for x in vector[:8]], "...")


# ---------- Command: ingest ----------
def ingest(pdf_path: str) -> dict:
    # 1. Load: one Document per page, each remembers its page number
    reader = PdfReader(pdf_path)
    pages = [
        Document(page_content=page.extract_text() or "", metadata={"page": i})
        for i, page in enumerate(reader.pages)
    ]
    print(f"Loaded {len(pages)} pages from {pdf_path}")

    # 2. Split each page into overlapping chunks (keeps the page number on every chunk)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
    print(f"Split into {len(chunks)} chunks")

    # 3. Stable IDs: re-ingesting the same file overwrites instead of duplicating
    source = os.path.basename(pdf_path)
    ids = []
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = source
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}-{i}")))

    # 4. Embed (dense via Gemini + sparse via BM25) and store, in batches so a long PDF
    #    doesn't hit the Gemini rate limit all at once
    store = get_store()
    batch = 50
    for start in range(0, len(chunks), batch):
        call_with_retry(
            lambda: store.add_documents(chunks[start:start + batch], ids=ids[start:start + batch])
        )
    print(f"Stored {len(chunks)} chunks in Qdrant collection '{COLLECTION}'")
    return {"filename": source, "pages": len(pages), "chunks": len(chunks)}


# ---------- Command: reindex ----------
def reindex(_: str = ""):
    """Re-ingest every PDF in docs/ (needed once after switching to hybrid search)."""
    pdfs = sorted(glob.glob(os.path.join(HERE, "docs", "*.pdf")))
    if not pdfs:
        print("No PDFs in docs/")
    for pdf in pdfs:
        ingest(pdf)


# ---------- Retrieval: hybrid search + rerank ----------
def retrieve(question: str, rerank: bool = True) -> list:
    """Returns [(Document, score), ...], best first.
    With rerank: score = reranker relevance from 0 to 1 (1 = clearly answers the question).
    Without:     score = hybrid search rank score (RRF)."""
    store = get_store()
    k = CANDIDATES if rerank else TOP_K
    results = store.similarity_search_with_score(question, k=k)
    if not rerank or not results:
        return results[:TOP_K]

    docs = [doc for doc, _ in results]
    scores = list(get_reranker().rerank(question, [d.page_content for d in docs]))
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    # The reranker gives raw scores like 5.4 or -11.2; squash them to 0..1 to read easily
    return [(doc, 1 / (1 + math.exp(-float(score)))) for doc, score in ranked[:TOP_K]]


# ---------- Command: search ----------
def search(question: str):
    """Shows how reranking changes the order. Useful for seeing WHY an answer is good or bad."""
    for label, rerank in [("HYBRID SEARCH ORDER (no rerank)", False), ("AFTER RERANKING", True)]:
        print(f"\n=== {label} ===")
        for rank, (doc, score) in enumerate(retrieve(question, rerank=rerank), 1):
            page = doc.metadata.get("page", 0) + 1
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"#{rank}  score={score:.3f}  {doc.metadata.get('source')} p.{page}  {preview}...")


# ---------- Command: ask ----------
PROMPT = """Answer the question using ONLY the context below.
Cite the page for each fact like (p. 12).
If the context does not contain the answer, say "I couldn't find that in the documents."

Context:
{context}

Question: {question}"""


def answer_question(question: str, rerank: bool = True) -> dict:
    """The core of RAG. Used by both the CLI (ask) and the web server (/chat)."""
    results = retrieve(question, rerank=rerank)
    if not results:
        return {"text": "No documents have been added yet. Upload a PDF first.", "sources": [],
                "reranked": rerank}

    context = "\n\n".join(
        f"[p. {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}" for doc, _ in results
    )
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)  # 0 = stick to the facts
    answer = call_with_retry(lambda: llm.invoke(PROMPT.format(context=context, question=question)))

    # Sources in rank order (most relevant first), one entry per page, with best score
    # and how many chunks came from that page
    sources = {}
    for doc, score in results:
        key = (doc.metadata.get("source"), doc.metadata.get("page", 0) + 1)
        if key in sources:
            sources[key]["chunks"] += 1
        else:
            sources[key] = {"source": key[0], "page": key[1], "score": round(score, 3), "chunks": 1}
    return {"text": answer_text(answer), "sources": list(sources.values()), "reranked": rerank}


def ask(question: str):
    result = answer_question(question)
    print(result["text"])
    if result["sources"]:
        pages = ", ".join(f"p. {s['page']} ({s['score']})" for s in result["sources"])
        print(f"\nSources, reranked: {pages}")


COMMANDS = {"embed": embed, "ingest": ingest, "reindex": reindex, "search": search, "ask": ask}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS or (sys.argv[1] != "reindex" and len(sys.argv) < 3):
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]](" ".join(sys.argv[2:]))
