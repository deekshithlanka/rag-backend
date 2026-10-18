"""
Knowledge graph RAG with LightRAG (Gemini for everything).

How it differs from vector search (rag.py):
  INGEST: Gemini READS every chunk and pulls out ENTITIES (companies, people, products,
          numbers...) and the RELATIONSHIPS between them, building a graph:
              [Apple] --designs--> [iPhone] --assembled by--> [Foxconn]
          Slow: about one Gemini call per chunk, so it runs in the background.
  ASK:    LightRAG "hybrid" mode searches the graph at two levels:
            local  = entities named in the question, plus their neighbours
            global = broad themes, via relationships across the whole graph
          Good for questions that CONNECT ideas from different parts of a document.

Each PDF page is inserted as its own document named "<file> p.<n>", so answers can cite pages.
Data: ./lightrag_data locally; MongoDB Atlas + Qdrant Cloud when MONGO_URI / QDRANT_URL are set.
"""

import asyncio
import os
import re
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(HERE, "lightrag_data")
KG_MODEL = os.getenv("KG_MODEL", "gemini-3.5-flash-lite")  # extraction + answers
EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_DIM = 768

_client = None
_rag = None
_rag_lock = asyncio.Lock()

# Background ingest progress, read by GET /kg/status
status = {"state": "idle", "file": None, "done": 0, "total": 0, "skipped": 0, "message": ""}


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


async def with_retry(make_call, retries=5):
    """Gemini free tier = few requests per minute. On 429, wait and retry."""
    for attempt in range(retries + 1):
        try:
            return await make_call()
        except Exception as e:
            msg = str(e)
            if attempt < retries and ("429" in msg or "RESOURCE_EXHAUSTED" in msg):
                wait = 20 + attempt * 10
                print(f"[kg] rate limit, waiting {wait}s (retry {attempt + 1}/{retries})")
                await asyncio.sleep(wait)
            else:
                raise


# ---------- The two functions LightRAG needs: an LLM and an embedder ----------
async def gemini_llm(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs) -> str:
    contents = []
    for m in history_messages or []:
        role = "model" if m.get("role") == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=str(m.get("content", "")))]))
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=0)
    response = await with_retry(
        lambda: client().aio.models.generate_content(model=KG_MODEL, contents=contents, config=config)
    )
    return response.text or ""


_embed_slots = asyncio.Semaphore(4)  # at most 4 embedding calls at once


async def _embed_one(text: str) -> list[float]:
    config = types.EmbedContentConfig(output_dimensionality=EMBED_DIM)
    async with _embed_slots:
        result = await with_retry(
            lambda: client().aio.models.embed_content(model=EMBED_MODEL, contents=text, config=config)
        )
    return result.embeddings[0].values


async def gemini_embed(texts: list[str]) -> np.ndarray:
    # One call PER text: gemini-embedding-2 is multimodal, so a list of texts in one call
    # is treated as ONE combined input and returns a single embedding.
    vectors = await asyncio.gather(*(_embed_one(t) for t in texts))
    return np.array(vectors, dtype=np.float32)


async def get_rag():
    """Create LightRAG once (imported lazily so the rest of the server works without it)."""
    global _rag
    async with _rag_lock:
        if _rag is None:
            from lightrag import LightRAG
            from lightrag.utils import wrap_embedding_func_with_attrs

            embed = wrap_embedding_func_with_attrs(
                embedding_dim=EMBED_DIM, max_token_size=2048, model_name=EMBED_MODEL
            )(gemini_embed)
            os.makedirs(WORKING_DIR, exist_ok=True)
            # Online (Render's disk is wiped on every restart), keep the graph in the cloud:
            #   MongoDB Atlas -> graph, documents, cache    Qdrant Cloud -> vectors
            # Locally, without those settings, everything stays in ./lightrag_data
            storage = {}
            if os.getenv("MONGO_URI"):
                os.environ.setdefault("MONGO_DATABASE", "lightrag")
                storage.update(kv_storage="MongoKVStorage", graph_storage="MongoGraphStorage",
                               doc_status_storage="MongoDocStatusStorage")
            if os.getenv("QDRANT_URL"):
                storage["vector_storage"] = "QdrantVectorDBStorage"
            rag = LightRAG(
                working_dir=WORKING_DIR,
                **storage,
                llm_model_func=gemini_llm,
                embedding_func=embed,
                llm_model_max_async=2,  # stay under the free-tier rate limit
                embedding_func_max_async=2,
            )
            await rag.initialize_storages()
            _rag = rag
    return _rag


# ---------- Ingest (runs in the background; can take minutes) ----------
async def ingest_pdf(pdf_path: str):
    source = os.path.basename(pdf_path)
    reader = PdfReader(pdf_path)
    pages = [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]
    pages = [(n, t) for n, t in pages if t]
    status.update(state="running", file=source, done=0, total=len(pages), skipped=0,
                  message=f"Building knowledge graph from {source}")
    started = time.time()
    try:
        rag = await get_rag()
        for n, text in pages:
            doc_id = f"{source}-p{n}"
            # Pages already in the graph are skipped (re-uploading costs nothing)
            try:
                existing = await rag.doc_status.get_by_id(doc_id)
            except Exception:
                existing = None  # LightRAG also skips duplicate ids by itself
            if existing and "processed" in str(existing.get("status", "")).lower():
                status["skipped"] += 1
            else:
                await rag.ainsert(text, ids=[doc_id], file_paths=[f"{source} p.{n}"])
                # LightRAG logs failures instead of raising them, so check the page's status
                try:
                    after = await rag.doc_status.get_by_id(doc_id)
                except Exception:
                    after = None
                if after and "failed" in str(after.get("status", "")).lower():
                    raise RuntimeError(f"page {n}: {after.get('error_msg') or after.get('error') or 'failed'}")
            status["done"] += 1
            print(f"[kg] {source}: {status['done']} of {status['total']} pages")
        secs = int(time.time() - started)
        status.update(state="done", message=(
            f"Knowledge graph ready: {source}, {status['total']} pages"
            + (f" ({status['skipped']} already in the graph)" if status["skipped"] else "")
            + f", {secs}s"))
    except Exception as e:
        print("[kg] ingest error:", e)
        status.update(state="error", message=f"Knowledge graph build failed: {e}")


def start_ingest(pdf_path: str) -> dict:
    if status["state"] == "running":
        raise RuntimeError("A knowledge graph build is already running. Wait for it to finish.")
    asyncio.get_running_loop().create_task(ingest_pdf(pdf_path))
    status.update(state="running", file=os.path.basename(pdf_path), done=0, total=0, skipped=0,
                  message="Starting...")
    return {"status": "started", "filename": os.path.basename(pdf_path)}


# ---------- Ask ----------
PAGE_REF = re.compile(r"([\w\-. ]+?\.pdf) p\.(\d+)", re.IGNORECASE)
NUMBERED_REF = re.compile(r"\[(\d+)\]\s*([\w\-. ]+?\.pdf) p\.(\d+)", re.IGNORECASE)
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
REFS_HEADING = re.compile(r"\n\s*(#+\s*References\b[^\n]*|References:?)\s*\n", re.IGNORECASE)


async def answer_question(question: str, history: list) -> dict:
    from lightrag import QueryParam

    rag = await get_rag()
    param = QueryParam(mode="hybrid", conversation_history=history[-6:], enable_rerank=False)
    text = await rag.aquery(question, param=param)
    text = text if isinstance(text, str) else str(text)

    # LightRAG cites like "[1, 3]" and ends with a "References" list: "- [1] rag_notes.pdf p.3".
    split = REFS_HEADING.search(text)
    body, refs = (text[:split.start()], text[split.end():]) if split else (text, "")
    number_to_page = {num: (name.strip(), int(page)) for num, name, page in NUMBERED_REF.findall(refs)}

    # Sources: pages the answer actually cites, in order of first citation
    sources, seen = [], set()
    for group in CITATION.findall(body):
        for num in re.split(r"\s*,\s*", group):
            if num in number_to_page and number_to_page[num] not in seen:
                seen.add(number_to_page[num])
                sources.append({"source": number_to_page[num][0], "page": number_to_page[num][1]})
    if not sources:  # fall back to any page names in the text
        for name, page in PAGE_REF.findall(text):
            key = (name.strip(), int(page))
            if key not in seen:
                seen.add(key)
                sources.append({"source": key[0], "page": key[1]})

    # Rewrite "[1, 3]" as "(p. 3, p. 2)" so citations look like the vector mode's
    def to_pages(match):
        pages = []
        for num in re.split(r"\s*,\s*", match.group(1)):
            if num in number_to_page and f"p. {number_to_page[num][1]}" not in pages:
                pages.append(f"p. {number_to_page[num][1]}")
        return f"({', '.join(pages)})" if pages else match.group(0)

    body = CITATION.sub(to_pages, body).strip()
    return {"text": body or text, "sources": sources[:5], "retrieval": "graph"}
