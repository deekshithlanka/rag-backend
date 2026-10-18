"""
Step 2: the RAG backend as a web server, so other apps (your Gemini-AI-Chat) can use it.

Start:  uvicorn server:app --port 8001
Docs:   http://localhost:8001/docs   (try every endpoint in the browser)

Endpoints:
  GET  /health   -> {"status": "ok"}                     is the server running?
  POST /chat     -> {"messages": [{"role": "user", "content": "..."}], "rerank": true}
                    returns {"text": "...", "reranked": true,
                             "sources": [{"source": "x.pdf", "page": 3, "score": 0.91, "chunks": 1}]}
  POST /ingest   -> upload a PDF (form field "file")
                    returns {"filename": "...", "pages": 5, "chunks": 5}

Knowledge graph (LightRAG):
  POST /kg/chat    -> same body as /chat, answers from the knowledge graph (hybrid mode)
  POST /kg/ingest  -> upload a PDF; builds the graph IN THE BACKGROUND, returns at once
  GET  /kg/status  -> progress of the current build {"state", "done", "total", "message"}

Same shapes as the reference repos (vector_embeddings, langchain_qdrant_vectordb), so the
chat app side looks the same.
"""

import os
import resource
import secrets
import shutil
import sys
import threading
from typing import List, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag  # all the RAG logic lives in rag.py, shared with the command line
import kg  # knowledge graph (LightRAG), in kg.py

MAX_UPLOAD_MB = 25
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

# Password for this server. When it's online, anyone who finds the URL could otherwise
# use it (and your Gemini quota). The chat app sends it in the X-API-Key header.
# Not set (e.g. on your Mac) = no password needed.
RAG_API_KEY = os.getenv("RAG_API_KEY")


def require_api_key(request: Request, x_api_key: str = Header(default="")):
    if request.url.path == "/health":  # Render's health check calls this without a key
        return
    if RAG_API_KEY and not secrets.compare_digest(x_api_key, RAG_API_KEY):
        raise HTTPException(401, "Missing or wrong API key")


app = FastAPI(
    title="rag-backend",
    description="Ask questions about your PDFs",
    dependencies=[Depends(require_api_key)],
)

# CORS: which websites may call this server from a browser. Your chat app calls it from
# its own server (Next.js API route), so this mostly matters for testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Local Qdrant is a single folder: let one request use it at a time.
db_lock = threading.Lock()


# ---------- Request shapes (FastAPI checks these automatically and returns 422 if wrong) ----------
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10_000)


class ChatRequest(BaseModel):
    messages: List[Message] = Field(min_length=1, max_length=100)
    rerank: bool = True  # rescore the 20 hybrid-search candidates and keep the best 5


# ---------- Endpoints ----------
@app.get("/health")
def health():
    # Peak memory so far (Render's free plan has 512 MB). macOS reports bytes, Linux KB.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    return {"status": "ok", "peak_memory_mb": round(peak_mb)}


@app.post("/chat")
def chat(request: ChatRequest):
    # Only the latest user message is used as the search query.
    question = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if question is None:
        raise HTTPException(400, "No user message found")
    try:
        with db_lock:
            return rag.answer_question(question, rerank=request.rerank)
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise HTTPException(429, "Gemini rate limit reached. Wait a minute and try again.")
        if "API_KEY_INVALID" in msg:
            raise HTTPException(500, "Gemini API key is not valid. Check .env")
        print("chat error:", msg)
        raise HTTPException(500, "Failed to answer the question")


def save_pdf(file: UploadFile) -> str:
    """Save an uploaded PDF into docs/ and return its path."""
    # Only accept PDFs, and keep just the file name (no folders) for safety
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    os.makedirs(DOCS_DIR, exist_ok=True)
    path = os.path.join(DOCS_DIR, filename)
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    if os.path.getsize(path) > MAX_UPLOAD_MB * 1024 * 1024:
        os.remove(path)
        raise HTTPException(413, f"PDF is larger than {MAX_UPLOAD_MB} MB")
    return path


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    path = save_pdf(file)

    try:
        with db_lock:
            return rag.ingest(path)
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise HTTPException(429, "Gemini rate limit reached while embedding. Try a smaller PDF or wait.")
        print("ingest error:", msg)
        raise HTTPException(500, "Failed to process the PDF")


# ---------- Knowledge graph (LightRAG) ----------
@app.post("/kg/chat")
async def kg_chat(request: ChatRequest):
    question = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if question is None:
        raise HTTPException(400, "No user message found")
    # Earlier turns, so follow-ups like "and the year before?" work
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]
    try:
        return await kg.answer_question(question, history)
    except ImportError:
        raise HTTPException(500, "LightRAG is not installed. Run upgrade_kg.command")
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise HTTPException(429, "Gemini rate limit reached. Wait a minute and try again.")
        print("kg chat error:", msg)
        raise HTTPException(500, "Failed to answer from the knowledge graph")


@app.post("/kg/ingest")
async def kg_ingest(file: UploadFile = File(...)):
    path = save_pdf(file)
    try:
        return kg.start_ingest(path)
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/kg/status")
def kg_status():
    return kg.status
