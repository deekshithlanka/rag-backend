"""
Step 2: the RAG backend as a web server, so other apps (your Gemini-AI-Chat) can use it.

Start:  uvicorn server:app --port 8001
Docs:   http://localhost:8001/docs   (try every endpoint in the browser)

Endpoints:
  GET  /health   -> {"status": "ok"}                     is the server running?
  POST /chat     -> {"messages": [{"role": "user", "content": "..."}]}
                    returns {"text": "...", "sources": [{"source": "x.pdf", "page": 3}]}
  POST /ingest   -> upload a PDF (form field "file")
                    returns {"filename": "...", "pages": 5, "chunks": 5}

Same shapes as the reference repos (vector_embeddings, langchain_qdrant_vectordb), so the
chat app side looks the same.
"""

import os
import shutil
import threading
from typing import List, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag  # all the RAG logic lives in rag.py, shared with the command line

MAX_UPLOAD_MB = 25
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

app = FastAPI(title="rag-backend", description="Ask questions about your PDFs")

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


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    # Only the latest user message is used as the search query.
    question = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if question is None:
        raise HTTPException(400, "No user message found")
    try:
        with db_lock:
            return rag.answer_question(question)
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise HTTPException(429, "Gemini rate limit reached. Wait a minute and try again.")
        if "API_KEY_INVALID" in msg:
            raise HTTPException(500, "Gemini API key is not valid. Check .env")
        print("chat error:", msg)
        raise HTTPException(500, "Failed to answer the question")


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
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

    try:
        with db_lock:
            return rag.ingest(path)
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise HTTPException(429, "Gemini rate limit reached while embedding. Try a smaller PDF or wait.")
        print("ingest error:", msg)
        raise HTTPException(500, "Failed to process the PDF")
