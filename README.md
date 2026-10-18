# rag-backend

Ask questions about your own PDFs using a vector database (Qdrant) and Gemini.

## Setup (Mac)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your Gemini key in .env
```

## Use

```bash
python rag.py embed "hello world"              # what an embedding looks like
python rag.py ingest docs/your.pdf             # load a PDF into Qdrant
python rag.py search "your question"           # which chunks match, with scores
python rag.py ask "your question"              # answer with page citations
```

Qdrant runs inside the script and saves data in `qdrant_data/`. No Docker needed.
Only one command can use that folder at a time.

## How it works

```
INGEST: PDF -> pages -> chunks (1000 chars) -> embeddings -> Qdrant
ASK:    question -> embedding -> top 5 closest chunks -> LLM -> answer + pages
```

## Next step

Wrap `ingest` and `ask` in a FastAPI server so the chat app can call them.

## Run as a web server (for the chat app)

```bash
uvicorn server:app --port 8001      # or double-click start_server.command
```

- `GET  /health` → `{"status": "ok"}`
- `POST /chat` with `{"messages": [{"role": "user", "content": "..."}]}` → `{"text", "sources"}`
- `POST /ingest` with a PDF upload (field `file`) → `{"filename", "pages", "chunks"}`
- Interactive docs: http://localhost:8001/docs

Local Qdrant allows one program at a time: stop the server before using `python rag.py ...`.
