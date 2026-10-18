#!/bin/bash
# Double-click to start the RAG server. Keep this window open while you use it.
# Stop it with Ctrl+C (or close the window).
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Installing any new packages (first time only)..."
pip install -q -r requirements.txt
echo
echo "Starting server at http://localhost:8001  (API docs: http://localhost:8001/docs)"
uvicorn server:app --port 8001 2>&1 | tee server_log.txt
