#!/bin/bash
# Double-click ONCE to switch to hybrid search + reranking.
# Stop the RAG server first (close its window). Output saved to upgrade_output.txt
cd "$(dirname "$0")"
source .venv/bin/activate
{
echo "===== 1. Installing fastembed (keyword search + reranker) ====="
pip install -q -r requirements.txt && echo "installed"
echo
echo "===== 2. Re-adding your PDFs with keyword vectors too ====="
python rag.py reindex
echo
echo "===== 3. Compare: before vs after reranking ====="
python rag.py search "How much overlap is there between chunks?"
echo
echo "===== 4. Ask (with reranking) ====="
python rag.py ask "How much overlap is there between chunks?"
echo
echo "===== DONE ====="
} 2>&1 | tee upgrade_output.txt
read -p "Press Enter to close this window..." _
