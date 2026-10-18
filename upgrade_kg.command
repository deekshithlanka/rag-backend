#!/bin/bash
# Double-click ONCE to install the knowledge graph (LightRAG) and build a test graph.
# Stop the RAG server first (close its window). Output saved to kg_output.txt
cd "$(dirname "$0")"
source .venv/bin/activate
{
echo "===== 1. Checking LightRAG ====="
python -c "import lightrag; print('LightRAG', getattr(lightrag, '__version__', 'installed'))"
echo
echo "===== 2. Building a knowledge graph from rag_notes.pdf (a few minutes) ====="
rm -rf lightrag_data  # clear the half-saved graph from the first attempt
python - <<'PY'
import asyncio, kg
async def main():
    await kg.ingest_pdf("docs/rag_notes.pdf")
    print(kg.status["message"])
    r = await kg.answer_question("How do chunking, embeddings and the vector database work together?", [])
    print("\nANSWER:\n" + r["text"])
    print("\nSOURCES:", r["sources"])
    rag = await kg.get_rag()
    await rag.finalize_storages()
asyncio.run(main())
PY
echo
echo "===== DONE ====="
} 2>&1 | tee kg_output.txt
read -p "Press Enter to close this window..." _
