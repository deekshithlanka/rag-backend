#!/bin/bash
# Double-click AFTER start_server.command is running. Results saved to server_test_output.txt
cd "$(dirname "$0")"
{
echo "===== 1. HEALTH ====="
curl -s http://localhost:8001/health; echo; echo
echo "===== 2. CHAT: How does chunking work? ====="
curl -s -X POST http://localhost:8001/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"How does chunking work?"}]}'; echo; echo
echo "===== 3. CHAT (not in the PDF): Who won the 2022 World Cup? ====="
curl -s -X POST http://localhost:8001/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Who won the 2022 World Cup?"}]}'; echo; echo
echo "===== 4. BAD REQUEST (should be rejected with an error) ====="
curl -s -X POST http://localhost:8001/chat -H "Content-Type: application/json" \
  -d '{"messages":[]}'; echo; echo
echo "===== 5. INGEST (upload the test PDF again) ====="
curl -s -X POST http://localhost:8001/ingest -F "file=@docs/rag_notes.pdf"; echo; echo
echo "===== DONE ====="
} 2>&1 | tee server_test_output.txt
