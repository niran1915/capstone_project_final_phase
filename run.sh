#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x backend/.venv/bin/python || ! -d frontend/node_modules ]]; then
  echo "Please run ./setup.sh first." >&2; exit 1
fi
( cd backend && ../backend/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 ) &
BACKEND_PID=$!
cleanup() { kill "$BACKEND_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
sleep 2
(cd frontend && npm run dev -- --host 127.0.0.1)
