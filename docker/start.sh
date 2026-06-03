#!/bin/sh
set -e

cd /app

uvicorn main:app --host 0.0.0.0 --port 8080 &
UVICORN_PID=$!

streamlit run dashboard/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true &
STREAMLIT_PID=$!

while kill -0 "${STREAMLIT_PID}" 2>/dev/null; do
  if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
    echo "uvicorn exited; stopping streamlit"
    kill "${STREAMLIT_PID}" 2>/dev/null || true
    wait "${STREAMLIT_PID}" 2>/dev/null || true
    wait "${UVICORN_PID}" 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

wait "${STREAMLIT_PID}"
STREAMLIT_STATUS=$?
kill "${UVICORN_PID}" 2>/dev/null || true
wait "${UVICORN_PID}" 2>/dev/null || true
exit "${STREAMLIT_STATUS}"
