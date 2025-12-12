#!/bin/bash

# Start FastAPI in background
uvicorn app:app --reload --host 127.0.0.1 --port 8000 &
FASTAPI_PID=$!

# Wait for FastAPI to start
sleep 3

# Start Streamlit
streamlit run streamlit_app.py

# Cleanup on exit
trap "kill $FASTAPI_PID" EXIT