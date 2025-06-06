#!/bin/bash

# Exit on error
set -e

# ======== CONFIGURATION ========
MODEL_NAME="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"  # Change to your preferred model
MODEL_FILE="mistral-7b-instruct-v0.2.Q4_K_M.gguf"     # Adjust for your hardware (check Hugging Face repo)
PORT=8000
# =================================

# echo "[1/6] Creating virtual environment..."
python3 -m venv .llm_env
source .llm_env/bin/activate

echo "[5/6] Running server at http://localhost:$PORT..."
uvicorn app:app --host 0.0.0.0 --port $PORT
