#!/bin/bash

# Exit on error
set -e

# ======== CONFIGURATION ========
MODEL_NAME="deepseek-ai/DeepSeek-R1"  # Change to your preferred model
MODEL_FILE="deepseek-ai/DeepSeek-R1"     # Adjust for your hardware (check Hugging Face repo)

# MODEL_NAME="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"  # Change to your preferred model
# MODEL_FILE="mistral-7b-instruct-v0.2.Q4_K_M.gguf"     # Adjust for your hardware (check Hugging Face repo)

PORT=8000
# =================================

echo "[1/6] Creating virtual environment..."
python3 -m venv .llm_env
source .llm_env/bin/activate

echo "[2/6] Installing dependencies..."
pip install --upgrade pip
pip install llama-cpp-python huggingface-hub fastapi uvicorn

echo "[3/6] Downloading model ($MODEL_NAME)..."
mkdir -p PhD-Windows/LLM/models && cd PhD-Windows/LLM/models
huggingface-cli download $MODEL_NAME --local-dir .
cd ..

echo "[4/6] Creating FastAPI app..."
cat <<EOF > app.py
from fastapi import FastAPI, Request
from llama_cpp import Llama

app = FastAPI()
llm = Llama(model_path="LLM/models/$MODEL_FILE", n_ctx=2048)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    result = llm(prompt, max_tokens=200)
    return {"response": result["choices"][0]["text"].strip()}
EOF

echo "[5/6] Running server at http://localhost:$PORT..."
uvicorn app:app --host 0.0.0.0 --port $PORT
