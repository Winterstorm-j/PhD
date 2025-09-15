from fastapi import FastAPI, Request
from llama_cpp import Llama

app = FastAPI()
llm = Llama(model_path="LLM/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=2048)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    result = llm(prompt, max_tokens=200)
    return {"response": result["choices"][0]["text"].strip()}
