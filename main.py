import os
from mem0 import Memory
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

# Use the Doppler-provided key for Gemini
api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

config = {
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-1.5-flash-latest",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
        }
    }
}

app = FastAPI()
memory = Memory.from_config(config)

class AddMemoryRequest(BaseModel):
    content: str
    user_id: str
    metadata: dict = {}

@app.post("/add")
async def add_memory(req: AddMemoryRequest):
    result = memory.add(req.content, user_id=req.user_id, metadata=req.metadata)
    return {"result": result}

class SearchRequest(BaseModel):
    query: str
    user_id: str

@app.post("/search")
async def search_memory(req: SearchRequest):
    result = memory.search(req.query, user_id=req.user_id)
    return {"result": result}
