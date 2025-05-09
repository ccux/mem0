import os
from mem0 import Memory
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from typing import Optional, List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

# Use the Doppler-provided key for Gemini
api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_GENERATIVE_AI_API_KEY or GOOGLE_API_KEY not found in environment variables.")
    # Decide if you want to exit or try to continue without it,
    # though mem0 will likely fail later if LLM/Embedder is needed.
    # exit(1) # Uncomment to exit if key is critical for startup

genai.configure(api_key=api_key)

# Default configuration for mem0
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
            "model": "models/text-embedding-004", # Produces 768 dimensions
        }
    },
    "vector_store": {                 # ADDED THIS SECTION
        "provider": "pgvector",       # Specify pgvector
        "config": {
            "embedding_model_dims": 768 # Explicitly set to 768
            # dbname, user, password, host, port are expected to be picked up
            # from environment variables by mem0's PGVector store if not specified here.
            # collection_name will default to 'memories' if not specified, which is fine.
        }
    }
}

app = FastAPI()

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",  # Frontend Next.js app
    # Add any other origins you need to allow (e.g., deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of allowed origins
    allow_credentials=True, # Allow cookies to be sent
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)
# --- End CORS Configuration ---

memory = Memory.from_config(config)

class AddMemoryRequest(BaseModel):
    messages: List[Dict[str, str]]
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@app.post("/memories")
async def add_memory_route(req: AddMemoryRequest):
    result = memory.add(
        messages=req.messages,
        user_id=req.user_id,
        agent_id=req.agent_id,
        run_id=req.run_id,
        metadata=req.metadata
    )
    return {"result": result}

class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 10

@app.post("/search")
async def search_memory_route(req: SearchRequest):
    result = memory.search(
        query=req.query,
        user_id=req.user_id,
        agent_id=req.agent_id,
        run_id=req.run_id,
        filters=req.filters,
        limit=req.limit or 10
    )
    return result

@app.get("/memories")
async def get_all_memories_route(user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None, limit: Optional[int] = 100):
    memories = memory.get_all(user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit)
    return memories

@app.delete("/memories")
async def delete_all_user_memories_route(user_id: str):
    result = memory.delete_all(user_id=user_id)
    return {"result": result, "message": f"All memories for user {user_id} deleted."}
