import os
from datetime import datetime, timezone
from mem0 import Memory
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from typing import Optional, List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the Doppler-provided key for Gemini
api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_GENERATIVE_AI_API_KEY or GOOGLE_API_KEY not found in environment variables.")
    # Decide if you want to exit or try to continue without it,
    # though mem0 will likely fail later if LLM/Embedder is needed.
    # exit(1) # Uncomment to exit if key is critical for startup

genai.configure(api_key=api_key)

# Configuration for mem0
config = {
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.5-flash",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/gemini-embedding-001",
            "embedding_dims": 1536,  # Use 1536 dimensions as supported by Gemini
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
            "collection_name": os.getenv("QDRANT_COLLECTION_NAME", "memories"),
            "embedding_model_dims": 1536,  # Match Gemini output dimensionality
            "on_disk": True
        }
    }
}

# Debug: Log the configuration being used
logger.info(f"Mem0 Configuration: {config}")

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

# Initialize memory with error handling
try:
    logger.info("Initializing Mem0 memory with Qdrant configuration...")
    memory = Memory.from_config(config)
    logger.info("Mem0 memory initialized successfully with Qdrant")
    memory_available = True
except Exception as e:
    logger.error(f"Failed to initialize Mem0 memory: {e}")
    memory_available = False
    memory = None

class AddMemoryRequest(BaseModel):
    messages: List[Dict[str, str]]
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@app.post("/memories")
async def add_memory_route(req: AddMemoryRequest):
    if not memory_available or memory is None:
        return {"error": "Memory service is not available", "detail": "Mem0 memory was not initialized properly"}

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
    if not memory_available or memory is None:
        return {"error": "Memory service is not available", "detail": "Mem0 memory was not initialized properly"}

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
    if not memory_available or memory is None:
        return {"error": "Memory service is not available", "detail": "Mem0 memory was not initialized properly"}

    memories = memory.get_all(user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit)
    return memories

@app.delete("/memories/{memory_id}")
async def delete_memory_route(memory_id: str):
    """Delete a specific memory by ID"""
    if not memory_available or memory is None:
        return {"error": "Memory service is not available", "detail": "Mem0 memory was not initialized properly"}

    try:
        print(f"[DEBUG] Attempting to delete memory: {memory_id}")

        # First check if the memory exists
        try:
            existing_memory = memory.get(memory_id=memory_id)
            print(f"[DEBUG] Memory exists check: {existing_memory is not None}")
            if not existing_memory:
                print(f"[DEBUG] Memory {memory_id} not found in get() call")
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
        except Exception as get_error:
            print(f"[DEBUG] Error checking memory existence: {get_error}")
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found: {str(get_error)}")

        # If memory exists, delete it
        result = memory.delete(memory_id=memory_id)
        print(f"[DEBUG] Delete result: {result}")
        return {"result": result, "message": f"Memory {memory_id} deleted successfully."}

    except Exception as e:
        print(f"[DEBUG] Exception in delete_memory_route: {e}")
        print(f"[DEBUG] Exception type: {type(e)}")
        from fastapi import HTTPException
        if "HTTPException" in str(type(e)):
            raise e  # Re-raise HTTPExceptions
        raise HTTPException(status_code=500, detail=f"Error deleting memory {memory_id}: {str(e)}")

@app.delete("/memories")
async def delete_all_user_memories_route(user_id: str):
    if not memory_available or memory is None:
        return {"error": "Memory service is not available", "detail": "Mem0 memory was not initialized properly"}

    result = memory.delete_all(user_id=user_id)
    return {"result": result, "message": f"All memories for user {user_id} deleted."}

@app.get("/health")
async def health_check():
    """Health check endpoint for production monitoring"""
    try:
        # Basic health check - ensure the service is running
        return {
            "status": "ok" if memory_available else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "mem0-service",
            "version": "1.0.0",
            "memory_provider": "qdrant" if memory_available else "none",
            "llm_provider": "gemini",
            "embedding_dimensions": 1536,
            "memory_available": memory_available,
            "memory_initialized": memory_available and memory is not None
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "mem0-service",
            "error": str(e),
            "memory_available": False,
            "memory_initialized": False
        }
