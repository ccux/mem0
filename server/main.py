import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from mem0 import Memory, AsyncMemory
from prompts import KNOWLEDGE_EXTRACTION_PROMPT, CATEGORIZATION_PROMPT, UPDATE_MEMORY_PROMPT, format_categorization_prompt

app = FastAPI(title="Mem0 Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory instances
memory_instances: Dict[str, Memory] = {}
async_memory_instances: Dict[str, AsyncMemory] = {}

class MemoryRequest(BaseModel):
    user_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class MemoryResponse(BaseModel):
    success: bool
    memory_id: Optional[str] = None
    message: str
    metadata: Optional[Dict[str, Any]] = None

class KnowledgeExtractionRequest(BaseModel):
    content: str

class KnowledgeExtractionResponse(BaseModel):
    success: bool
    items: List[Dict[str, Any]]
    message: str

class CategorizationRequest(BaseModel):
    content: str

class CategorizationResponse(BaseModel):
    success: bool
    category: str
    confidence: float
    message: str

@app.get("/")
async def root():
    return {"message": "Mem0 Server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mem0-server"}

@app.post("/extract-knowledge", response_model=KnowledgeExtractionResponse)
async def extract_knowledge(request: KnowledgeExtractionRequest):
    """Extract knowledge from content using the centralized prompt"""
    try:
        # Use the centralized knowledge extraction prompt
        prompt = f"{KNOWLEDGE_EXTRACTION_PROMPT}\n\nContent to analyze:\n{request.content}"
        
        # For now, return a mock response since we don't have LLM integration here
        # In a real implementation, you would call the LLM with this prompt
        mock_response = {
            "success": True,
            "items": [
                {
                    "content": "Example extracted knowledge",
                    "category": "general",
                    "confidence": 0.8
                }
            ],
            "message": "Knowledge extracted successfully using centralized prompt"
        }
        
        return KnowledgeExtractionResponse(**mock_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge extraction failed: {str(e)}")

@app.post("/categorize", response_model=CategorizationResponse)
async def categorize_content(request: CategorizationRequest):
    """Categorize content using the centralized prompt"""
    try:
        # Use the centralized categorization prompt
        prompt = format_categorization_prompt(request.content)
        
        # For now, return a mock response since we don't have LLM integration here
        # In a real implementation, you would call the LLM with this prompt
        mock_response = {
            "success": True,
            "category": "general",
            "confidence": 0.8,
            "message": "Content categorized successfully using centralized prompt"
        }
        
        return CategorizationResponse(**mock_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Categorization failed: {str(e)}")

@app.post("/memory/add", response_model=MemoryResponse)
async def add_memory(request: MemoryRequest):
    """Add a new memory"""
    try:
        if request.user_id not in memory_instances:
            memory_instances[request.user_id] = Memory()
        
        memory = memory_instances[request.user_id]
        result = memory.add(request.content, request.metadata or {})
        
        return MemoryResponse(
            success=True,
            memory_id=result.get("id"),
            message="Memory added successfully",
            metadata=result.get("metadata")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add memory: {str(e)}")

@app.post("/memory/async/add", response_model=MemoryResponse)
async def add_memory_async(request: MemoryRequest):
    """Add a new memory asynchronously"""
    try:
        if request.user_id not in async_memory_instances:
            async_memory_instances[request.user_id] = AsyncMemory()
        
        memory = async_memory_instances[request.user_id]
        result = await memory.add(request.content, request.metadata or {})
        
        return MemoryResponse(
            success=True,
            memory_id=result.get("id"),
            message="Memory added successfully (async)",
            metadata=result.get("metadata")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add memory: {str(e)}")

@app.get("/memory/{user_id}")
async def get_memories(user_id: str, limit: int = 100, offset: int = 0):
    """Get memories for a user"""
    try:
        if user_id not in memory_instances:
            return {"memories": [], "total": 0}
        
        memory = memory_instances[user_id]
        memories = memory.get(limit=limit, offset=offset)
        
        return {
            "memories": memories,
            "total": len(memories)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memories: {str(e)}")

@app.delete("/memory/{user_id}/{memory_id}")
async def delete_memory(user_id: str, memory_id: str):
    """Delete a memory"""
    try:
        if user_id not in memory_instances:
            raise HTTPException(status_code=404, detail="User not found")
        
        memory = memory_instances[user_id]
        result = memory.delete(memory_id)
        
        return {"success": True, "message": "Memory deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
