import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to import mem0 and enhanced memory
try:
    from mem0 import Memory
    from mem0.memory.enhanced_memory import EnhancedMemory
    import google.generativeai as genai
    import json
    MEM0_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import mem0: {e}")
    MEM0_AVAILABLE = False

# LLM Configuration for categorization
GOOGLE_API_KEY = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google Gemini API configured for categorization")
else:
    logger.warning("Google API key not found - categorization will be disabled")

# Knowledge extraction prompt for categorization
KNOWLEDGE_EXTRACTION_PROMPT = """
You are an expert at analyzing conversations and extracting valuable information, responsible for an AI memory database related to the user that is using the system.
You will be presented with user chat information, documents and other content that you are responsible for extracting important information from, and storing to a memory database that will provide better and more contextual responses in the future chat conversations.
Your task is to:
1. Extract key facts, preferences, commitments, or other important information that are relevant to remember
2. Format each piece of information as a complete, self-contained statement with context
3. Only extract factual information or clear user preferences, not opinions or hypotheticals
4. Assign each piece of information a category:
   - person: Information about people or the user
   - work: Work, company or business related information
   - place: Information related to a Location or geography
   - event: Information about events or dates
   - task: Information about tasks or goals
   - general: Other important information that does not fit into the other categories

**IMPORTANT**: If the input contains only images, page headers, minimal text, or no meaningful/valuable content, return an empty array. Do NOT invent or hallucinate information that is not explicitly present in the input. Do not extract information that would not be helpful for future chat interactions.

Return a JSON array of extracted knowledge items, with each item having:
- content: The extracted knowledge in a complete sentence
- category: The assigned category
- confidence: A number from 0 to 1 indicating confidence level

Example:
User: "I need to finish the marketing report by Friday. Oh, and I'm moving to Seattle next month."
Output:
{
  "items": [
    {
      "content": "User needs to finish the marketing report by Friday",
      "category": "task",
      "confidence": 0.9
    },
    {
      "content": "User is moving to Seattle next month",
      "category": "place",
      "confidence": 0.85
    }
  ]
}

Example of minimal content:
User: "--- Page 1 --- ![image.jpg](image.jpg)"
Output:
{
  "items": []
}

Only include high-confidence information (>0.7). If there's nothing significant to extract, return an empty array.
"""

async def categorize_memory_content(content: str) -> str:
    """
    Use LLM to categorize memory content and return the most appropriate category.
    Returns 'general' as fallback if categorization fails.
    """
    if not GOOGLE_API_KEY:
        logger.warning("No Google API key available for categorization, using 'general'")
        return "general"

    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')

        prompt = f"{KNOWLEDGE_EXTRACTION_PROMPT}\n\nContent to analyze:\n{content}"

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        # Parse the JSON response
        if result_text.startswith('```json'):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith('```'):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        # Extract the category from the first high-confidence item
        if result.get("items") and len(result["items"]) > 0:
            for item in result["items"]:
                if item.get("confidence", 0) > 0.7:
                    category = item.get("category", "general")
                    logger.info(f"Categorized memory as '{category}' with confidence {item.get('confidence')}")
                    return category

        # No high-confidence categorization found
        logger.info("No high-confidence categorization found, using 'general'")
        return "general"

    except Exception as e:
        logger.error(f"Error during categorization: {e}")
        return "general"

app = FastAPI(
    title="Mem0 Enhanced REST APIs",
    description="A REST API for managing and searching memories with enhanced capabilities for your AI Agents and Apps.",
    version="1.0.0",
)

# CORS Configuration
origins = [
    "http://localhost:3000",  # Frontend Next.js app
    "http://localhost:3001",  # Alternative frontend port
    "http://localhost:8000",  # Self
    "http://localhost:8002",  # Alternative API port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory if available
memory = None
if MEM0_AVAILABLE:
    try:
        # Use the Doppler-provided key for Gemini
        api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

            # Configuration for mem0
            config = {
                "llm": {
                    "provider": "gemini",
                    "config": {
                        "model": os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite"),
                        "temperature": 0.2,
                        "max_tokens": 2000,
                    }
                },
                "embedder": {
                    "provider": "gemini",
                    "config": {
                        "model": os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
                        "embedding_dims": int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),  # Use 1536 dimensions as supported by Gemini
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

            # Try to create enhanced memory, fallback to regular memory
            try:
                memory = EnhancedMemory.from_config(config)
                logger.info("Enhanced memory initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced memory: {e}, falling back to regular memory")
                memory = Memory.from_config(config)
                logger.info("Regular memory initialized successfully")
        else:
            logger.warning("No Google API key found, memory will not be initialized")
    except Exception as e:
        logger.error(f"Failed to initialize memory: {e}")


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AddMemoryRequest(BaseModel):
    messages: List[Dict[str, str]]
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 10


class MemoryUpdateData(BaseModel):
    content: str = Field(..., description="The new content for the memory.")


@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any]):
    """Set memory configuration."""
    if not MEM0_AVAILABLE:
        raise HTTPException(status_code=503, detail="Mem0 is not available")
    return {"message": "Configuration set successfully", "status": "success"}


@app.post("/memories", summary="Create memories")
async def add_memory(memory_create: MemoryCreate):
    """Store new memories with automatic categorization."""
    if not MEM0_AVAILABLE or not memory:
        raise HTTPException(status_code=503, detail="Memory service is not available")

    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(
            status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required."
        )

    try:
        # Convert messages to the format expected by mem0
        messages = [{"role": msg.role, "content": msg.content} for msg in memory_create.messages]

        # Extract content for categorization (combine all message content)
        content_for_categorization = " ".join([msg.content for msg in memory_create.messages if msg.content])

        # Get category using LLM categorization
        category = await categorize_memory_content(content_for_categorization)

        # Enhance metadata with the determined category
        enhanced_metadata = memory_create.metadata or {}
        enhanced_metadata["category"] = category

        logger.info(f"Adding memory with category '{category}' for user {memory_create.user_id}")

        result = memory.add(
            messages=messages,
            user_id=memory_create.user_id,
            agent_id=memory_create.agent_id,
            run_id=memory_create.run_id,
            metadata=enhanced_metadata
        )

        return JSONResponse(content={
            "message": "Memory stored successfully",
            "status": "success",
            "result": result,
            "category": category
        })
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories/add", summary="Add memory (alternative endpoint)")
async def add_memory_route(req: AddMemoryRequest):
    """Alternative endpoint for adding memories with automatic categorization."""
    if not MEM0_AVAILABLE or not memory:
        raise HTTPException(status_code=503, detail="Memory service is not available")

    try:
        # Extract content for categorization (combine all message content)
        content_for_categorization = " ".join([msg.get("content", "") for msg in req.messages if msg.get("content")])

        # Get category using LLM categorization
        category = await categorize_memory_content(content_for_categorization)

        # Enhance metadata with the determined category
        enhanced_metadata = req.metadata or {}
        enhanced_metadata["category"] = category

        logger.info(f"Adding memory with category '{category}' for user {req.user_id}")

        result = memory.add(
            messages=req.messages,
            user_id=req.user_id,
            agent_id=req.agent_id,
            run_id=req.run_id,
            metadata=enhanced_metadata
        )
        return {"result": result, "category": category}
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", summary="Get memories")
def get_all_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: Optional[int] = 100
):
    """Retrieve stored memories."""
    if not MEM0_AVAILABLE or not memory:
        raise HTTPException(status_code=503, detail="Memory service is not available")

    if not any([user_id, agent_id, run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")

    try:
        memories = memory.get_all(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit
        )

        return {
            "memories": memories,
            "status": "success",
            "count": len(memories)
        }
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(memory_id: str):
    """Retrieve a specific memory by ID."""
    if not MEM0_AVAILABLE or not memory:
        raise HTTPException(status_code=503, detail="Memory service is not available")

    try:
        result = memory.get(memory_id)
        if not result:
            raise HTTPException(status_code=404, detail="Memory not found")

        return {
            "memory": result,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="Search memories")
def search_memories(search_req: SearchRequest):
    """Search for memories based on a query."""
    if not MEM0_AVAILABLE or not memory:
        raise HTTPException(status_code=503, detail="Memory service is not available")

    try:
        # Use enhanced search if available
        if hasattr(memory, 'search_with_fallback'):
            results = memory.search_with_fallback(
                query=search_req.query,
                user_id=search_req.user_id,
                agent_id=search_req.agent_id,
                run_id=search_req.run_id,
                limit=search_req.limit or 10,
                filters=search_req.filters
            )
        else:
            results = memory.search(
                query=search_req.query,
                user_id=search_req.user_id,
                agent_id=search_req.agent_id,
                run_id=search_req.run_id,
                limit=search_req.limit or 10,
                filters=search_req.filters
            )

        return {
            "results": results,
            "query": search_req.query,
            "status": "success",
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid", summary="Hybrid search memories")
def hybrid_search_memories(search_req: SearchRequest):
    """Hybrid search for memories (enhanced feature)."""
    if not MEM0_AVAILABLE or not memory:
        raise HTTPException(status_code=503, detail="Memory service is not available")

    if not hasattr(memory, 'hybrid_search'):
        raise HTTPException(status_code=501, detail="Hybrid search not available")

    try:
        results = memory.hybrid_search(
            query=search_req.query,
            user_id=search_req.user_id,
            agent_id=search_req.agent_id,
            run_id=search_req.run_id,
            limit=search_req.limit or 10,
            filters=search_req.filters
        )

        return {
            "results": results,
            "query": search_req.query,
            "status": "success",
            "search_type": "hybrid",
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/stats", summary="Get memory statistics")
def get_memory_stats(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None
):
    """Get memory statistics (enhanced feature)."""
    if not MEM0_AVAILABLE or not memory:
        raise HTTPException(status_code=503, detail="Memory service is not available")

    if not hasattr(memory, 'get_memory_stats'):
        raise HTTPException(status_code=501, detail="Memory stats not available")

    try:
        stats = memory.get_memory_stats(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id
        )

        return {
            "stats": stats,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory_endpoint(memory_id: str, update_data: MemoryUpdateData):
    """Update an existing memory's content."""
    if not MEM0_AVAILABLE or not memory:
        raise HTTPException(status_code=503, detail="Memory service is not available")

    try:
        result = memory.update(memory_id, update_data.content)
        return {
            "memory_id": memory_id,
            "result": result,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(memory_id: str):
    """Retrieve memory history."""
    if not MEM0_AVAILABLE or not memory:
        raise HTTPException(status_code=503, detail="Memory service is not available")

    try:
        history = memory.history(memory_id)
        return {
            "memory_id": memory_id,
            "history": history,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error retrieving memory history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}", summary="Delete a memory")
def delete_memory(memory_id: str):
    """Delete a specific memory by ID."""
    if not MEM0_AVAILABLE or not memory:
        raise HTTPException(status_code=503, detail="Memory service is not available")

    try:
        result = memory.delete(memory_id)
        return {
            "message": "Memory deleted successfully",
            "memory_id": memory_id,
            "result": result,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories", summary="Delete all memories")
def delete_all_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
):
    """Delete all memories for a given identifier."""
    if not MEM0_AVAILABLE or not memory:
        raise HTTPException(status_code=503, detail="Memory service is not available")

    if not any([user_id, agent_id, run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")

    try:
        result = memory.delete_all(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id
        )
        return {
            "message": "All relevant memories deleted successfully",
            "result": result,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error deleting all memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", summary="Reset all memories")
def reset_memory():
    """Completely reset stored memories."""
    if not MEM0_AVAILABLE or not memory:
        raise HTTPException(status_code=503, detail="Memory service is not available")

    try:
        result = memory.reset()
        return {
            "message": "All memories reset successfully",
            "result": result,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error resetting memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", summary="Health check", include_in_schema=False)
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "mem0-enhanced",
        "version": "1.0.0",
        "mem0_available": MEM0_AVAILABLE,
        "memory_initialized": memory is not None
    }


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url='/docs')
