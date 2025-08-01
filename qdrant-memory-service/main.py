import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import os

from models import (
    MemoryCreate, MemoryUpdate, MemoryDelete, MemoryBatchDelete, MemorySearch,
    MemoryResponse, MemoryListResponse, MemorySearchResponse, BatchDeleteResponse, HealthResponse,
    GraphEntityResponse, GraphRelationshipResponse, GraphSummaryResponse,
    GraphSearchRequest, GraphSearchResponse, HybridSearchRequest, MemoryAnalyticsResponse
)
from qdrant_memory_client import QdrantMemoryClient
from gemini_client import GeminiClient
from neo4j_graph_client import Neo4jGraphClient
from graph_extractor import GraphExtractor
from hybrid_search import HybridSearchService
from memory_analytics import MemoryAnalytics
from config import API_HOST, API_PORT
from pydantic import BaseModel, Field, ValidationError
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Memory Service with Graph and Analytics",
    description="Advanced memory service with graph memory, hybrid search, and analytics",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
qdrant_client = QdrantMemoryClient()
gemini_client = GeminiClient()

# Initialize Neo4j client if available
graph_client = None
graph_extractor = None
hybrid_search_service = None
memory_analytics = None

try:
    neo4j_host = os.getenv('NEO4J_HOST', 'localhost')
    neo4j_port = os.getenv('NEO4J_PORT', '7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
    neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')

    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"

    graph_client = Neo4jGraphClient(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
    graph_client.connect()

    graph_extractor = GraphExtractor(gemini_client)
    hybrid_search_service = HybridSearchService(qdrant_client, graph_client, graph_extractor)
    memory_analytics = MemoryAnalytics(gemini_client)

    logger.info("Graph memory and analytics services initialized successfully")

except Exception as e:
    logger.warning(f"Graph services not available: {e}")
    logger.info("Running with basic vector search only")
    memory_analytics = MemoryAnalytics(gemini_client)  # Analytics can work without graph

class BatchMemoryItem(BaseModel):
    content: str = Field(..., description="Memory content")
    user_id: str = Field(..., description="User ID")
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BatchMemoryResult(BaseModel):
    id: Optional[str] = None
    status: str
    error: Optional[str] = None
    memory: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        qdrant_health = qdrant_client.health_check()

        # Check graph status
        graph_status = "not_configured"
        graph_connected = False

        if graph_client:
            try:
                graph_connected = graph_client.connected
                graph_status = "connected" if graph_connected else "disconnected"
            except Exception:
                graph_status = "error"

        return HealthResponse(
            qdrant_status=qdrant_health["qdrant_status"],
            collection_exists=qdrant_health["collection_exists"],
            graph_status=graph_status,
            graph_connected=graph_connected
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/memories", response_model=MemoryListResponse)
async def add_memories(request: MemoryCreate):
    """Add new memories from messages with deduplication."""
    try:
        logger.info(f"Adding memories for user: {request.user_id}")

        # Extract memories from messages
        memories = gemini_client.extract_memories_from_messages(request.messages)

        if not memories:
            logger.info("No memorable information extracted")
            return MemoryListResponse(results=[], count=0)

        added_memories = []
        updated_memories = []

        # OPTIMIZATION: Process memories in batches for better performance
        batch_size = 5  # Process 5 memories at a time
        for i in range(0, len(memories), batch_size):
            batch = memories[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(memories) + batch_size - 1)//batch_size} ({len(batch)} memories)")

            for memory_text in batch:
                try:
                    # Check for similar existing memories before adding
                    logger.info(f"🔍 Checking for similar memories to: '{memory_text[:50]}...'")

                    try:
                        similar_memories = await _find_similar_memories(memory_text, request.user_id)
                        logger.info(f"🔍 Found {len(similar_memories)} similar memories")
                    except Exception as dedup_error:
                        logger.error(f"❌ Error in deduplication check: {dedup_error}")
                        similar_memories = []

                    if similar_memories:
                        # Found similar memory - update it instead of creating new
                        most_similar = similar_memories[0]  # Get the most similar
                        logger.info(f"✅ Found similar memory {most_similar['id']} for '{memory_text[:50]}...' (similarity: {most_similar['score']:.3f})")

                        try:
                            # Update the existing memory with new information
                            updated_memory = await _update_memory_with_new_info(most_similar['id'], memory_text, request.user_id)
                            updated_memories.append(updated_memory)
                        except Exception as update_error:
                            logger.error(f"❌ Error updating memory: {update_error}")
                            # Fall back to adding new memory
                            similar_memories = []

                    if not similar_memories:
                        # No similar memory found - add new memory
                        logger.info(f"🆕 No similar memory found for '{memory_text[:50]}...' - adding new")

                        # Generate embedding
                        embedding = gemini_client.generate_embedding(memory_text)

                        # Categorize memory
                        category = gemini_client.categorize_memory(memory_text)

                        # Prepare metadata
                        metadata = {
                            "category": category,
                            **(request.metadata or {})
                        }

                        if request.agent_id:
                            metadata["agent_id"] = request.agent_id
                        if request.run_id:
                            metadata["run_id"] = request.run_id

                        # Add to Qdrant
                        memory_id = qdrant_client.add_memory(
                            memory=memory_text,
                            vector=embedding,
                            user_id=request.user_id,
                            metadata=metadata
                        )

                        # OPTIMIZATION: Make graph extraction optional and async
                        # Only do graph extraction if explicitly requested or for important memories
                        should_extract_graph = (
                            graph_client and
                            graph_extractor and
                            graph_client.connected and
                            request.metadata and
                            request.metadata.get("extract_graph", False)  # Only if explicitly requested
                        )

                        if should_extract_graph:
                            try:
                                # OPTIMIZATION: Run graph extraction asynchronously to not block memory storage
                                import asyncio
                                asyncio.create_task(
                                    _extract_graph_async(memory_text, request.user_id, memory_id)
                                )
                                logger.info(f"Queued graph extraction for memory {memory_id}")
                            except Exception as e:
                                logger.warning(f"Failed to queue graph extraction for memory {memory_id}: {e}")

                        # Create response object
                        memory_response = MemoryResponse(
                            id=memory_id,
                            memory=memory_text,
                            hash="",  # Will be generated by Qdrant client
                            metadata=metadata,
                            created_at=datetime.now(),  # Set current timestamp
                            user_id=request.user_id
                        )
                        added_memories.append(memory_response)

                except Exception as e:
                    logger.error(f"Error processing memory '{memory_text[:50]}...': {e}")
                    continue

        # Combine added and updated memories for response
        all_memories = added_memories + updated_memories

        logger.info(f"Successfully processed {len(added_memories)} new memories and {len(updated_memories)} updated memories")
        return MemoryListResponse(results=all_memories, count=len(all_memories))

    except Exception as e:
        logger.error(f"Error adding memories: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding memories: {str(e)}")

async def _find_similar_memories(memory_text: str, user_id: str, similarity_threshold: float = 0.95) -> List[Dict]:
    """Find similar existing memories for deduplication."""
    try:
        # Search for similar memories using semantic search
        search_results = qdrant_client.search_memories_by_text_with_threshold(
            query=memory_text,
            user_id=user_id,
            limit=5,  # Check top 5 most similar
            score_threshold=similarity_threshold
        )

        return search_results
    except Exception as e:
        logger.error(f"Error finding similar memories: {e}")
        return []

async def _update_memory_with_new_info(memory_id: str, new_memory_text: str, user_id: str) -> MemoryResponse:
    """Update existing memory with new information."""
    try:
        # Get existing memory
        existing_memory = qdrant_client.get_memory(memory_id)
        if not existing_memory:
            raise ValueError(f"Memory {memory_id} not found")

        # Combine old and new information intelligently
        combined_text = _combine_memory_texts(existing_memory.get('memory', ''), new_memory_text)

        # Generate new embedding for combined text
        new_embedding = gemini_client.generate_embedding(combined_text)

        # Update the memory in Qdrant
        qdrant_client.update_memory(
            memory_id=memory_id,
            memory=combined_text,
            vector=new_embedding,
            user_id=user_id
        )

        # Return updated memory response
        return MemoryResponse(
            id=memory_id,
            memory=combined_text,
            hash="",
            metadata=existing_memory.get('metadata', {}),
            created_at=existing_memory.get('created_at'),
            updated_at=datetime.now(),
            user_id=user_id
        )

    except Exception as e:
        logger.error(f"Error updating memory {memory_id}: {e}")
        raise

def _combine_memory_texts(old_text: str, new_text: str) -> str:
    """Keep memories separate - return the new text only to avoid violating 150 char limit."""
    # IMPORTANT: Don't combine memories as this creates long strings that violate our rules
    # Just return the new text
    logger.info(f"Memory update: replacing '{old_text}' with '{new_text}'")
    return new_text

async def _extract_graph_async(memory_text: str, user_id: str, memory_id: str):
    """Extract graph entities and relationships asynchronously."""
    try:
        entities, relationships = graph_extractor.extract_entities_and_relationships(
            memory_text, user_id
        )

        # Add entities to graph
        for entity in entities:
            graph_client.add_entity(entity, user_id, memory_id)

        # Add relationships to graph
        for relationship in relationships:
            graph_client.add_relationship(relationship, user_id, memory_id)

        logger.info(f"Async graph extraction completed for memory {memory_id}: {len(entities)} entities, {len(relationships)} relationships")

    except Exception as e:
        logger.warning(f"Async graph extraction failed for memory {memory_id}: {e}")

@app.post("/memories/bulk", response_model=MemoryListResponse)
async def add_memories_bulk(request: MemoryCreate):
    """Add new memories from messages with optimized bulk processing (no graph extraction by default)."""
    try:
        logger.info(f"Adding memories in bulk for user: {request.user_id}")

        # Extract memories from messages
        memories = gemini_client.extract_memories_from_messages(request.messages)

        if not memories:
            logger.info("No memorable information extracted")
            return MemoryListResponse(results=[], count=0)

        added_memories = []

        # OPTIMIZATION: Process all memories without graph extraction for speed
        logger.info(f"Processing {len(memories)} memories in bulk mode (no graph extraction)")

        for memory_text in memories:
            try:
                # Generate embedding
                embedding = gemini_client.generate_embedding(memory_text)

                # Categorize memory
                category = gemini_client.categorize_memory(memory_text)

                # Prepare metadata
                metadata = {
                    "category": category,
                    "processing_mode": "bulk",  # Mark as bulk processed
                    **(request.metadata or {})
                }

                if request.agent_id:
                    metadata["agent_id"] = request.agent_id
                if request.run_id:
                    metadata["run_id"] = request.run_id

                # Add to Qdrant
                memory_id = qdrant_client.add_memory(
                    memory=memory_text,
                    vector=embedding,
                    user_id=request.user_id,
                    metadata=metadata
                )

                # Create response object
                memory_response = MemoryResponse(
                    id=memory_id,
                    memory=memory_text,
                    hash="",  # Will be generated by Qdrant client
                    metadata=metadata,
                    created_at=datetime.now(),  # Set current timestamp
                    user_id=request.user_id
                )
                added_memories.append(memory_response)

            except Exception as e:
                logger.error(f"Error processing memory '{memory_text}': {e}")
                continue

        logger.info(f"Successfully added {len(added_memories)} memories in bulk mode")
        return MemoryListResponse(results=added_memories, count=len(added_memories))

    except Exception as e:
        logger.error(f"Error adding memories in bulk: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding memories in bulk: {str(e)}")

@app.post("/memories/batch", response_model=List[BatchMemoryResult])
async def add_memories_batch(
    items: List[BatchMemoryItem] = Body(..., description="List of memories to add (max 50)")
):
    """Add multiple memories in a single batch, with full categorization and metadata support."""
    if not items or len(items) == 0:
        raise HTTPException(status_code=400, detail="No memories provided.")
    if len(items) > 50:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit (50).")

    async def process_memory(item: BatchMemoryItem) -> BatchMemoryResult:
        try:
            # Generate embedding
            embedding = await asyncio.to_thread(gemini_client.generate_embedding, item.content)
            # Categorize memory
            category = await asyncio.to_thread(gemini_client.categorize_memory, item.content)
            # Prepare metadata
            metadata = dict(item.metadata) if item.metadata else {}
            metadata["category"] = category
            if item.agent_id:
                metadata["agent_id"] = item.agent_id
            if item.run_id:
                metadata["run_id"] = item.run_id
            # Add to Qdrant
            memory_id = await asyncio.to_thread(
                qdrant_client.add_memory,
                memory=item.content,
                vector=embedding,
                user_id=item.user_id,
                metadata=metadata
            )
            # Optionally trigger graph extraction
            should_extract_graph = (
                graph_client and graph_extractor and graph_client.connected and
                metadata.get("extract_graph", False)
            )
            if should_extract_graph:
                try:
                    asyncio.create_task(_extract_graph_async(item.content, item.user_id, memory_id))
                except Exception as e:
                    # Log but do not fail the memory
                    logger.warning(f"Failed to queue graph extraction for memory {memory_id}: {e}")
            return BatchMemoryResult(
                id=memory_id,
                status="success",
                memory=item.content,
                category=category,
                created_at=datetime.now().isoformat(),
                user_id=item.user_id,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Batch memory error for user {item.user_id}: {e}")
            return BatchMemoryResult(
                status="error",
                error=str(e),
                memory=item.content,
                user_id=item.user_id,
                metadata=item.metadata
            )

    # Process all memories in parallel
    results = await asyncio.gather(*(process_memory(item) for item in items))
    return results

@app.get("/memories", response_model=MemoryListResponse)
async def get_memories(
    user_id: str,
    limit: int = 50,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None
):
    """Get all memories for a user."""
    try:
        logger.info(f"Getting memories for user: {user_id}, limit: {limit}")

        # Build filters
        filters = {}
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        # Get memories from Qdrant
        memories = qdrant_client.get_memories(
            user_id=user_id,
            limit=limit,
            filters=filters if filters else None
        )

        logger.info(f"Retrieved {len(memories)} memories")
        return MemoryListResponse(results=memories, count=len(memories))

    except Exception as e:
        logger.error(f"Error getting memories: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting memories: {str(e)}")

@app.post("/search", response_model=MemorySearchResponse)
async def search_memories(request: MemorySearch):
    """Search memories using vector similarity."""
    try:
        logger.info(f"Searching memories for user: {request.user_id}, query: {request.query}")

        # Generate embedding for search query
        query_embedding = gemini_client.generate_embedding(request.query)

        # Build filters
        filters = {}
        if request.agent_id:
            filters["agent_id"] = request.agent_id
        if request.run_id:
            filters["run_id"] = request.run_id
        if request.filters:
            filters.update(request.filters)

        # Search in Qdrant
        memories = qdrant_client.search_memories(
            query_vector=query_embedding,
            user_id=request.user_id,
            limit=request.limit,
            filters=filters if filters else None
        )

        logger.info(f"Found {len(memories)} memories for search query")
        return MemorySearchResponse(results=memories, count=len(memories))

    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching memories: {str(e)}")

@app.put("/memories/{memory_id}", response_model=dict)
async def update_memory(memory_id: str, request: MemoryUpdate, user_id: Optional[str] = Query(None)):
    """Update a memory."""
    try:
        # Use user_id from request body if provided, otherwise from query parameter
        effective_user_id = request.user_id or user_id
        
        if not effective_user_id:
            raise HTTPException(status_code=400, detail="user_id is required either in request body or as query parameter")
        
        logger.info(f"Updating memory: {memory_id} for user: {effective_user_id}")

        # Generate new embedding
        embedding = gemini_client.generate_embedding(request.data)

        # Categorize updated memory
        category = gemini_client.categorize_memory(request.data)

        # Update in Qdrant
        success = qdrant_client.update_memory(
            memory_id=memory_id,
            user_id=effective_user_id,
            new_memory=request.data,
            new_vector=embedding,
            metadata={"category": category}
        )

        if success:
            logger.info(f"Successfully updated memory: {memory_id}")
            return {"status": "success", "message": "Memory updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Memory not found or access denied")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating memory: {str(e)}")

@app.delete("/memories/batch", response_model=BatchDeleteResponse)
async def batch_delete_memories(request: MemoryBatchDelete):
    """Batch delete multiple memories."""
    try:
        user_id = request.user_id
        logger.info(f"Batch deleting {len(request.memory_ids)} memories for user: {user_id}")

        result = qdrant_client.batch_delete_memories(
            memory_ids=request.memory_ids,
            user_id=user_id
        )

        logger.info(f"Batch delete completed: {result['deleted_count']} deleted, {result['failed_count']} failed")

        return BatchDeleteResponse(
            deleted_count=result["deleted_count"],
            failed_count=result["failed_count"],
            deleted_ids=result["deleted_ids"],
            failed_ids=result["failed_ids"],
            status="success"
        )

    except Exception as e:
        logger.error(f"Error in batch delete: {e}")
        raise HTTPException(status_code=500, detail=f"Error in batch delete: {str(e)}")

@app.delete("/memories/{memory_id}", response_model=dict)
async def delete_memory(memory_id: str, user_id: str):
    """Delete a memory."""
    try:
        logger.info(f"Deleting memory: {memory_id} for user: {user_id}")

        success = qdrant_client.delete_memory(memory_id=memory_id, user_id=user_id)

        if success:
            logger.info(f"Successfully deleted memory: {memory_id}")
            return {"status": "success", "message": "Memory deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Memory not found or access denied")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting memory: {str(e)}")

@app.delete("/memories", response_model=dict)
async def delete_all_memories(user_id: str = Query(..., description="User ID to delete all memories for")):
    """Delete all memories for a user."""
    try:
        logger.info(f"Deleting all memories for user: {user_id}")

        result = qdrant_client.delete_all_memories(user_id=user_id)

        logger.info(f"Delete all completed: {result['deleted_count']} memories deleted for user {user_id}")

        return {
            "status": "success",
            "deleted_count": result["deleted_count"],
            "message": result["message"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting all memories: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting all memories: {str(e)}")

# Graph Memory Endpoints
@app.get("/graph/entities", response_model=List[GraphEntityResponse])
async def get_graph_entities(user_id: str, limit: int = 50):
    """Get all graph entities for a user."""
    if not graph_client or not graph_client.connected:
        raise HTTPException(status_code=503, detail="Graph services not available")

    try:
        entities = graph_client.get_entities_by_user(user_id, limit)
        return [GraphEntityResponse(**entity) for entity in entities]
    except Exception as e:
        logger.error(f"Error getting graph entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/summary", response_model=GraphSummaryResponse)
async def get_graph_summary(user_id: str):
    """Get graph summary statistics for a user."""
    if not graph_client or not graph_client.connected:
        raise HTTPException(status_code=503, detail="Graph services not available")

    try:
        summary = graph_client.get_graph_summary(user_id)
        return GraphSummaryResponse(**summary)
    except Exception as e:
        logger.error(f"Error getting graph summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph/search", response_model=GraphSearchResponse)
async def search_graph(request: GraphSearchRequest):
    """Search the knowledge graph."""
    if not graph_client or not graph_client.connected:
        raise HTTPException(status_code=503, detail="Graph services not available")

    try:
        entities = []
        relationships = []
        related_memories = []

        if request.entity_id:
            # Search from specific entity
            entities = graph_client.search_related_entities(
                request.entity_id, request.user_id, request.max_depth
            )
            relationships = graph_client.get_entity_relationships(
                request.entity_id, request.user_id
            )
        else:
            # Extract entities from query and search
            search_entities = graph_extractor.extract_search_entities(request.query)
            all_entities = graph_client.get_entities_by_user(request.user_id, limit=100)

            for entity in all_entities:
                entity_name = entity.get('name', '').lower()
                for search_term in search_entities:
                    if search_term in entity_name:
                        entities.append(entity)
                        break

        # Convert to response models
        entity_responses = [GraphEntityResponse(**entity) for entity in entities]
        relationship_responses = [GraphRelationshipResponse(**rel) for rel in relationships]

        return GraphSearchResponse(
            entities=entity_responses,
            relationships=relationship_responses,
            related_memories=related_memories
        )
    except Exception as e:
        logger.error(f"Error in graph search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Hybrid Search Endpoints
@app.post("/search/hybrid", response_model=MemorySearchResponse)
async def perform_hybrid_search(request: HybridSearchRequest):
    """Perform hybrid search across multiple modalities."""
    try:
        if hybrid_search_service:
            memories = hybrid_search_service.search(request)
        else:
            # Fallback to basic semantic search
            memories = qdrant_client.search_memories_by_text(
                query=request.query,
                user_id=request.user_id,
                limit=request.limit
            )

        return MemorySearchResponse(results=memories, count=len(memories))
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/suggestions")
async def get_search_suggestions(query: str, user_id: str, limit: int = 5):
    """Get search suggestions based on partial query."""
    try:
        if hybrid_search_service:
            suggestions = hybrid_search_service.get_search_suggestions(query, user_id, limit)
        else:
            suggestions = []

        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Error getting search suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Memory Analytics Endpoints
@app.get("/analytics/memories", response_model=MemoryAnalyticsResponse)
async def get_memory_analytics(user_id: str):
    """Get comprehensive memory analytics for a user."""
    try:
        if not memory_analytics:
            raise HTTPException(status_code=503, detail="Analytics services not available")

        # Get all user memories
        memories = qdrant_client.get_memories(user_id, limit=1000)

        # Convert to dict format for analytics
        memory_dicts = []
        for mem in memories:
            memory_dict = {
                'id': mem.id,
                'memory': mem.memory,
                'metadata': mem.metadata,
                'created_at': mem.created_at,
                'score': mem.score
            }
            memory_dicts.append(memory_dict)

        # Perform analytics
        analytics = memory_analytics.analyze_memories(memory_dicts, user_id)
        return analytics

    except Exception as e:
        logger.error(f"Error in memory analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/trends")
async def get_memory_trends(user_id: str, days: int = 30):
    """Get memory creation trends over time."""
    try:
        if not memory_analytics:
            raise HTTPException(status_code=503, detail="Analytics services not available")

        # Get recent memories
        memories = qdrant_client.get_memories(user_id, limit=1000)

        # Convert to dict format
        memory_dicts = []
        for mem in memories:
            memory_dict = {
                'id': mem.id,
                'memory': mem.memory,
                'metadata': mem.metadata,
                'created_at': mem.created_at,
                'score': mem.score
            }
            memory_dicts.append(memory_dict)

        trends = memory_analytics.get_memory_trends(memory_dicts, days)
        return trends

    except Exception as e:
        logger.error(f"Error getting memory trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/similar")
async def find_similar_memories(user_id: str, memory_text: str, threshold: float = 0.3, limit: int = 5):
    """Find memories similar to given text."""
    try:
        if not memory_analytics:
            raise HTTPException(status_code=503, detail="Analytics services not available")

        # Get all user memories
        memories = qdrant_client.get_memories(user_id, limit=1000)

        # Convert to dict format
        memory_dicts = []
        for mem in memories:
            memory_dict = {
                'id': mem.id,
                'memory': mem.memory,
                'metadata': mem.metadata,
                'created_at': mem.created_at,
                'score': mem.score
            }
            memory_dicts.append(memory_dict)

        similar_memories = memory_analytics.find_similar_memories(
            memory_text, memory_dicts, threshold, limit
        )
        return {"similar_memories": similar_memories}

    except Exception as e:
        logger.error(f"Error finding similar memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Enhanced Memory Service on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
