import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse
import hashlib
from pydantic import BaseModel

def parse_datetime_safe(dt_value):
    """Parse datetime value with safe timezone handling."""
    import logging
    if not dt_value:
        return None

    # If it's already a datetime object, handle timezone properly and return ISO string
    if isinstance(dt_value, datetime):
        # If datetime is naive (no timezone), assume UTC
        if dt_value.tzinfo is None:
            dt_value = dt_value.replace(tzinfo=timezone.utc)
        return dt_value.isoformat()

    # If it's a string, try to parse it and return ISO string
    if isinstance(dt_value, str):
        try:
            # Try parsing with timezone info
            dt = datetime.fromisoformat(dt_value)
            # If parsed datetime is naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            try:
                # Try parsing without timezone info
                dt = datetime.fromisoformat(dt_value.replace('Z', '+00:00'))
                return dt.isoformat()
            except ValueError:
                try:
                    # Fallback to naive datetime, assume UTC
                    dt = datetime.fromisoformat(dt_value.replace('Z', ''))
                    dt = dt.replace(tzinfo=timezone.utc)
                    return dt.isoformat()
                except Exception as e:
                    logging.error(f"parse_datetime_safe: Failed to parse value '{dt_value}' (type: {type(dt_value)}): {e}")
                    return None

    # If it's something else, log and return None
    logging.error(f"parse_datetime_safe: Unexpected type for value '{dt_value}': {type(dt_value)}")
    return None

from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, VECTOR_DIMENSION
from models import MemoryResponse

logger = logging.getLogger(__name__)

class QdrantMemoryClient:
    def __init__(self):
        # Use HTTPS for Qdrant connection since TLS is enabled
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, https=True, verify=False)
        self.collection_name = QDRANT_COLLECTION_NAME
        self.create_collection()

    def create_collection(self):
        """Create the memory collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)

            if collection_exists:
                logger.info(f"Collection {self.collection_name} already exists. Reusing existing collection.")
                return

            # Create new collection only if it doesn't exist
            logger.info(f"Collection {self.collection_name} does not exist. Creating new collection with {VECTOR_DIMENSION} dimensions...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Successfully created collection {self.collection_name}")

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def add_memory(self, memory: str, vector: List[float], user_id: str,
                   metadata: Dict[str, Any]) -> str:
        """Add a memory to Qdrant."""
        try:
            # Generate unique ID and hash
            memory_id = str(uuid.uuid4())
            memory_hash = hashlib.md5(memory.encode()).hexdigest()

            # Prepare payload
            payload = {
                "memory": memory,
                "hash": memory_hash,
                "user_id": user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": None,
                **metadata
            }

            # Create point
            point = PointStruct(
                id=memory_id,
                vector=vector,
                payload=payload
            )

            # Insert into Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.info(f"Memory {memory_id} added successfully")
            return memory_id

        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise

    def search_memories_by_text(self, query: str, user_id: str, limit: int = 10) -> List[MemoryResponse]:
        """Search memories by generating embedding from text query."""
        try:
            from gemini_client import GeminiClient
            gemini_client = GeminiClient()
            query_vector = gemini_client.generate_embedding(query)
            return self.search_memories(query_vector, user_id, limit)
        except Exception as e:
            logger.error(f"Error in text-based search: {e}")
            return []

    def search_memories_by_vector(self, query_vector: List[float], user_id: str,
                       limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[MemoryResponse]:
        """Search memories using vector similarity."""
        return self.search_memories(query_vector, user_id, limit, filters)

    def search_memories(self, query_vector: List[float], user_id: str,
                       limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[MemoryResponse]:
        """Search memories using vector similarity."""
        try:
            # Build filter for user_id
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )

            # Add additional filters if provided
            if filters:
                for key, value in filters.items():
                    search_filter.must.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )

            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )

            # Convert to MemoryResponse objects
            memories = []
            for point in search_result:
                # Extract similarity score from Qdrant
                score_value = float(point.score) if point.score is not None else None

                # Handle both "memory" and "data" keys for backward compatibility
                memory_text = point.payload.get("memory") or point.payload.get("data") or "Unknown memory"

                memory = MemoryResponse(
                    id=str(point.id),
                    memory=memory_text,
                    hash=point.payload.get("hash", ""),  # Use get() with default empty string
                    metadata={k: v for k, v in point.payload.items()
                             if k not in ["memory", "data", "hash", "user_id", "created_at", "updated_at"]},
                    created_at=parse_datetime_safe(point.payload.get("created_at", datetime.now(timezone.utc).isoformat())),
                    updated_at=parse_datetime_safe(point.payload.get("updated_at")),
                    user_id=point.payload["user_id"],
                    score=score_value  # Include similarity score from Qdrant
                )
                memories.append(memory)

            logger.info(f"Found {len(memories)} memories for user {user_id}")
            return memories

        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def search_memories_by_text_with_threshold(self, query: str, user_id: str, limit: int = 10, score_threshold: float = 0.0) -> List[Dict]:
        """Search memories by text query with similarity threshold."""
        try:
            from gemini_client import GeminiClient
            gemini_client = GeminiClient()
            query_vector = gemini_client.generate_embedding(query)

            # Build filter for user_id
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )

            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )

            # Convert to dictionary format and filter by score threshold
            memories = []
            for point in search_result:
                score_value = float(point.score) if point.score is not None else 0.0

                # Only include memories above the threshold
                if score_value >= score_threshold:
                    memory_text = point.payload.get("memory") or point.payload.get("data") or "Unknown memory"

                    memory_dict = {
                        "id": str(point.id),
                        "memory": memory_text,
                        "hash": point.payload.get("hash", ""),  # Use get() with default empty string
                        "metadata": {k: v for k, v in point.payload.items()
                                   if k not in ["memory", "data", "hash", "user_id", "created_at", "updated_at"]},
                        "created_at": parse_datetime_safe(point.payload.get("created_at", datetime.now(timezone.utc).isoformat())),
                        "updated_at": parse_datetime_safe(point.payload.get("updated_at")),
                        "user_id": point.payload["user_id"],
                        "score": score_value
                    }
                    memories.append(memory_dict)

            logger.info(f"Found {len(memories)} memories above threshold {score_threshold} for user {user_id}")
            return memories

        except Exception as e:
            logger.error(f"Error searching memories by text: {e}")
            return []

    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """Get a single memory by ID."""
        try:
            # Retrieve the point from Qdrant
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True
            )

            if not points:
                logger.warning(f"Memory {memory_id} not found")
                return None

            point = points[0]
            memory_text = point.payload.get("memory") or point.payload.get("data") or "Unknown memory"

            memory_dict = {
                "id": str(point.id),
                "memory": memory_text,
                "hash": point.payload.get("hash", ""),  # Use get() with default empty string
                "metadata": {k: v for k, v in point.payload.items()
                           if k not in ["memory", "data", "hash", "user_id", "created_at", "updated_at"]},
                "created_at": parse_datetime_safe(point.payload.get("created_at", datetime.now(timezone.utc).isoformat())),
                "updated_at": parse_datetime_safe(point.payload.get("updated_at")),
                "user_id": point.payload["user_id"]
            }

            return memory_dict

        except Exception as e:
            logger.error(f"Error getting memory {memory_id}: {e}")
            return None

    def get_memories(self, user_id: str, limit: int = 50,
                    filters: Optional[Dict[str, Any]] = None) -> List[MemoryResponse]:
        """Get all memories for a user."""
        try:
            # Build filter for user_id
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )

            # Add additional filters if provided
            if filters:
                for key, value in filters.items():
                    search_filter.must.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )

            # Scroll through all points
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True
            )

            # Convert to MemoryResponse objects
            memories = []
            for point in points:
                # Handle both "memory" and "data" keys for backward compatibility
                memory_text = point.payload.get("memory") or point.payload.get("data") or "Unknown memory"

                memory = MemoryResponse(
                    id=str(point.id),
                    memory=memory_text,
                    hash=point.payload.get("hash", ""),  # Use get() with default empty string
                    metadata={k: v for k, v in point.payload.items()
                             if k not in ["memory", "data", "hash", "user_id", "created_at", "updated_at"]},
                    created_at=parse_datetime_safe(point.payload.get("created_at", datetime.now(timezone.utc).isoformat())),
                    updated_at=parse_datetime_safe(point.payload.get("updated_at")),
                    user_id=point.payload["user_id"]
                )
                memories.append(memory)

            # Sort by created_at descending
            memories.sort(key=lambda x: x.created_at, reverse=True)

            logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
            return memories

        except Exception as e:
            logger.error(f"Error getting memories: {e}")
            raise

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory."""
        try:
            # Verify the memory belongs to the user
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True
            )

            if not points or points[0].payload.get("user_id") != user_id:
                logger.warning(f"Memory {memory_id} not found or doesn't belong to user {user_id}")
                return False

            # Delete the point
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[memory_id]
            )

            logger.info(f"Memory {memory_id} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise

    def batch_delete_memories(self, memory_ids: List[str], user_id: str) -> Dict[str, Any]:
        """Batch delete multiple memories."""
        try:
            deleted_ids = []
            failed_ids = []

            # First, validate memory IDs and filter out invalid ones
            valid_memory_ids = []
            for memory_id in memory_ids:
                try:
                    # Try to parse as UUID to validate format
                    uuid.UUID(memory_id)
                    valid_memory_ids.append(memory_id)
                except ValueError:
                    failed_ids.append(memory_id)
                    logger.warning(f"Invalid memory ID format: {memory_id}")

            # If no valid IDs, return early
            if not valid_memory_ids:
                logger.info("No valid memory IDs provided for batch delete")
                return {
                    "deleted_count": 0,
                    "failed_count": len(failed_ids),
                    "deleted_ids": [],
                    "failed_ids": failed_ids
                }

            # Retrieve and verify memories belong to the user
            try:
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=valid_memory_ids,
                    with_payload=True
                )

                # Check which memories exist and belong to the user
                user_owned_ids = []
                for point in points:
                    if point.payload.get("user_id") == user_id:
                        user_owned_ids.append(point.id)
                    else:
                        failed_ids.append(point.id)
                        logger.warning(f"Memory {point.id} doesn't belong to user {user_id}")

                # Add non-existent memory IDs to failed list
                existing_ids = {point.id for point in points}
                for memory_id in valid_memory_ids:
                    if memory_id not in existing_ids:
                        failed_ids.append(memory_id)
                        logger.warning(f"Memory {memory_id} not found")

                # Batch delete user-owned memories
                if user_owned_ids:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=user_owned_ids
                    )
                    deleted_ids = user_owned_ids
                    logger.info(f"Successfully deleted {len(deleted_ids)} memories")

            except Exception as retrieve_error:
                logger.error(f"Error retrieving memories for batch delete: {retrieve_error}")
                # If retrieval fails, add all valid IDs to failed list
                failed_ids.extend(valid_memory_ids)

            return {
                "deleted_count": len(deleted_ids),
                "failed_count": len(failed_ids),
                "deleted_ids": deleted_ids,
                "failed_ids": failed_ids
            }

        except Exception as e:
            logger.error(f"Error in batch delete: {e}")
            raise

    def update_memory(self, memory_id: str, memory: str, vector: List[float], user_id: str) -> bool:
        """Update an existing memory."""
        try:
            # Get existing memory to preserve metadata
            existing_points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True
            )

            if not existing_points:
                logger.error(f"Memory {memory_id} not found for update")
                return False

            existing_payload = existing_points[0].payload

            # Generate new hash for the updated memory
            import hashlib
            new_hash = hashlib.md5(memory.encode()).hexdigest()

            # Prepare updated payload, preserving existing metadata
            updated_payload = {
                **existing_payload,  # Preserve all existing fields
                "memory": memory,
                "hash": new_hash,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }

            # Update the point in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=memory_id,
                        vector=vector,
                        payload=updated_payload
                    )
                ]
            )

            logger.info(f"Memory {memory_id} updated successfully")
            return True

        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health and collection status."""
        try:
            # Check if Qdrant is accessible
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            return {
                "qdrant_status": "healthy",
                "collection_exists": self.collection_name in collection_names,
                "collection_name": self.collection_name,
                "total_collections": len(collection_names)
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "qdrant_status": "unhealthy",
                "collection_exists": False,
                "error": str(e)
            }

    def delete_all_memories(self, user_id: str) -> Dict[str, Any]:
        """Delete all memories for a specific user."""
        try:
            logger.info(f"Deleting all memories for user: {user_id}")

            # First, get all memory IDs for the user
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=10000,  # Large limit to get all memories
                with_payload=True
            )

            memories = search_result[0]  # Get the points
            memory_ids = [point.id for point in memories]
            memory_count = len(memory_ids)

            logger.info(f"Found {memory_count} memories to delete for user {user_id}")

            if memory_count == 0:
                return {
                    "deleted_count": 0,
                    "message": f"No memories found for user {user_id}",
                    "status": "success"
                }

            # Delete all memories for this user
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                )
            )

            logger.info(f"Successfully deleted all {memory_count} memories for user {user_id}")

            return {
                "deleted_count": memory_count,
                "message": f"Successfully deleted {memory_count} memories for user {user_id}",
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error deleting all memories for user {user_id}: {e}")
            raise
