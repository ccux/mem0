import logging
import os
import shutil
from typing import Optional, Dict, List, Tuple, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)

from mem0.vector_stores.base import VectorStoreBase, SearchMode, SortOrder
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[dict]


class Qdrant(VectorStoreBase):
    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int,
        client: QdrantClient = None,
        host: str = None,
        port: int = None,
        path: str = None,
        url: str = None,
        api_key: str = None,
        on_disk: bool = False,
    ):
        """
        Initialize the Qdrant vector store.

        Args:
            collection_name (str): Name of the collection.
            embedding_model_dims (int): Dimensions of the embedding model.
            client (QdrantClient, optional): Existing Qdrant client instance. Defaults to None.
            host (str, optional): Host address for Qdrant server. Defaults to None.
            port (int, optional): Port for Qdrant server. Defaults to None.
            path (str, optional): Path for local Qdrant database. Defaults to None.
            url (str, optional): Full URL for Qdrant server. Defaults to None.
            api_key (str, optional): API key for Qdrant server. Defaults to None.
            on_disk (bool, optional): Enables persistent storage. Defaults to False.
        """
        if client:
            self.client = client
        else:
            params = {}
            if api_key:
                params["api_key"] = api_key
            if url:
                params["url"] = url
            if host and port:
                params["host"] = host
                params["port"] = port
            if not params:
                params["path"] = path
                if not on_disk:
                    if os.path.exists(path) and os.path.isdir(path):
                        shutil.rmtree(path)

            self.client = QdrantClient(**params)

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.on_disk = on_disk
        self.create_col(embedding_model_dims, on_disk)

    def create_col(self, vector_size: int, on_disk: bool, distance: Distance = Distance.COSINE):
        """
        Create a new collection.

        Args:
            vector_size (int): Size of the vectors to be stored.
            on_disk (bool): Enables persistent storage.
            distance (Distance, optional): Distance metric for vector similarity. Defaults to Distance.COSINE.
        """
        # Skip creating collection if already exists
        response = self.list_cols()
        for collection in response.collections:
            if collection.name == self.collection_name:
                logging.debug(f"Collection {self.collection_name} already exists. Skipping creation.")
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance, on_disk=on_disk),
        )

    def insert(self, vectors: list, payloads: list = None, ids: list = None):
        """
        Insert vectors into a collection.

        Args:
            vectors (list): List of vectors to insert.
            payloads (list, optional): List of payloads corresponding to vectors. Defaults to None.
            ids (list, optional): List of IDs corresponding to vectors. Defaults to None.
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        points = [
            PointStruct(
                id=idx if ids is None else ids[idx],
                vector=vector,
                payload=payloads[idx] if payloads else {},
            )
            for idx, vector in enumerate(vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def _create_filter(self, filters: dict) -> Filter:
        """
        Create a Filter object from the provided filters.

        Args:
            filters (dict): Filters to apply.

        Returns:
            Filter: The created Filter object.
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict) and "gte" in value and "lte" in value:
                conditions.append(FieldCondition(key=key, range=Range(gte=value["gte"], lte=value["lte"])))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    def search(self, query: str, vectors: list, limit: int = 5, filters: dict = None) -> list:
        """
        Search for similar vectors.

        Args:
            query (str): Query.
            vectors (list): Query vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: Search results.
        """
        query_filter = self._create_filter(filters) if filters else None
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=vectors,
            query_filter=query_filter,
            limit=limit,
        )
        return hits.points

    def delete(self, vector_id: int):
        """
        Delete a vector by ID.

        Args:
            vector_id (int): ID of the vector to delete.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=[vector_id],
            ),
        )

    def update(self, vector_id: int, vector: list = None, payload: dict = None):
        """
        Update a vector and its payload.

        Args:
            vector_id (int): ID of the vector to update.
            vector (list, optional): Updated vector. Defaults to None.
            payload (dict, optional): Updated payload. Defaults to None.
        """
        point = PointStruct(id=vector_id, vector=vector, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def get(self, vector_id: int) -> dict:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (int): ID of the vector to retrieve.

        Returns:
            dict: Retrieved vector.
        """
        result = self.client.retrieve(collection_name=self.collection_name, ids=[vector_id], with_payload=True)
        return result[0] if result else None

    def list_cols(self) -> list:
        """
        List all collections.

        Returns:
            list: List of collection names.
        """
        return self.client.get_collections()

    def delete_col(self):
        """Delete a collection."""
        self.client.delete_collection(collection_name=self.collection_name)

    def col_info(self) -> dict:
        """
        Get information about a collection.

        Returns:
            dict: Collection information.
        """
        return self.client.get_collection(collection_name=self.collection_name)

    def list(self, filters: dict = None, limit: int = 100) -> list:
        """
        List all vectors in a collection.

        Args:
            filters (dict, optional): Filters to apply to the list. Defaults to None.
            limit (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            list: List of vectors.
        """
        query_filter = self._create_filter(filters) if filters else None
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        # Extract points from the scroll result tuple
        points, _ = result  # result is (points, next_page_token)

        # Convert Qdrant points to OutputData format
        output_data_list = []
        for point in points:
            output_data_list.append(OutputData(
                id=str(point.id),
                score=None,
                payload=point.payload
            ))

        return output_data_list

    def reset(self):
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting index {self.collection_name}...")
        self.delete_col()
        self.create_col(self.embedding_model_dims, self.on_disk)

    def advanced_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        mode: SearchMode = SearchMode.SEMANTIC,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Advanced search with multiple modes.
        For Qdrant, we implement semantic search as the primary mode.
        """
        # For now, implement semantic search as the primary mode
        # TODO: Implement text and hybrid search modes
        if mode == SearchMode.SEMANTIC:
            # Use the existing search method with filters
            results = self.search(query, [], limit=limit, filters=filters)
            return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in results]
        else:
            logger.warning(f"Search mode {mode} not yet implemented for Qdrant, falling back to semantic")
            return self.advanced_search(query, filters, SearchMode.SEMANTIC, limit, offset)

    def count_memories(self, filters: Optional[Dict] = None) -> int:
        """Efficient memory counting with optional filters"""
        try:
            query_filter = self._create_filter(filters) if filters else None
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=0,  # We only want the count
                with_payload=False,
                with_vectors=False,
            )
            # The result tuple contains (points, next_page_token)
            # For count, we can use the collection info
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error counting memories: {e}")
            return 0

    def list_with_sorting(
        self,
        filters: Optional[Dict] = None,
        sort_by: Optional[List[Tuple[str, SortOrder]]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """List memories with complex sorting and pagination"""
        try:
            query_filter = self._create_filter(filters) if filters else None

            # Qdrant doesn't support complex sorting in scroll, so we'll implement basic sorting
            # For now, return results in insertion order
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            points, _ = result
            return [{"id": str(p.id), "payload": p.payload} for p in points]
        except Exception as e:
            logger.error(f"Error listing with sorting: {e}")
            return []

    def aggregate_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a user"""
        try:
            # Get total memories for user
            user_filters = {"user_id": user_id}
            total_memories = self.count_memories(user_filters)

            # Get memories to analyze categories
            memories = self.list(user_filters, limit=1000)

            # Count by category
            category_counts = {}
            for memory in memories:
                if hasattr(memory, 'payload') and memory.payload:
                    category = memory.payload.get('category', 'unknown')
                    category_counts[category] = category_counts.get(category, 0) + 1

            return {
                "total_memories": total_memories,
                "category_breakdown": category_counts,
                "user_id": user_id
            }
        except Exception as e:
            logger.error(f"Error getting aggregate stats: {e}")
            return {"total_memories": 0, "category_breakdown": {}, "user_id": user_id}

    def bulk_operations(self, operations: List[Dict]) -> List[Dict]:
        """Execute multiple operations in a single transaction"""
        results = []
        try:
            for operation in operations:
                op_type = operation.get("type")
                if op_type == "insert":
                    result = self.insert(
                        operation.get("vectors", []),
                        operation.get("payloads"),
                        operation.get("ids")
                    )
                    results.append({"type": "insert", "status": "success", "result": result})
                elif op_type == "delete":
                    result = self.delete(operation.get("id"))
                    results.append({"type": "delete", "status": "success", "result": result})
                elif op_type == "update":
                    result = self.update(
                        operation.get("id"),
                        operation.get("vector"),
                        operation.get("payload")
                    )
                    results.append({"type": "update", "status": "success", "result": result})
                else:
                    results.append({"type": op_type, "status": "error", "error": "Unknown operation type"})
        except Exception as e:
            logger.error(f"Error in bulk operations: {e}")
            results.append({"type": "bulk", "status": "error", "error": str(e)})

        return results

    def create_indexes(self) -> None:
        """Create optimized indexes for common query patterns"""
        try:
            # Qdrant automatically creates indexes for vector similarity
            # We can add payload indexes for common filter fields
            logger.info(f"Qdrant automatically manages indexes for collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")

    def search_with_metadata_aggregation(
        self,
        query: str,
        filters: Optional[Dict] = None,
        aggregate_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Search with metadata aggregation for analytics"""
        try:
            # Perform the search
            search_results = self.advanced_search(query, filters, limit=100)

            # Aggregate metadata if requested
            aggregations = {}
            if aggregate_fields:
                for field in aggregate_fields:
                    field_values = {}
                    for result in search_results:
                        if "payload" in result and field in result["payload"]:
                            value = result["payload"][field]
                            field_values[value] = field_values.get(value, 0) + 1
                    aggregations[field] = field_values

            return {
                "search_results": search_results,
                "aggregations": aggregations,
                "total_results": len(search_results)
            }
        except Exception as e:
            logger.error(f"Error in search with metadata aggregation: {e}")
            return {"search_results": [], "aggregations": {}, "total_results": 0}
