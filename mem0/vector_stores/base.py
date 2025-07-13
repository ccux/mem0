from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

class SearchMode(Enum):
    SEMANTIC = "semantic"
    TEXT = "text"
    HYBRID = "hybrid"

class SortOrder(Enum):
    ASC = "asc"
    DESC = "desc"

class VectorStoreBase(ABC):
    @abstractmethod
    def create_col(self, name, vector_size, distance):
        """Create a new collection."""
        pass

    @abstractmethod
    def insert(self, vectors, payloads=None, ids=None):
        """Insert vectors into a collection."""
        pass

    @abstractmethod
    def search(self, query, vectors, limit=5, filters=None):
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, vector_id):
        """Delete a vector by ID."""
        pass

    @abstractmethod
    def update(self, vector_id, vector=None, payload=None):
        """Update a vector and its payload."""
        pass

    @abstractmethod
    def get(self, vector_id):
        """Retrieve a vector by ID."""
        pass

    @abstractmethod
    def list_cols(self):
        """List all collections."""
        pass

    @abstractmethod
    def delete_col(self):
        """Delete a collection."""
        pass

    @abstractmethod
    def col_info(self):
        """Get information about a collection."""
        pass

    @abstractmethod
    def list(self, filters=None, limit=None):
        """List all memories."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset by delete the collection and recreate it."""
        pass

    # Enhanced methods for advanced querying
    @abstractmethod
    def advanced_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        mode: SearchMode = SearchMode.SEMANTIC,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Advanced search with multiple modes:
        - SEMANTIC: Vector similarity search
        - TEXT: Full-text search
        - HYBRID: Combined semantic + text search
        """
        pass

    @abstractmethod
    def count_memories(self, filters: Optional[Dict] = None) -> int:
        """Efficient memory counting with optional filters"""
        pass

    @abstractmethod
    def list_with_sorting(
        self,
        filters: Optional[Dict] = None,
        sort_by: Optional[List[Tuple[str, SortOrder]]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """List memories with complex sorting and pagination"""
        pass

    @abstractmethod
    def aggregate_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a user"""
        pass

    @abstractmethod
    def bulk_operations(self, operations: List[Dict]) -> List[Dict]:
        """Execute multiple operations in a single transaction"""
        pass

    @abstractmethod
    def create_indexes(self) -> None:
        """Create optimized indexes for common query patterns"""
        pass

    @abstractmethod
    def search_with_metadata_aggregation(
        self,
        query: str,
        filters: Optional[Dict] = None,
        aggregate_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Search with metadata aggregation for analytics"""
        pass
