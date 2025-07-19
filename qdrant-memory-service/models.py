from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class MemoryCreate(BaseModel):
    messages: List[Dict[str, Any]]
    user_id: str
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MemoryUpdate(BaseModel):
    memory_id: str
    data: str

class MemoryDelete(BaseModel):
    memory_id: str

class MemoryBatchDelete(BaseModel):
    memory_ids: List[str]
    user_id: str

class MemorySearch(BaseModel):
    query: str
    user_id: str
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    limit: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = None

class MemoryResponse(BaseModel):
    id: str
    memory: str
    hash: str
    metadata: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    user_id: str
    score: Optional[float] = None  # Similarity score from vector search

class MemoryListResponse(BaseModel):
    results: List[MemoryResponse]
    status: str = "success"
    count: int

class MemorySearchResponse(BaseModel):
    results: List[MemoryResponse]
    status: str = "success"
    count: int

class BatchDeleteResponse(BaseModel):
    deleted_count: int
    failed_count: int
    deleted_ids: List[str]
    failed_ids: List[str]
    status: str = "success"

class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "qdrant-memory-service"
    version: str = "1.0.0"
    qdrant_status: str
    collection_exists: bool
    graph_status: Optional[str] = None
    graph_connected: bool = False

# Graph Memory Models
class GraphEntityResponse(BaseModel):
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    memory_id: str

class GraphRelationshipResponse(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    memory_id: str

class GraphSummaryResponse(BaseModel):
    total_entities: int
    total_relationships: int
    entity_types: Dict[str, int]
    relationship_types: Dict[str, int]

class GraphSearchRequest(BaseModel):
    query: str
    user_id: str
    entity_id: Optional[str] = None
    max_depth: Optional[int] = 2
    limit: Optional[int] = 20

class GraphSearchResponse(BaseModel):
    entities: List[GraphEntityResponse]
    relationships: List[GraphRelationshipResponse]
    related_memories: List[MemoryResponse]
    status: str = "success"

# Hybrid Search Models
class SearchMode(BaseModel):
    mode: str = "semantic"  # semantic, text, hybrid
    weights: Optional[Dict[str, float]] = None  # For hybrid search

class HybridSearchRequest(BaseModel):
    query: str
    user_id: str
    search_mode: SearchMode = SearchMode()
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    limit: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = None
    include_graph: Optional[bool] = False

# Memory Analytics Models
class MemoryCluster(BaseModel):
    cluster_id: int
    topic: str
    memory_count: int
    representative_memories: List[str]
    keywords: List[str]

class MemoryAnalyticsResponse(BaseModel):
    total_memories: int
    memory_clusters: List[MemoryCluster]
    temporal_distribution: Dict[str, int]  # Date -> count
    category_distribution: Dict[str, int]
    average_similarity_scores: Dict[str, float]
    insights: List[str]
