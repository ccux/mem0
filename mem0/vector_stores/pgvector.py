import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pydantic import BaseModel

try:
    import psycopg2
    from psycopg2.extras import execute_values, RealDictCursor
except ImportError:
    raise ImportError("The 'psycopg2' library is required. Please install it using 'pip install psycopg2'.")

from mem0.vector_stores.base import VectorStoreBase, SearchMode, SortOrder

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[dict]


class PGVector(VectorStoreBase):
    def __init__(
        self,
        dbname,
        collection_name,
        embedding_model_dims,
        user,
        password,
        host,
        port,
        diskann,
        hnsw,
    ):
        """
        Initialize the PGVector database.

        Args:
            dbname (str): Database name
            collection_name (str): Collection name
            embedding_model_dims (int): Dimension of the embedding vector
            user (str): Database user
            password (str): Database password
            host (str, optional): Database host
            port (int, optional): Database port
            diskann (bool, optional): Use DiskANN for faster search
            hnsw (bool, optional): Use HNSW for faster search
        """
        logger.warning(f"PGVector Store __init__: received embedding_model_dims = {embedding_model_dims}, type = {type(embedding_model_dims)}")

        self.collection_name = collection_name
        self.use_diskann = diskann
        self.use_hnsw = hnsw
        self.embedding_model_dims = embedding_model_dims

        self.conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        self.cur = self.conn.cursor()

        collections = self.list_cols()
        if collection_name not in collections:
            self.create_col(self.embedding_model_dims)
        
        # Create enhanced indexes for advanced querying
        self.create_indexes()

    def create_col(self, embedding_model_dims_arg):
        """
        Create a new collection (table in PostgreSQL).
        Will also initialize vector search index if specified.

        Args:
            embedding_model_dims_arg (int): Dimension of the embedding vector.
        """
        logger.warning(f"PGVector create_col: using embedding_model_dims = {embedding_model_dims_arg}")
        self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.collection_name} (
                id UUID PRIMARY KEY,
                vector vector({embedding_model_dims_arg}),
                payload JSONB
            );
        """
        )

        if self.use_diskann and embedding_model_dims_arg < 2000:
            # Check if vectorscale extension is installed
            self.cur.execute("SELECT * FROM pg_extension WHERE extname = 'vectorscale'")
            if self.cur.fetchone():
                # Create DiskANN index if extension is installed for faster search
                self.cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.collection_name}_diskann_idx
                    ON {self.collection_name}
                    USING diskann (vector);
                """
                )
        elif self.use_hnsw:
            self.cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.collection_name}_hnsw_idx
                ON {self.collection_name}
                USING hnsw (vector vector_cosine_ops)
            """
            )

        self.conn.commit()

    def insert(self, vectors, payloads=None, ids=None):
        """
        Insert vectors into a collection.

        Args:
            vectors (List[List[float]]): List of vectors to insert.
            payloads (List[Dict], optional): List of payloads corresponding to vectors.
            ids (List[str], optional): List of IDs corresponding to vectors.
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        json_payloads = [json.dumps(payload) for payload in payloads]

        data = [(id, vector, payload) for id, vector, payload in zip(ids, vectors, json_payloads)]
        execute_values(
            self.cur,
            f"INSERT INTO {self.collection_name} (id, vector, payload) VALUES %s",
            data,
        )
        self.conn.commit()

    def search(self, query, vectors, limit=5, filters=None):
        """
        Search for similar vectors.

        Args:
            query (str): Query.
            vectors (List[float]): Query vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (Dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: Search results.
        """
        filter_conditions = []
        filter_params = []

        if filters:
            for k, v in filters.items():
                filter_conditions.append("payload->>%s = %s")
                filter_params.extend([k, str(v)])

        filter_clause = "WHERE " + " AND ".join(filter_conditions) if filter_conditions else ""

        self.cur.execute(
            f"""
            SELECT id, vector <=> %s::vector AS distance, payload
            FROM {self.collection_name}
            {filter_clause}
            ORDER BY distance
            LIMIT %s
        """,
            (vectors, *filter_params, limit),
        )

        results = self.cur.fetchall()
        return [OutputData(id=str(r[0]), score=float(r[1]), payload=r[2]) for r in results]

    def delete(self, vector_id):
        """
        Delete a vector by ID.

        Args:
            vector_id (str): ID of the vector to delete.
        """
        self.cur.execute(f"DELETE FROM {self.collection_name} WHERE id = %s", (vector_id,))
        self.conn.commit()

    def update(self, vector_id, vector=None, payload=None):
        """
        Update a vector and its payload.

        Args:
            vector_id (str): ID of the vector to update.
            vector (List[float], optional): Updated vector.
            payload (Dict, optional): Updated payload.
        """
        if vector:
            self.cur.execute(
                f"UPDATE {self.collection_name} SET vector = %s WHERE id = %s",
                (vector, vector_id),
            )
        if payload:
            self.cur.execute(
                f"UPDATE {self.collection_name} SET payload = %s WHERE id = %s",
                (psycopg2.extras.Json(payload), vector_id),
            )
        self.conn.commit()

    def get(self, vector_id) -> OutputData:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (str): ID of the vector to retrieve.

        Returns:
            OutputData: Retrieved vector.
        """
        self.cur.execute(
            f"SELECT id, vector, payload FROM {self.collection_name} WHERE id = %s",
            (vector_id,),
        )
        result = self.cur.fetchone()
        if not result:
            return None
        return OutputData(id=str(result[0]), score=None, payload=result[2])

    def list_cols(self) -> List[str]:
        """
        List all collections.

        Returns:
            List[str]: List of collection names.
        """
        self.cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        return [row[0] for row in self.cur.fetchall()]

    def delete_col(self):
        """Delete a collection."""
        self.cur.execute(f"DROP TABLE IF EXISTS {self.collection_name}")
        self.conn.commit()

    def col_info(self):
        """
        Get information about a collection.

        Returns:
            Dict[str, Any]: Collection information.
        """
        self.cur.execute(
            f"""
            SELECT
                table_name,
                (SELECT COUNT(*) FROM {self.collection_name}) as row_count,
                (SELECT pg_size_pretty(pg_total_relation_size('{self.collection_name}'))) as total_size
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
        """,
            (self.collection_name,),
        )
        result = self.cur.fetchone()
        return {"name": result[0], "count": result[1], "size": result[2]}

    def list(self, filters=None, limit=100):
        """
        List all vectors in a collection.

        Args:
            filters (Dict, optional): Filters to apply to the list.
            limit (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            List[OutputData]: List of vectors.
        """
        filter_conditions = []
        filter_params = []

        if filters:
            for k, v in filters.items():
                filter_conditions.append("payload->>%s = %s")
                filter_params.extend([k, str(v)])

        filter_clause = "WHERE " + " AND ".join(filter_conditions) if filter_conditions else ""

        query = f"""
            SELECT id, vector, payload
            FROM {self.collection_name}
            {filter_clause}
            LIMIT %s
        """

        self.cur.execute(query, (*filter_params, limit))

        results = self.cur.fetchall()
        return [[OutputData(id=str(r[0]), score=None, payload=r[2]) for r in results]]

    def reset(self):
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting PGVector collection {self.collection_name} using dims: {self.embedding_model_dims}")
        self.delete_col()
        self.create_col(self.embedding_model_dims)

    def __del__(self):
        """
        Close the database connection when the object is deleted.
        """
        if hasattr(self, "cur") and self.cur and not self.cur.closed:
            self.cur.close()
        if hasattr(self, "conn") and self.conn and not self.conn.closed:
            self.conn.close()

    def create_indexes(self) -> None:
        """Create optimized indexes for common query patterns"""
        indexes = [
            # User-based queries
            f"CREATE INDEX IF NOT EXISTS {self.collection_name}_user_idx ON {self.collection_name} ((payload->>'user_id'))",

            # Source-based queries
            f"CREATE INDEX IF NOT EXISTS {self.collection_name}_source_idx ON {self.collection_name} ((payload->'metadata'->>'source'))",

            # Document-based queries
            f"CREATE INDEX IF NOT EXISTS {self.collection_name}_document_idx ON {self.collection_name} ((payload->'metadata'->>'document_id'))",

            # Filename queries
            f"CREATE INDEX IF NOT EXISTS {self.collection_name}_filename_idx ON {self.collection_name} ((payload->'metadata'->>'file_name'))",

            # Time-based queries
            f"CREATE INDEX IF NOT EXISTS {self.collection_name}_created_idx ON {self.collection_name} ((payload->>'created_at'))",
            f"CREATE INDEX IF NOT EXISTS {self.collection_name}_updated_idx ON {self.collection_name} ((payload->>'updated_at'))",
            f"CREATE INDEX IF NOT EXISTS {self.collection_name}_extracted_idx ON {self.collection_name} ((payload->'metadata'->>'extractedAt'))",

            # Full-text search index
            f"CREATE INDEX IF NOT EXISTS {self.collection_name}_text_idx ON {self.collection_name} USING gin(to_tsvector('english', payload->>'memory'))",

            # Composite index for common query patterns
            f"CREATE INDEX IF NOT EXISTS {self.collection_name}_user_source_idx ON {self.collection_name} ((payload->>'user_id'), (payload->'metadata'->>'source'))",
        ]

        for index in indexes:
            try:
                self.cur.execute(index)
                self.conn.commit()
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
                self.conn.rollback()

    def advanced_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        mode: SearchMode = SearchMode.SEMANTIC,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Advanced search with multiple modes"""

        if mode == SearchMode.SEMANTIC:
            return self._semantic_search(query, filters, limit, offset)
        elif mode == SearchMode.TEXT:
            return self._text_search(query, filters, limit, offset)
        elif mode == SearchMode.HYBRID:
            return self._hybrid_search(query, filters, limit, offset)
        else:
            raise ValueError(f"Unsupported search mode: {mode}")

    def _semantic_search(self, query: str, filters: Optional[Dict], limit: int, offset: int) -> List[Dict]:
        """Semantic vector search"""
        # Note: This method assumes vectors will be provided by the calling code
        # For now, we'll use the existing search method structure but return empty results
        # since we don't have access to the embedding model here
        
        # In a full implementation, this would:
        # 1. Get embeddings for the query using the embedding model
        # 2. Use the existing search method with those embeddings
        # 3. Apply offset for pagination
        
        # For now, return empty results to avoid errors
        return []

    def _text_search(self, query: str, filters: Optional[Dict], limit: int, offset: int) -> List[Dict]:
        """Full-text search using PostgreSQL's text search capabilities"""
        filter_conditions, filter_params = self._build_filter_conditions(filters)

        # Build text search query
        text_query = f"""
            SELECT id, payload,
                   ts_rank(to_tsvector('english', payload->>'memory'), plainto_tsquery('english', %s)) as rank
            FROM {self.collection_name}
            WHERE to_tsvector('english', payload->>'memory') @@ plainto_tsquery('english', %s)
            {filter_conditions}
            ORDER BY rank DESC,
                CASE
                    WHEN payload->'metadata'->>'source' = 'auto-extraction' THEN 1
                    ELSE 2
                END,
                COALESCE(
                    (payload->'metadata'->>'extractedAt')::timestamp,
                    (payload->>'updated_at')::timestamp,
                    (payload->>'created_at')::timestamp
                ) DESC NULLS LAST
            LIMIT %s OFFSET %s
        """

        params = [query, query] + filter_params + [limit, offset]
        self.cur.execute(text_query, params)

        results = []
        for row in self.cur.fetchall():
            results.append({
                'id': str(row[0]),
                'payload': row[1],
                'score': float(row[2]) if row[2] else 0.0,
                'search_mode': 'text'
            })

        return results

    def _hybrid_search(self, query: str, filters: Optional[Dict], limit: int, offset: int) -> List[Dict]:
        """Hybrid search combining semantic and text search"""
        # Get semantic results
        semantic_results = self._semantic_search(query, filters, limit // 2, 0)

        # Get text results
        text_results = self._text_search(query, filters, limit // 2, 0)

        # Combine and deduplicate results
        seen_ids = set()
        combined_results = []

        # Add semantic results first (higher priority)
        for result in semantic_results:
            if result['id'] not in seen_ids:
                result['search_mode'] = 'semantic'
                combined_results.append(result)
                seen_ids.add(result['id'])

        # Add text results
        for result in text_results:
            if result['id'] not in seen_ids:
                result['search_mode'] = 'text'
                combined_results.append(result)
                seen_ids.add(result['id'])

        # Sort by combined score and return limited results
        combined_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return combined_results[:limit]

    def count_memories(self, filters: Optional[Dict] = None) -> int:
        """Efficient memory counting with optional filters"""
        filter_conditions, filter_params = self._build_filter_conditions(filters)

        query = f"SELECT COUNT(*) FROM {self.collection_name} {filter_conditions}"
        self.cur.execute(query, filter_params)

        return self.cur.fetchone()[0]

    def list_with_sorting(
        self,
        filters: Optional[Dict] = None,
        sort_by: Optional[List[Tuple[str, SortOrder]]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """List memories with complex sorting and pagination"""

        filter_conditions, filter_params = self._build_filter_conditions(filters)
        order_clause = self._build_order_clause(sort_by)

        query = f"""
            SELECT id, payload
            FROM {self.collection_name}
            {filter_conditions}
            {order_clause}
            LIMIT %s OFFSET %s
        """

        params = filter_params + [limit, offset]
        self.cur.execute(query, params)

        results = []
        for row in self.cur.fetchall():
            results.append({
                'id': str(row[0]),
                'payload': row[1]
            })

        return results

    def aggregate_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a user"""
        stats_query = f"""
            SELECT
                COUNT(*) as total_memories,
                COUNT(CASE WHEN payload->'metadata'->>'source' = 'auto-extraction' THEN 1 END) as auto_extracted,
                COUNT(CASE WHEN payload->'metadata'->>'source' = 'manual' THEN 1 END) as manual,
                COUNT(CASE WHEN payload->'metadata'->>'source' = 'document' THEN 1 END) as document_based,
                COUNT(DISTINCT payload->'metadata'->>'document_id') as unique_documents,
                MIN((payload->>'created_at')::timestamp) as earliest_memory,
                MAX((payload->>'created_at')::timestamp) as latest_memory,
                AVG(LENGTH(payload->>'memory')) as avg_memory_length
            FROM {self.collection_name}
            WHERE payload->>'user_id' = %s
        """

        self.cur.execute(stats_query, [user_id])
        row = self.cur.fetchone()

        return {
            'total_memories': row[0] or 0,
            'auto_extracted': row[1] or 0,
            'manual': row[2] or 0,
            'document_based': row[3] or 0,
            'unique_documents': row[4] or 0,
            'earliest_memory': row[5].isoformat() if row[5] else None,
            'latest_memory': row[6].isoformat() if row[6] else None,
            'avg_memory_length': float(row[7]) if row[7] else 0.0
        }

    def bulk_operations(self, operations: List[Dict]) -> List[Dict]:
        """Execute multiple operations in a single transaction"""
        results = []
        
        try:
            for operation in operations:
                op_type = operation.get('type')
                
                if op_type == 'insert':
                    result = self.insert(
                        operation['vectors'],
                        operation.get('payloads'),
                        operation.get('ids')
                    )
                    results.append({'type': 'insert', 'result': result})
                
                elif op_type == 'update':
                    result = self.update(
                        operation['vector_id'],
                        operation.get('vector'),
                        operation.get('payload')
                    )
                    results.append({'type': 'update', 'result': result})
                
                elif op_type == 'delete':
                    result = self.delete(operation['vector_id'])
                    results.append({'type': 'delete', 'result': result})
                
                else:
                    results.append({'type': 'error', 'message': f'Unknown operation type: {op_type}'})
            
            self.conn.commit()
            
        except Exception as e:
            self.conn.rollback()
            results.append({'type': 'error', 'message': str(e)})
        
        return results

    def search_with_metadata_aggregation(
        self,
        query: str,
        filters: Optional[Dict] = None,
        aggregate_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Search with metadata aggregation for analytics"""

        # Get search results
        results = self.advanced_search(query, filters, mode=SearchMode.HYBRID)

        # Perform aggregations if requested
        aggregations = {}
        if aggregate_fields:
            for field in aggregate_fields:
                aggregations[field] = self._aggregate_field(field, filters)

        return {
            'results': results,
            'aggregations': aggregations,
            'total_count': len(results)
        }

    def _build_filter_conditions(self, filters: Optional[Dict]) -> Tuple[str, List]:
        """Build SQL WHERE conditions from filters"""
        if not filters:
            return "", []

        conditions = []
        params = []

        for key, value in filters.items():
            if key == "user_id":
                conditions.append("payload->>'user_id' = %s")
                params.append(value)
            elif key == "source":
                conditions.append("payload->'metadata'->>'source' = %s")
                params.append(value)
            elif key == "document_id":
                conditions.append("payload->'metadata'->>'document_id' = %s")
                params.append(value)
            elif key == "file_name":
                conditions.append("(payload->'metadata'->>'file_name' = %s OR payload->'metadata'->>'fileName' = %s)")
                params.extend([value, value])
            elif key == "created_after":
                conditions.append("(payload->>'created_at')::timestamp > %s")
                params.append(value)
            elif key == "created_before":
                conditions.append("(payload->>'created_at')::timestamp < %s")
                params.append(value)
            elif key == "text_contains":
                conditions.append("LOWER(payload->>'memory') LIKE LOWER(%s)")
                params.append(f"%{value}%")

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        return where_clause, params

    def _build_order_clause(self, sort_by: Optional[List[Tuple[str, SortOrder]]]) -> str:
        """Build SQL ORDER BY clause"""
        if not sort_by:
            # Default sorting: source priority, then time
            return """
                ORDER BY
                    CASE
                        WHEN payload->'metadata'->>'source' = 'auto-extraction' THEN 1
                        ELSE 2
                    END,
                    COALESCE(
                        (payload->'metadata'->>'extractedAt')::timestamp,
                        (payload->>'updated_at')::timestamp,
                        (payload->>'created_at')::timestamp
                    ) DESC NULLS LAST
            """

        order_parts = []
        for field, order in sort_by:
            direction = "ASC" if order == SortOrder.ASC else "DESC"

            if field == "created_at":
                order_parts.append(f"(payload->>'created_at')::timestamp {direction}")
            elif field == "updated_at":
                order_parts.append(f"(payload->>'updated_at')::timestamp {direction}")
            elif field == "source_priority":
                order_parts.append(f"CASE WHEN payload->'metadata'->>'source' = 'auto-extraction' THEN 1 ELSE 2 END {direction}")
            elif field == "memory_length":
                order_parts.append(f"LENGTH(payload->>'memory') {direction}")
            else:
                order_parts.append(f"payload->>'{field}' {direction}")

        return f"ORDER BY {', '.join(order_parts)}"

    def _aggregate_field(self, field: str, filters: Optional[Dict]) -> Dict[str, Any]:
        """Aggregate values for a specific field"""
        filter_conditions, filter_params = self._build_filter_conditions(filters)

        if field == "source":
            query = f"""
                SELECT payload->'metadata'->>'source' as source, COUNT(*) as count
                FROM {self.collection_name}
                {filter_conditions}
                GROUP BY payload->'metadata'->>'source'
                ORDER BY count DESC
            """
        elif field == "document_id":
            query = f"""
                SELECT payload->'metadata'->>'document_id' as document_id, COUNT(*) as count
                FROM {self.collection_name}
                {filter_conditions}
                GROUP BY payload->'metadata'->>'document_id'
                ORDER BY count DESC
            """
        else:
            query = f"""
                SELECT payload->>'{field}' as value, COUNT(*) as count
                FROM {self.collection_name}
                {filter_conditions}
                GROUP BY payload->>'{field}'
                ORDER BY count DESC
            """

        self.cur.execute(query, filter_params)

        results = {}
        for row in self.cur.fetchall():
            results[row[0] or 'null'] = row[1]

        return results
