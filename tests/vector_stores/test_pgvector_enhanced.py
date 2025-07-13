import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import List, Dict, Any

from mem0.vector_stores.pgvector import PGVector
from mem0.vector_stores.base import SearchMode, SortOrder


class TestEnhancedPGVector:
    """Test suite for enhanced PGVector functionality"""
    
    @pytest.fixture
    def mock_connection(self):
        """Mock database connection"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        return mock_conn, mock_cursor

    @pytest.fixture
    def pgvector_instance(self, mock_connection):
        """Create PGVector instance with mocked connection"""
        mock_conn, mock_cursor = mock_connection
        
        with patch('psycopg2.connect', return_value=mock_conn):
            # Mock list_cols to return empty list to avoid create_col
            with patch.object(PGVector, 'list_cols', return_value=['memories']):
                with patch.object(PGVector, 'create_indexes'):
                    instance = PGVector(
                        dbname="test_db",
                        collection_name="memories",
                        embedding_model_dims=768,
                        user="test_user",
                        password="test_pass",
                        host="localhost",
                        port=5432,
                        diskann=False,
                        hnsw=True
                    )
                    instance.cur = mock_cursor
                    instance.conn = mock_conn
                    return instance

    def test_create_indexes(self, pgvector_instance):
        """Test index creation"""
        mock_cursor = pgvector_instance.cur
        mock_conn = pgvector_instance.conn
        
        # Reset mock to track calls
        mock_cursor.reset_mock()
        mock_conn.reset_mock()
        
        pgvector_instance.create_indexes()
        
        # Check that multiple index creation statements were executed
        assert mock_cursor.execute.call_count >= 8  # We expect at least 8 indexes
        assert mock_conn.commit.call_count >= 8
        
        # Check that user index was created
        calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        user_index_calls = [call for call in calls if 'user_idx' in call]
        assert len(user_index_calls) > 0

    def test_advanced_search_semantic_mode(self, pgvector_instance):
        """Test advanced search with semantic mode"""
        with patch.object(pgvector_instance, '_semantic_search', return_value=[]) as mock_semantic:
            result = pgvector_instance.advanced_search(
                "test query",
                filters={"user_id": "test_user"},
                mode=SearchMode.SEMANTIC,
                limit=10
            )
            
            mock_semantic.assert_called_once_with("test query", {"user_id": "test_user"}, 10, 0)
            assert result == []

    def test_advanced_search_text_mode(self, pgvector_instance):
        """Test advanced search with text mode"""
        mock_cursor = pgvector_instance.cur
        mock_cursor.fetchall.return_value = [
            ('id1', {'memory': 'test memory'}, 0.8),
            ('id2', {'memory': 'another memory'}, 0.6)
        ]
        
        result = pgvector_instance.advanced_search(
            "test query",
            filters={"user_id": "test_user"},
            mode=SearchMode.TEXT,
            limit=10
        )
        
        assert len(result) == 2
        assert result[0]['id'] == 'id1'
        assert result[0]['search_mode'] == 'text'
        assert result[0]['score'] == 0.8

    def test_advanced_search_hybrid_mode(self, pgvector_instance):
        """Test advanced search with hybrid mode"""
        with patch.object(pgvector_instance, '_semantic_search', return_value=[]):
            with patch.object(pgvector_instance, '_text_search', return_value=[
                {'id': 'id1', 'payload': {'memory': 'test'}, 'score': 0.8}
            ]):
                result = pgvector_instance.advanced_search(
                    "test query",
                    mode=SearchMode.HYBRID,
                    limit=10
                )
                
                assert len(result) == 1
                assert result[0]['search_mode'] == 'text'

    def test_count_memories(self, pgvector_instance):
        """Test memory counting functionality"""
        mock_cursor = pgvector_instance.cur
        mock_cursor.fetchone.return_value = [42]
        
        result = pgvector_instance.count_memories(filters={"user_id": "test_user"})
        
        assert result == 42
        mock_cursor.execute.assert_called_once()
        
        # Check that the query includes WHERE clause
        executed_query = mock_cursor.execute.call_args[0][0]
        assert "WHERE" in executed_query
        assert "user_id" in executed_query

    def test_list_with_sorting_default(self, pgvector_instance):
        """Test list with default sorting"""
        mock_cursor = pgvector_instance.cur
        mock_cursor.fetchall.return_value = [
            ('id1', {'memory': 'test memory 1'}),
            ('id2', {'memory': 'test memory 2'})
        ]
        
        result = pgvector_instance.list_with_sorting(
            filters={"user_id": "test_user"},
            limit=10
        )
        
        assert len(result) == 2
        assert result[0]['id'] == 'id1'
        
        # Check that ORDER BY clause includes source priority
        executed_query = mock_cursor.execute.call_args[0][0]
        assert "ORDER BY" in executed_query
        assert "auto-extraction" in executed_query

    def test_list_with_sorting_custom(self, pgvector_instance):
        """Test list with custom sorting"""
        mock_cursor = pgvector_instance.cur
        mock_cursor.fetchall.return_value = [
            ('id1', {'memory': 'test memory 1'}),
        ]
        
        result = pgvector_instance.list_with_sorting(
            sort_by=[("created_at", SortOrder.DESC), ("memory_length", SortOrder.ASC)],
            limit=5
        )
        
        assert len(result) == 1
        
        # Check that custom ORDER BY clause was used
        executed_query = mock_cursor.execute.call_args[0][0]
        assert "ORDER BY" in executed_query
        assert "created_at" in executed_query
        assert "DESC" in executed_query

    def test_aggregate_stats(self, pgvector_instance):
        """Test aggregate statistics functionality"""
        mock_cursor = pgvector_instance.cur
        mock_cursor.fetchone.return_value = [
            100,  # total_memories
            60,   # auto_extracted
            30,   # manual
            10,   # document_based
            5,    # unique_documents
            datetime(2024, 1, 1, tzinfo=timezone.utc),  # earliest_memory
            datetime(2024, 12, 31, tzinfo=timezone.utc), # latest_memory
            250.5  # avg_memory_length
        ]
        
        result = pgvector_instance.aggregate_stats("test_user")
        
        assert result['total_memories'] == 100
        assert result['auto_extracted'] == 60
        assert result['manual'] == 30
        assert result['document_based'] == 10
        assert result['unique_documents'] == 5
        assert result['earliest_memory'] == '2024-01-01T00:00:00+00:00'
        assert result['latest_memory'] == '2024-12-31T00:00:00+00:00'
        assert result['avg_memory_length'] == 250.5

    def test_bulk_operations_success(self, pgvector_instance):
        """Test successful bulk operations"""
        mock_conn = pgvector_instance.conn
        
        with patch.object(pgvector_instance, 'insert', return_value=['id1']):
            with patch.object(pgvector_instance, 'update', return_value=True):
                with patch.object(pgvector_instance, 'delete', return_value=True):
                    operations = [
                        {'type': 'insert', 'vectors': [[0.1, 0.2]], 'payloads': [{'memory': 'test'}]},
                        {'type': 'update', 'vector_id': 'id1', 'payload': {'memory': 'updated'}},
                        {'type': 'delete', 'vector_id': 'id2'}
                    ]
                    
                    result = pgvector_instance.bulk_operations(operations)
                    
                    assert len(result) == 3
                    assert result[0]['type'] == 'insert'
                    assert result[1]['type'] == 'update'
                    assert result[2]['type'] == 'delete'
                    
                    mock_conn.commit.assert_called_once()

    def test_bulk_operations_error(self, pgvector_instance):
        """Test bulk operations with error handling"""
        mock_conn = pgvector_instance.conn
        
        with patch.object(pgvector_instance, 'insert', side_effect=Exception("Insert failed")):
            operations = [
                {'type': 'insert', 'vectors': [[0.1, 0.2]], 'payloads': [{'memory': 'test'}]}
            ]
            
            result = pgvector_instance.bulk_operations(operations)
            
            assert len(result) == 1
            assert result[0]['type'] == 'error'
            assert 'Insert failed' in result[0]['message']
            
            mock_conn.rollback.assert_called_once()

    def test_search_with_metadata_aggregation(self, pgvector_instance):
        """Test search with metadata aggregation"""
        mock_search_results = [
            {'id': 'id1', 'payload': {'memory': 'test'}, 'score': 0.8}
        ]
        
        with patch.object(pgvector_instance, 'advanced_search', return_value=mock_search_results):
            with patch.object(pgvector_instance, '_aggregate_field', return_value={'manual': 5, 'auto': 3}):
                result = pgvector_instance.search_with_metadata_aggregation(
                    "test query",
                    filters={"user_id": "test_user"},
                    aggregate_fields=["source"]
                )
                
                assert result['results'] == mock_search_results
                assert result['total_count'] == 1
                assert result['aggregations']['source'] == {'manual': 5, 'auto': 3}

    def test_build_filter_conditions_user_id(self, pgvector_instance):
        """Test filter conditions building for user_id"""
        filters = {"user_id": "test_user"}
        where_clause, params = pgvector_instance._build_filter_conditions(filters)
        
        assert "WHERE" in where_clause
        assert "user_id" in where_clause
        assert params == ["test_user"]

    def test_build_filter_conditions_multiple(self, pgvector_instance):
        """Test filter conditions building for multiple filters"""
        filters = {
            "user_id": "test_user",
            "source": "manual",
            "document_id": "doc123"
        }
        where_clause, params = pgvector_instance._build_filter_conditions(filters)
        
        assert "WHERE" in where_clause
        assert "user_id" in where_clause
        assert "source" in where_clause
        assert "document_id" in where_clause
        assert len(params) == 3

    def test_build_filter_conditions_file_name(self, pgvector_instance):
        """Test filter conditions building for file_name (handles both variants)"""
        filters = {"file_name": "test.pdf"}
        where_clause, params = pgvector_instance._build_filter_conditions(filters)
        
        assert "file_name" in where_clause
        assert "fileName" in where_clause  # Check both variants
        assert len(params) == 2  # Should have both variants

    def test_build_filter_conditions_date_range(self, pgvector_instance):
        """Test filter conditions building for date ranges"""
        filters = {
            "created_after": "2024-01-01",
            "created_before": "2024-12-31"
        }
        where_clause, params = pgvector_instance._build_filter_conditions(filters)
        
        assert "created_at" in where_clause
        assert ">" in where_clause
        assert "<" in where_clause
        assert len(params) == 2

    def test_build_filter_conditions_text_contains(self, pgvector_instance):
        """Test filter conditions building for text contains"""
        filters = {"text_contains": "search term"}
        where_clause, params = pgvector_instance._build_filter_conditions(filters)
        
        assert "LIKE" in where_clause
        assert "LOWER" in where_clause
        assert params == ["%search term%"]

    def test_build_order_clause_default(self, pgvector_instance):
        """Test default order clause building"""
        order_clause = pgvector_instance._build_order_clause(None)
        
        assert "ORDER BY" in order_clause
        assert "auto-extraction" in order_clause
        assert "COALESCE" in order_clause

    def test_build_order_clause_custom(self, pgvector_instance):
        """Test custom order clause building"""
        sort_by = [("created_at", SortOrder.DESC), ("memory_length", SortOrder.ASC)]
        order_clause = pgvector_instance._build_order_clause(sort_by)
        
        assert "ORDER BY" in order_clause
        assert "created_at" in order_clause
        assert "DESC" in order_clause
        assert "LENGTH" in order_clause
        assert "ASC" in order_clause

    def test_aggregate_field_source(self, pgvector_instance):
        """Test field aggregation for source"""
        mock_cursor = pgvector_instance.cur
        mock_cursor.fetchall.return_value = [
            ('manual', 10),
            ('auto-extraction', 5),
            ('document', 3)
        ]
        
        result = pgvector_instance._aggregate_field("source", {"user_id": "test_user"})
        
        assert result == {'manual': 10, 'auto-extraction': 5, 'document': 3}
        
        # Check that GROUP BY was used
        executed_query = mock_cursor.execute.call_args[0][0]
        assert "GROUP BY" in executed_query
        assert "source" in executed_query

    def test_aggregate_field_document_id(self, pgvector_instance):
        """Test field aggregation for document_id"""
        mock_cursor = pgvector_instance.cur
        mock_cursor.fetchall.return_value = [
            ('doc1', 15),
            ('doc2', 8),
            (None, 2)  # Test null handling
        ]
        
        result = pgvector_instance._aggregate_field("document_id", {})
        
        assert result == {'doc1': 15, 'doc2': 8, 'null': 2}

    def test_aggregate_field_generic(self, pgvector_instance):
        """Test field aggregation for generic field"""
        mock_cursor = pgvector_instance.cur
        mock_cursor.fetchall.return_value = [
            ('value1', 20),
            ('value2', 12)
        ]
        
        result = pgvector_instance._aggregate_field("custom_field", {})
        
        assert result == {'value1': 20, 'value2': 12}
        
        # Check that custom field was used
        executed_query = mock_cursor.execute.call_args[0][0]
        assert "custom_field" in executed_query

    def test_text_search_with_filters(self, pgvector_instance):
        """Test text search with filters applied"""
        mock_cursor = pgvector_instance.cur
        mock_cursor.fetchall.return_value = [
            ('id1', {'memory': 'test memory'}, 0.9)
        ]
        
        result = pgvector_instance._text_search(
            "test query",
            filters={"user_id": "test_user", "source": "manual"},
            limit=5,
            offset=0
        )
        
        assert len(result) == 1
        assert result[0]['id'] == 'id1'
        assert result[0]['score'] == 0.9
        assert result[0]['search_mode'] == 'text'
        
        # Check that ts_rank was used for full-text search
        executed_query = mock_cursor.execute.call_args[0][0]
        assert "ts_rank" in executed_query
        assert "plainto_tsquery" in executed_query

    def test_hybrid_search_deduplication(self, pgvector_instance):
        """Test that hybrid search properly deduplicates results"""
        semantic_results = [
            {'id': 'id1', 'payload': {'memory': 'test'}, 'score': 0.9},
            {'id': 'id2', 'payload': {'memory': 'test2'}, 'score': 0.8}
        ]
        
        text_results = [
            {'id': 'id1', 'payload': {'memory': 'test'}, 'score': 0.7},  # Duplicate
            {'id': 'id3', 'payload': {'memory': 'test3'}, 'score': 0.6}
        ]
        
        with patch.object(pgvector_instance, '_semantic_search', return_value=semantic_results):
            with patch.object(pgvector_instance, '_text_search', return_value=text_results):
                result = pgvector_instance._hybrid_search("test query", {}, 10, 0)
                
                # Should have 3 unique results (id1, id2, id3)
                assert len(result) == 3
                
                # Check that semantic results are preferred (id1 should have semantic mode)
                id1_result = next(r for r in result if r['id'] == 'id1')
                assert id1_result['search_mode'] == 'semantic'
                
                # Check that text-only results are included
                id3_result = next(r for r in result if r['id'] == 'id3')
                assert id3_result['search_mode'] == 'text'

    def test_advanced_search_invalid_mode(self, pgvector_instance):
        """Test advanced search with invalid mode raises error"""
        with pytest.raises(ValueError, match="Unsupported search mode"):
            pgvector_instance.advanced_search("test", mode="invalid_mode")

    def test_bulk_operations_unknown_type(self, pgvector_instance):
        """Test bulk operations with unknown operation type"""
        operations = [
            {'type': 'unknown_operation', 'data': 'test'}
        ]
        
        result = pgvector_instance.bulk_operations(operations)
        
        assert len(result) == 1
        assert result[0]['type'] == 'error'
        assert 'Unknown operation type' in result[0]['message']


if __name__ == "__main__":
    pytest.main([__file__]) 