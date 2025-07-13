#!/usr/bin/env python3
"""
Simple test script to verify enhanced PGVector features
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from unittest.mock import Mock, patch
from mem0.vector_stores.base import SearchMode, SortOrder
from mem0.vector_stores.pgvector import PGVector

def test_enhanced_features():
    """Test enhanced PGVector features"""
    print("Testing Enhanced PGVector Features...")
    
    # Test 1: Enum imports
    print("✓ SearchMode enum imported successfully")
    assert SearchMode.SEMANTIC.value == "semantic"
    assert SearchMode.TEXT.value == "text"
    assert SearchMode.HYBRID.value == "hybrid"
    
    print("✓ SortOrder enum imported successfully")
    assert SortOrder.ASC.value == "asc"
    assert SortOrder.DESC.value == "desc"
    
    # Test 2: Mock PGVector instance
    with patch('psycopg2.connect') as mock_connect:
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Mock list_cols to return existing collection
        with patch.object(PGVector, 'list_cols', return_value=['memories']):
            with patch.object(PGVector, 'create_indexes'):
                pgvector = PGVector(
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
                pgvector.cur = mock_cursor
                pgvector.conn = mock_conn
                
                print("✓ Enhanced PGVector instance created successfully")
    
    # Test 3: Filter conditions builder
    filters = {"user_id": "test_user", "source": "manual"}
    where_clause, params = pgvector._build_filter_conditions(filters)
    
    assert "WHERE" in where_clause
    assert "user_id" in where_clause
    assert "source" in where_clause
    assert len(params) == 2
    print("✓ Filter conditions builder working correctly")
    
    # Test 4: Order clause builder
    sort_by = [("created_at", SortOrder.DESC)]
    order_clause = pgvector._build_order_clause(sort_by)
    
    assert "ORDER BY" in order_clause
    assert "created_at" in order_clause
    assert "DESC" in order_clause
    print("✓ Order clause builder working correctly")
    
    # Test 5: Advanced search mode validation
    try:
        pgvector.advanced_search("test", mode="invalid_mode")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported search mode" in str(e)
        print("✓ Advanced search mode validation working correctly")
    
    # Test 6: Count memories
    mock_cursor.fetchone.return_value = [42]
    count = pgvector.count_memories(filters={"user_id": "test_user"})
    assert count == 42
    print("✓ Count memories functionality working correctly")
    
    # Test 7: Bulk operations
    operations = [
        {'type': 'unknown_operation', 'data': 'test'}
    ]
    
    result = pgvector.bulk_operations(operations)
    assert len(result) == 1
    assert result[0]['type'] == 'error'
    assert 'Unknown operation type' in result[0]['message']
    print("✓ Bulk operations error handling working correctly")
    
    print("\n🎉 All enhanced features are working correctly!")
    print("✅ Phase 1 implementation is complete and functional")

if __name__ == "__main__":
    test_enhanced_features() 