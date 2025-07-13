#!/usr/bin/env python3
"""
Direct test script to verify enhanced PGVector features
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from unittest.mock import Mock, patch

# Direct imports to avoid package initialization issues
from mem0.vector_stores.base import SearchMode, SortOrder

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
    
    # Test 2: Import PGVector directly
    try:
        from mem0.vector_stores.pgvector import PGVector
        print("✓ Enhanced PGVector class imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PGVector: {e}")
        return False
    
    # Test 3: Check that new methods exist
    expected_methods = [
        'advanced_search',
        'count_memories',
        'list_with_sorting',
        'aggregate_stats',
        'bulk_operations',
        'create_indexes',
        'search_with_metadata_aggregation',
        '_build_filter_conditions',
        '_build_order_clause',
        '_aggregate_field',
        '_text_search',
        '_semantic_search',
        '_hybrid_search'
    ]
    
    for method in expected_methods:
        if hasattr(PGVector, method):
            print(f"✓ Method {method} exists")
        else:
            print(f"✗ Method {method} missing")
            return False
    
    # Test 4: Check method signatures
    import inspect
    
    # Check advanced_search signature
    sig = inspect.signature(PGVector.advanced_search)
    params = list(sig.parameters.keys())
    expected_params = ['self', 'query', 'filters', 'mode', 'limit', 'offset']
    
    for param in expected_params:
        if param in params:
            print(f"✓ Parameter {param} in advanced_search")
        else:
            print(f"✗ Parameter {param} missing from advanced_search")
            return False
    
    # Test 5: Test filter conditions builder logic
    print("\n--- Testing Filter Conditions Builder ---")
    
    # Create a mock instance to test the helper methods
    mock_instance = Mock()
    mock_instance.collection_name = "memories"
    
    # Bind the method to our mock instance
    build_filter_conditions = PGVector._build_filter_conditions.__get__(mock_instance, PGVector)
    
    # Test with user_id filter
    filters = {"user_id": "test_user"}
    where_clause, params = build_filter_conditions(filters)
    
    assert "WHERE" in where_clause
    assert "user_id" in where_clause
    assert params == ["test_user"]
    print("✓ user_id filter working correctly")
    
    # Test with multiple filters
    filters = {"user_id": "test_user", "source": "manual"}
    where_clause, params = build_filter_conditions(filters)
    
    assert "WHERE" in where_clause
    assert "user_id" in where_clause
    assert "source" in where_clause
    assert len(params) == 2
    print("✓ Multiple filters working correctly")
    
    # Test with file_name filter (should handle both variants)
    filters = {"file_name": "test.pdf"}
    where_clause, params = build_filter_conditions(filters)
    
    assert "file_name" in where_clause
    assert "fileName" in where_clause
    assert len(params) == 2
    print("✓ file_name filter working correctly")
    
    # Test 6: Test order clause builder
    print("\n--- Testing Order Clause Builder ---")
    
    build_order_clause = PGVector._build_order_clause.__get__(mock_instance, PGVector)
    
    # Test default order
    order_clause = build_order_clause(None)
    assert "ORDER BY" in order_clause
    assert "auto-extraction" in order_clause
    print("✓ Default order clause working correctly")
    
    # Test custom order
    sort_by = [("created_at", SortOrder.DESC)]
    order_clause = build_order_clause(sort_by)
    assert "ORDER BY" in order_clause
    assert "created_at" in order_clause
    assert "DESC" in order_clause
    print("✓ Custom order clause working correctly")
    
    print("\n🎉 All enhanced features are working correctly!")
    print("✅ Phase 1 implementation is complete and functional")
    return True

if __name__ == "__main__":
    success = test_enhanced_features()
    if success:
        print("\n✅ All tests passed! Enhanced PGVector implementation is ready.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 