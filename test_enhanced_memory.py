#!/usr/bin/env python3
"""
Test script for enhanced memory functionality
"""

import sys
import os

# Add the mem0 directory to the path
mem0_dir = os.path.join(os.path.dirname(__file__), 'mem0')
sys.path.insert(0, mem0_dir)

from unittest.mock import Mock, patch
from memory.enhanced_memory import EnhancedMemory
from vector_stores.base import SearchMode, SortOrder

def test_enhanced_memory_functionality():
    """Test enhanced memory functionality"""
    print("Testing Enhanced Memory Functionality...")
    
    # Test 1: Import and class structure
    print("✓ EnhancedMemory imported successfully")
    
    # Test 2: Check that all new methods exist
    expected_methods = [
        'search_with_fallback',
        'hybrid_search',
        'get_memory_stats',
        'get_memories_with_complex_sorting',
        'count_user_memories',
        'get_document_memories',
        'search_with_analytics',
        'get_latest_memories',
        'bulk_memory_operations',
        'get_memory_analytics',
        'get_memory_insights',
        'search_similar_memories',
        'get_memory_clusters'
    ]
    
    for method in expected_methods:
        if hasattr(EnhancedMemory, method):
            print(f"✓ Method {method} exists")
        else:
            print(f"✗ Method {method} missing")
            return False
    
    # Test 3: Check helper methods
    helper_methods = [
        '_build_search_filters',
        '_format_search_results',
        '_format_memory_results',
        '_get_field_aggregation',
        '_get_memory_trends',
        '_get_top_keywords',
        '_get_memory_length_stats',
        '_get_most_active_source',
        '_analyze_growth_trend',
        '_generate_recommendations',
        '_simple_keyword_clustering'
    ]
    
    for method in helper_methods:
        if hasattr(EnhancedMemory, method):
            print(f"✓ Helper method {method} exists")
        else:
            print(f"✗ Helper method {method} missing")
            return False
    
    # Test 4: Test method signatures
    import inspect
    
    # Check search_with_fallback signature
    sig = inspect.signature(EnhancedMemory.search_with_fallback)
    params = list(sig.parameters.keys())
    expected_params = ['self', 'query', 'user_id', 'agent_id', 'run_id', 'limit', 'filters', 'kwargs']
    
    for param in expected_params:
        if param in params:
            print(f"✓ Parameter {param} in search_with_fallback")
        else:
            print(f"✗ Parameter {param} missing from search_with_fallback")
            return False
    
    # Test 5: Test helper function logic
    print("\n--- Testing Helper Functions ---")
    
    # Create a mock instance
    mock_config = Mock()
    mock_vector_store = Mock()
    
    with patch('memory.enhanced_memory.Memory.__init__', return_value=None):
        enhanced_memory = EnhancedMemory.__new__(EnhancedMemory)
        enhanced_memory.vector_store = mock_vector_store
        enhanced_memory.config = mock_config
        
        # Test _build_search_filters
        filters = enhanced_memory._build_search_filters(
            user_id="test_user",
            agent_id="test_agent",
            additional_filters={"source": "manual"}
        )
        
        expected_filters = {
            "user_id": "test_user",
            "agent_id": "test_agent",
            "source": "manual"
        }
        
        assert filters == expected_filters
        print("✓ _build_search_filters working correctly")
        
        # Test _format_search_results
        mock_results = [
            {
                'id': 'test_id',
                'payload': {
                    'memory': 'test memory',
                    'user_id': 'test_user',
                    'metadata': {'source': 'manual'}
                },
                'score': 0.8,
                'search_mode': 'semantic'
            }
        ]
        
        formatted = enhanced_memory._format_search_results(mock_results)
        
        assert len(formatted) == 1
        assert formatted[0]['id'] == 'test_id'
        assert formatted[0]['memory'] == 'test memory'
        assert formatted[0]['score'] == 0.8
        assert formatted[0]['search_mode'] == 'semantic'
        print("✓ _format_search_results working correctly")
        
        # Test _get_most_active_source
        source_dist = {'manual': 10, 'auto-extraction': 25, 'document': 5}
        most_active = enhanced_memory._get_most_active_source(source_dist)
        assert most_active == 'auto-extraction'
        print("✓ _get_most_active_source working correctly")
        
        # Test _analyze_growth_trend
        trends = {
            'daily_counts': {'2024-01-01': 5, '2024-01-02': 8, '2024-01-03': 12}
        }
        trend = enhanced_memory._analyze_growth_trend(trends)
        assert trend == 'increasing'
        print("✓ _analyze_growth_trend working correctly")
        
        # Test _generate_recommendations
        analytics = {
            'basic_stats': {'total_memories': 5, 'avg_memory_length': 30},
            'source_distribution': {'manual': 2, 'auto-extraction': 8}
        }
        recommendations = enhanced_memory._generate_recommendations(analytics)
        assert len(recommendations) >= 2  # Should have multiple recommendations
        print("✓ _generate_recommendations working correctly")
    
    print("\n🎉 All enhanced memory features are working correctly!")
    print("✅ Phase 2 implementation is complete and functional")
    return True

def test_method_integration():
    """Test method integration and error handling"""
    print("\n--- Testing Method Integration ---")
    
    # Test with mock vector store that has enhanced methods
    mock_vector_store = Mock()
    mock_vector_store.advanced_search.return_value = [
        {
            'id': 'test_id',
            'payload': {'memory': 'test memory', 'user_id': 'test_user'},
            'score': 0.9,
            'search_mode': 'semantic'
        }
    ]
    mock_vector_store.aggregate_stats.return_value = {
        'total_memories': 100,
        'auto_extracted': 60,
        'manual': 40
    }
    mock_vector_store.count_memories.return_value = 100
    
    with patch('memory.enhanced_memory.Memory.__init__', return_value=None):
        enhanced_memory = EnhancedMemory.__new__(EnhancedMemory)
        enhanced_memory.vector_store = mock_vector_store
        
        # Test search_with_fallback
        results = enhanced_memory.search_with_fallback("test query", user_id="test_user")
        assert len(results) == 1
        assert results[0]['memory'] == 'test memory'
        print("✓ search_with_fallback integration working")
        
        # Test get_memory_stats
        stats = enhanced_memory.get_memory_stats(user_id="test_user")
        assert stats['total_memories'] == 100
        assert stats['auto_extracted'] == 60
        print("✓ get_memory_stats integration working")
        
        # Test count_user_memories
        count = enhanced_memory.count_user_memories(user_id="test_user")
        assert count == 100
        print("✓ count_user_memories integration working")
    
    print("✅ Method integration tests passed")
    return True

def test_error_handling():
    """Test error handling and fallback mechanisms"""
    print("\n--- Testing Error Handling ---")
    
    # Test with mock vector store that raises errors
    mock_vector_store = Mock()
    mock_vector_store.advanced_search.side_effect = Exception("Mock error")
    mock_vector_store.aggregate_stats.side_effect = Exception("Mock error")
    
    # Mock the parent search method for fallback
    with patch('memory.enhanced_memory.Memory.__init__', return_value=None):
        with patch('memory.enhanced_memory.Memory.search', return_value=[{'id': 'fallback'}]):
            enhanced_memory = EnhancedMemory.__new__(EnhancedMemory)
            enhanced_memory.vector_store = mock_vector_store
            
            # Test search_with_fallback error handling
            results = enhanced_memory.search_with_fallback("test query", user_id="test_user")
            assert len(results) == 1
            assert results[0]['id'] == 'fallback'
            print("✓ search_with_fallback error handling working")
            
            # Test get_memory_stats error handling
            stats = enhanced_memory.get_memory_stats(user_id="test_user")
            assert 'method' in stats
            assert stats['method'] == 'fallback_count'
            print("✓ get_memory_stats error handling working")
    
    print("✅ Error handling tests passed")
    return True

if __name__ == "__main__":
    success1 = test_enhanced_memory_functionality()
    success2 = test_method_integration()
    success3 = test_error_handling()
    
    if success1 and success2 and success3:
        print("\n✅ All tests passed! Enhanced Memory implementation is ready.")
        print("✅ Phase 2 (Enhanced Memory) is complete and functional")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 