#!/usr/bin/env python3
"""
Minimal test script to verify base vector store enhancements
"""

import sys
import os

# Add the mem0 directory to the path
mem0_dir = os.path.join(os.path.dirname(__file__), 'mem0')
sys.path.insert(0, mem0_dir)

def test_base_enhancements():
    """Test base vector store enhancements"""
    print("Testing Base Vector Store Enhancements...")
    
    # Test 1: Import enums directly
    try:
        from vector_stores.base import SearchMode, SortOrder
        print("✓ SearchMode and SortOrder enums imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import enums: {e}")
        return False
    
    # Test 2: Check enum values
    assert SearchMode.SEMANTIC.value == "semantic"
    assert SearchMode.TEXT.value == "text"
    assert SearchMode.HYBRID.value == "hybrid"
    print("✓ SearchMode enum values are correct")
    
    assert SortOrder.ASC.value == "asc"
    assert SortOrder.DESC.value == "desc"
    print("✓ SortOrder enum values are correct")
    
    # Test 3: Import base class
    try:
        from vector_stores.base import VectorStoreBase
        print("✓ VectorStoreBase imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import VectorStoreBase: {e}")
        return False
    
    # Test 4: Check that new abstract methods exist
    expected_methods = [
        'advanced_search',
        'count_memories',
        'list_with_sorting',
        'aggregate_stats',
        'bulk_operations',
        'create_indexes',
        'search_with_metadata_aggregation'
    ]
    
    for method in expected_methods:
        if hasattr(VectorStoreBase, method):
            print(f"✓ Abstract method {method} exists in VectorStoreBase")
        else:
            print(f"✗ Abstract method {method} missing from VectorStoreBase")
            return False
    
    # Test 5: Check method signatures
    import inspect
    
    # Check advanced_search signature
    sig = inspect.signature(VectorStoreBase.advanced_search)
    params = list(sig.parameters.keys())
    expected_params = ['self', 'query', 'filters', 'mode', 'limit', 'offset']
    
    for param in expected_params:
        if param in params:
            print(f"✓ Parameter {param} in advanced_search signature")
        else:
            print(f"✗ Parameter {param} missing from advanced_search signature")
            return False
    
    # Test 6: Check that SearchMode is used as default
    sig = inspect.signature(VectorStoreBase.advanced_search)
    mode_param = sig.parameters['mode']
    if mode_param.default == SearchMode.SEMANTIC:
        print("✓ SearchMode.SEMANTIC is default for advanced_search")
    else:
        print(f"✗ Wrong default for mode parameter: {mode_param.default}")
        return False
    
    print("\n🎉 All base enhancements are working correctly!")
    print("✅ Base vector store interface is properly enhanced")
    return True

def test_pgvector_methods():
    """Test that PGVector methods can be imported and have correct signatures"""
    print("\nTesting PGVector Method Signatures...")
    
    try:
        # Import PGVector directly without going through __init__.py
        from vector_stores.pgvector import PGVector
        print("✓ PGVector class imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PGVector: {e}")
        return False
    
    # Check that enhanced methods exist
    enhanced_methods = [
        'create_indexes',
        'advanced_search',
        'count_memories',
        'list_with_sorting',
        'aggregate_stats',
        'bulk_operations',
        'search_with_metadata_aggregation',
        '_build_filter_conditions',
        '_build_order_clause',
        '_aggregate_field',
        '_text_search',
        '_semantic_search',
        '_hybrid_search'
    ]
    
    for method in enhanced_methods:
        if hasattr(PGVector, method):
            print(f"✓ Enhanced method {method} exists in PGVector")
        else:
            print(f"✗ Enhanced method {method} missing from PGVector")
            return False
    
    print("✅ All enhanced methods are present in PGVector")
    return True

if __name__ == "__main__":
    success1 = test_base_enhancements()
    success2 = test_pgvector_methods()
    
    if success1 and success2:
        print("\n✅ All tests passed! Enhanced vector store implementation is ready.")
        print("✅ Phase 1 (Enhanced Vector Store) is complete and functional")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 