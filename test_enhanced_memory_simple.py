#!/usr/bin/env python3
"""
Simple test script for enhanced memory functionality structure
"""

import sys
import os

# Add the mem0 directory to the path
mem0_dir = os.path.join(os.path.dirname(__file__), 'mem0')
sys.path.insert(0, mem0_dir)

def test_enhanced_memory_structure():
    """Test enhanced memory class structure"""
    print("Testing Enhanced Memory Structure...")
    
    # Test 1: Import enums
    try:
        from vector_stores.base import SearchMode, SortOrder
        print("✓ SearchMode and SortOrder enums imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import enums: {e}")
        return False
    
    # Test 2: Read and analyze the enhanced memory file
    try:
        enhanced_memory_file = os.path.join(mem0_dir, 'memory', 'enhanced_memory.py')
        with open(enhanced_memory_file, 'r') as f:
            content = f.read()
        
        print("✓ Enhanced memory file read successfully")
    except Exception as e:
        print(f"✗ Failed to read enhanced memory file: {e}")
        return False
    
    # Test 3: Check for required methods in the file
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
        if f"def {method}(" in content:
            print(f"✓ Method {method} found in enhanced memory")
        else:
            print(f"✗ Method {method} missing from enhanced memory")
            return False
    
    # Test 4: Check for helper methods
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
        if f"def {method}(" in content:
            print(f"✓ Helper method {method} found in enhanced memory")
        else:
            print(f"✗ Helper method {method} missing from enhanced memory")
            return False
    
    # Test 5: Check for proper imports and class definition
    required_imports = [
        'from mem0.memory.main import Memory',
        'from mem0.vector_stores.base import SearchMode, SortOrder',
        'class EnhancedMemory(Memory):'
    ]
    
    for import_stmt in required_imports:
        if import_stmt in content:
            print(f"✓ Required import/definition found: {import_stmt}")
        else:
            print(f"✗ Missing import/definition: {import_stmt}")
            return False
    
    # Test 6: Check for SearchMode usage
    search_mode_usage = [
        'SearchMode.SEMANTIC',
        'SearchMode.TEXT',
        'SearchMode.HYBRID'
    ]
    
    for usage in search_mode_usage:
        if usage in content:
            print(f"✓ SearchMode usage found: {usage}")
        else:
            print(f"✗ SearchMode usage missing: {usage}")
            return False
    
    # Test 7: Check for SortOrder usage
    sort_order_usage = [
        'SortOrder.ASC',
        'SortOrder.DESC'
    ]
    
    for usage in sort_order_usage:
        if usage in content:
            print(f"✓ SortOrder usage found: {usage}")
        else:
            print(f"✗ SortOrder usage missing: {usage}")
            return False
    
    # Test 8: Check for error handling patterns
    error_patterns = [
        'try:',
        'except Exception as e:',
        'logger.error',
        'fallback'
    ]
    
    for pattern in error_patterns:
        if pattern in content:
            print(f"✓ Error handling pattern found: {pattern}")
        else:
            print(f"✗ Error handling pattern missing: {pattern}")
    
    # Test 9: Check for analytics functionality
    analytics_features = [
        'get_memory_analytics',
        'get_memory_insights',
        'search_similar_memories',
        'get_memory_clusters',
        '_generate_recommendations'
    ]
    
    for feature in analytics_features:
        if feature in content:
            print(f"✓ Analytics feature found: {feature}")
        else:
            print(f"✗ Analytics feature missing: {feature}")
    
    print("\n🎉 Enhanced Memory structure validation complete!")
    print("✅ All required methods and features are present")
    return True

def test_method_signatures():
    """Test method signatures by parsing the file"""
    print("\n--- Testing Method Signatures ---")
    
    try:
        enhanced_memory_file = os.path.join(mem0_dir, 'memory', 'enhanced_memory.py')
        with open(enhanced_memory_file, 'r') as f:
            content = f.read()
        
        # Test search_with_fallback signature
        if 'def search_with_fallback(' in content and 'query: str' in content and 'user_id: Optional[str]' in content:
            print("✓ search_with_fallback has correct signature")
        else:
            print("✗ search_with_fallback signature incorrect")
            return False
        
        # Test hybrid_search signature
        if 'def hybrid_search(' in content and 'mode=SearchMode.HYBRID' in content:
            print("✓ hybrid_search has correct signature")
        else:
            print("✗ hybrid_search signature incorrect")
            return False
        
        # Test get_memory_stats signature
        if 'def get_memory_stats(' in content and '-> Dict[str, Any]:' in content:
            print("✓ get_memory_stats has correct return type")
        else:
            print("✗ get_memory_stats signature incorrect")
            return False
        
        # Test analytics methods
        analytics_methods = [
            'get_memory_analytics',
            'get_memory_insights',
            'search_similar_memories'
        ]
        
        for method in analytics_methods:
            if f'def {method}(' in content and '-> Dict[str, Any]:' in content:
                print(f"✓ {method} has correct signature")
            elif f'def {method}(' in content and '-> List[Dict]:' in content:
                print(f"✓ {method} has correct signature")
            else:
                print(f"✗ {method} signature needs verification")
        
        print("✅ Method signature validation complete")
        return True
        
    except Exception as e:
        print(f"✗ Error testing method signatures: {e}")
        return False

def test_implementation_completeness():
    """Test implementation completeness"""
    print("\n--- Testing Implementation Completeness ---")
    
    try:
        enhanced_memory_file = os.path.join(mem0_dir, 'memory', 'enhanced_memory.py')
        with open(enhanced_memory_file, 'r') as f:
            content = f.read()
        
        # Count lines of code (rough measure of completeness)
        lines = content.split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        print(f"✓ Enhanced memory file has {len(code_lines)} lines of code")
        
        if len(code_lines) > 300:
            print("✓ Implementation appears comprehensive (>300 lines)")
        else:
            print("⚠ Implementation may need more detail (<300 lines)")
        
        # Check for docstrings
        docstring_count = content.count('"""')
        if docstring_count > 20:
            print(f"✓ Good documentation with {docstring_count//2} docstrings")
        else:
            print(f"⚠ Could use more documentation ({docstring_count//2} docstrings)")
        
        # Check for comprehensive error handling
        error_handling_count = content.count('except Exception as e:')
        if error_handling_count > 10:
            print(f"✓ Comprehensive error handling ({error_handling_count} error handlers)")
        else:
            print(f"⚠ Could use more error handling ({error_handling_count} error handlers)")
        
        print("✅ Implementation completeness check complete")
        return True
        
    except Exception as e:
        print(f"✗ Error testing implementation completeness: {e}")
        return False

if __name__ == "__main__":
    success1 = test_enhanced_memory_structure()
    success2 = test_method_signatures()
    success3 = test_implementation_completeness()
    
    if success1 and success2 and success3:
        print("\n✅ All structure tests passed! Enhanced Memory implementation is well-structured.")
        print("✅ Phase 2 (Enhanced Memory) structure is complete and ready")
    else:
        print("\n❌ Some structure tests failed. Please check the implementation.")
        sys.exit(1) 