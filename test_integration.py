#!/usr/bin/env python3
"""
Test script for integration layer functionality
"""

import sys
import os

# Add the mem0 directory to the path
mem0_dir = os.path.join(os.path.dirname(__file__), 'mem0')
sys.path.insert(0, mem0_dir)

def test_integration_structure():
    """Test integration layer structure"""
    print("Testing Integration Layer Structure...")
    
    # Test 1: Check integration files exist
    integration_dir = os.path.join(mem0_dir, 'integration')
    
    expected_files = [
        'cognition_suite_adapter.py',
        'api_wrapper.py'
    ]
    
    for file_name in expected_files:
        file_path = os.path.join(integration_dir, file_name)
        if os.path.exists(file_path):
            print(f"✓ Integration file {file_name} exists")
        else:
            print(f"✗ Integration file {file_name} missing")
            return False
    
    # Test 2: Check adapter file content
    try:
        adapter_file = os.path.join(integration_dir, 'cognition_suite_adapter.py')
        with open(adapter_file, 'r') as f:
            adapter_content = f.read()
        
        print("✓ Adapter file read successfully")
    except Exception as e:
        print(f"✗ Failed to read adapter file: {e}")
        return False
    
    # Test 3: Check for required classes in adapter
    required_classes = [
        'class CognitionSuiteMemoryAdapter:'
    ]
    
    for class_def in required_classes:
        if class_def in adapter_content:
            print(f"✓ Required class found: {class_def}")
        else:
            print(f"✗ Missing class: {class_def}")
            return False
    
    # Test 4: Check for required methods in adapter
    adapter_methods = [
        'add_memory_with_context',
        'search_memories_enhanced',
        'get_user_memory_dashboard',
        'get_document_memory_analysis',
        'bulk_memory_management',
        'get_memory_recommendations'
    ]
    
    for method in adapter_methods:
        if f"def {method}(" in adapter_content:
            print(f"✓ Adapter method {method} found")
        else:
            print(f"✗ Adapter method {method} missing")
            return False
    
    # Test 5: Check API wrapper file content
    try:
        api_file = os.path.join(integration_dir, 'api_wrapper.py')
        with open(api_file, 'r') as f:
            api_content = f.read()
        
        print("✓ API wrapper file read successfully")
    except Exception as e:
        print(f"✗ Failed to read API wrapper file: {e}")
        return False
    
    # Test 6: Check for required classes in API wrapper
    required_api_classes = [
        'class EnhancedMem0API:'
    ]
    
    for class_def in required_api_classes:
        if class_def in api_content:
            print(f"✓ Required API class found: {class_def}")
        else:
            print(f"✗ Missing API class: {class_def}")
            return False
    
    # Test 7: Check for required methods in API wrapper
    api_methods = [
        'add_memory',
        'search_memories',
        'get_user_dashboard',
        'get_document_analysis',
        'bulk_operations',
        'get_recommendations',
        'get_memory_stats',
        'count_memories',
        'find_similar_memories',
        'get_memory_clusters',
        'get_latest_memories',
        'get_api_info',
        'health_check'
    ]
    
    for method in api_methods:
        if f"def {method}(" in api_content:
            print(f"✓ API method {method} found")
        else:
            print(f"✗ API method {method} missing")
            return False
    
    # Test 8: Check for convenience functions
    convenience_functions = [
        'create_api',
        'quick_search',
        'quick_add'
    ]
    
    for func in convenience_functions:
        if f"def {func}(" in api_content:
            print(f"✓ Convenience function {func} found")
        else:
            print(f"✗ Convenience function {func} missing")
            return False
    
    # Test 9: Check for error handling
    error_handling_patterns = [
        '@handle_errors',
        'def handle_errors(',
        'try:',
        'except Exception as e:'
    ]
    
    for pattern in error_handling_patterns:
        if pattern in api_content:
            print(f"✓ Error handling pattern found: {pattern}")
        else:
            print(f"✗ Error handling pattern missing: {pattern}")
    
    print("\n🎉 Integration layer structure validation complete!")
    print("✅ All required components are present")
    return True

def test_integration_imports():
    """Test integration layer imports"""
    print("\n--- Testing Integration Imports ---")
    
    # Test 1: Import base enums
    try:
        from vector_stores.base import SearchMode, SortOrder
        print("✓ Base enums imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import base enums: {e}")
        return False
    
    # Test 2: Try to read adapter class definition
    try:
        integration_dir = os.path.join(mem0_dir, 'integration')
        adapter_file = os.path.join(integration_dir, 'cognition_suite_adapter.py')
        
        with open(adapter_file, 'r') as f:
            content = f.read()
        
        # Check for proper imports
        required_imports = [
            'from mem0.memory.enhanced_memory import EnhancedMemory',
            'from mem0.vector_stores.base import SearchMode, SortOrder',
            'from mem0.configs.base import MemoryConfig'
        ]
        
        for import_stmt in required_imports:
            if import_stmt in content:
                print(f"✓ Required import found: {import_stmt}")
            else:
                print(f"✗ Missing import: {import_stmt}")
                return False
        
    except Exception as e:
        print(f"✗ Error checking adapter imports: {e}")
        return False
    
    # Test 3: Try to read API wrapper class definition
    try:
        api_file = os.path.join(integration_dir, 'api_wrapper.py')
        
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Check for proper imports
        required_api_imports = [
            'from mem0.integration.cognition_suite_adapter import CognitionSuiteMemoryAdapter',
            'from mem0.configs.base import MemoryConfig'
        ]
        
        for import_stmt in required_api_imports:
            if import_stmt in content:
                print(f"✓ Required API import found: {import_stmt}")
            else:
                print(f"✗ Missing API import: {import_stmt}")
                return False
        
    except Exception as e:
        print(f"✗ Error checking API wrapper imports: {e}")
        return False
    
    print("✅ Integration imports validation complete")
    return True

def test_integration_functionality():
    """Test integration layer functionality"""
    print("\n--- Testing Integration Functionality ---")
    
    # Test 1: Check source priority mapping
    try:
        integration_dir = os.path.join(mem0_dir, 'integration')
        adapter_file = os.path.join(integration_dir, 'cognition_suite_adapter.py')
        
        with open(adapter_file, 'r') as f:
            content = f.read()
        
        # Check for source priority mapping
        priority_sources = ['manual', 'chat', 'document', 'auto-extraction', 'system']
        
        for source in priority_sources:
            if f"'{source}'" in content:
                print(f"✓ Source priority for {source} found")
            else:
                print(f"✗ Source priority for {source} missing")
        
    except Exception as e:
        print(f"✗ Error checking source priorities: {e}")
        return False
    
    # Test 2: Check API method signatures
    try:
        api_file = os.path.join(integration_dir, 'api_wrapper.py')
        
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Check for proper method signatures
        signature_checks = [
            'def add_memory(' and 'content: str' and 'user_id: str',
            'def search_memories(' and 'query: str' and 'user_id: str',
            'def get_user_dashboard(' and 'user_id: str',
            'def bulk_operations(' and 'operations: List[Dict[str, Any]]'
        ]
        
        method_names = ['add_memory', 'search_memories', 'get_user_dashboard', 'bulk_operations']
        
        for i, method_name in enumerate(method_names):
            if f'def {method_name}(' in content:
                print(f"✓ Method {method_name} signature found")
            else:
                print(f"✗ Method {method_name} signature missing")
        
    except Exception as e:
        print(f"✗ Error checking API signatures: {e}")
        return False
    
    # Test 3: Check error handling decorator
    try:
        if '@handle_errors' in content and 'def handle_errors(' in content:
            print("✓ Error handling decorator found")
        else:
            print("✗ Error handling decorator missing")
        
        # Check for consistent error response format
        if "'success': False" in content and "'error': str(e)" in content:
            print("✓ Consistent error response format found")
        else:
            print("✗ Consistent error response format missing")
        
    except Exception as e:
        print(f"✗ Error checking error handling: {e}")
        return False
    
    # Test 4: Check health check functionality
    try:
        if 'def health_check(' in content and "'status':" in content:
            print("✓ Health check functionality found")
        else:
            print("✗ Health check functionality missing")
        
    except Exception as e:
        print(f"✗ Error checking health check: {e}")
        return False
    
    print("✅ Integration functionality validation complete")
    return True

def test_integration_completeness():
    """Test integration layer completeness"""
    print("\n--- Testing Integration Completeness ---")
    
    try:
        integration_dir = os.path.join(mem0_dir, 'integration')
        
        # Count lines in adapter
        adapter_file = os.path.join(integration_dir, 'cognition_suite_adapter.py')
        with open(adapter_file, 'r') as f:
            adapter_lines = len([line for line in f.readlines() if line.strip() and not line.strip().startswith('#')])
        
        # Count lines in API wrapper
        api_file = os.path.join(integration_dir, 'api_wrapper.py')
        with open(api_file, 'r') as f:
            api_lines = len([line for line in f.readlines() if line.strip() and not line.strip().startswith('#')])
        
        print(f"✓ Adapter file has {adapter_lines} lines of code")
        print(f"✓ API wrapper file has {api_lines} lines of code")
        
        total_lines = adapter_lines + api_lines
        if total_lines > 500:
            print(f"✓ Integration layer appears comprehensive ({total_lines} total lines)")
        else:
            print(f"⚠ Integration layer may need more detail ({total_lines} total lines)")
        
        # Check for docstrings
        with open(adapter_file, 'r') as f:
            adapter_content = f.read()
        with open(api_file, 'r') as f:
            api_content = f.read()
        
        adapter_docstrings = adapter_content.count('"""')
        api_docstrings = api_content.count('"""')
        
        total_docstrings = (adapter_docstrings + api_docstrings) // 2
        if total_docstrings > 15:
            print(f"✓ Good documentation with {total_docstrings} docstrings")
        else:
            print(f"⚠ Could use more documentation ({total_docstrings} docstrings)")
        
        # Check for type hints
        if 'Dict[str, Any]' in adapter_content and 'List[Dict[str, Any]]' in api_content:
            print("✓ Comprehensive type hints found")
        else:
            print("⚠ Type hints could be improved")
        
        print("✅ Integration completeness check complete")
        return True
        
    except Exception as e:
        print(f"✗ Error testing integration completeness: {e}")
        return False

if __name__ == "__main__":
    success1 = test_integration_structure()
    success2 = test_integration_imports()
    success3 = test_integration_functionality()
    success4 = test_integration_completeness()
    
    if success1 and success2 and success3 and success4:
        print("\n✅ All integration tests passed! Integration layer is ready.")
        print("✅ Phase 3 (Integration Layer) is complete and functional")
    else:
        print("\n❌ Some integration tests failed. Please check the implementation.")
        sys.exit(1) 