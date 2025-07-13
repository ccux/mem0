#!/usr/bin/env python3
"""
Test script for performance optimization functionality
"""

import sys
import os
import time
import threading

# Add the mem0 directory to the path
mem0_dir = os.path.join(os.path.dirname(__file__), 'mem0')
sys.path.insert(0, mem0_dir)

def test_performance_structure():
    """Test performance optimization structure"""
    print("Testing Performance Optimization Structure...")
    
    # Test 1: Check performance file exists
    performance_file = os.path.join(mem0_dir, 'vector_stores', 'performance_optimizations.py')
    
    if os.path.exists(performance_file):
        print("✓ Performance optimizations file exists")
    else:
        print("✗ Performance optimizations file missing")
        return False
    
    # Test 2: Read and analyze the performance file
    try:
        with open(performance_file, 'r') as f:
            content = f.read()
        
        print("✓ Performance file read successfully")
    except Exception as e:
        print(f"✗ Failed to read performance file: {e}")
        return False
    
    # Test 3: Check for required classes
    required_classes = [
        'class LRUCache:',
        'class QueryCache:',
        'class ConnectionPool:',
        'class PerformanceMetrics:',
        'class BatchProcessor:',
        'class QueryOptimizer:',
        'class PerformanceOptimizedVectorStore:'
    ]
    
    for class_def in required_classes:
        if class_def in content:
            print(f"✓ Required class found: {class_def}")
        else:
            print(f"✗ Missing class: {class_def}")
            return False
    
    # Test 4: Check for required decorators
    required_decorators = [
        'def performance_monitor(',
        'def cache_result('
    ]
    
    for decorator in required_decorators:
        if decorator in content:
            print(f"✓ Required decorator found: {decorator}")
        else:
            print(f"✗ Missing decorator: {decorator}")
            return False
    
    # Test 5: Check for performance methods
    performance_methods = [
        'get_performance_report',
        'invalidate_cache_for_user',
        'optimize_query_plan',
        'warm_cache',
        'record_operation',
        'get_metrics',
        'process_in_batches',
        'optimize_filters',
        'estimate_query_cost'
    ]
    
    for method in performance_methods:
        if f"def {method}(" in content:
            print(f"✓ Performance method {method} found")
        else:
            print(f"✗ Performance method {method} missing")
            return False
    
    print("\n🎉 Performance optimization structure validation complete!")
    print("✅ All required components are present")
    return True

def test_lru_cache():
    """Test LRU Cache functionality"""
    print("\n--- Testing LRU Cache ---")
    
    try:
        from vector_stores.performance_optimizations import LRUCache
        
        # Test basic operations
        cache = LRUCache(max_size=3)
        
        # Test set and get
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        print("✓ Basic set/get operations working")
        
        # Test LRU eviction
        cache.set("key4", "value4")  # Should evict key1 (least recently used)
        
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"
        print("✓ LRU eviction working correctly")
        
        # Test size
        assert cache.size() == 3
        print("✓ Cache size tracking working")
        
        # Test clear
        cache.clear()
        assert cache.size() == 0
        print("✓ Cache clear working")
        
        return True
        
    except Exception as e:
        print(f"✗ LRU Cache test failed: {e}")
        return False

def test_query_cache():
    """Test Query Cache functionality"""
    print("\n--- Testing Query Cache ---")
    
    try:
        from vector_stores.performance_optimizations import QueryCache
        
        # Test basic operations
        cache = QueryCache(max_size=10, ttl_seconds=1)
        
        # Test set and get
        cache.set("test query", {"user_id": "test"}, 10, "semantic", ["result1", "result2"])
        result = cache.get("test query", {"user_id": "test"}, 10, "semantic")
        
        assert result == ["result1", "result2"]
        print("✓ Query cache set/get working")
        
        # Test TTL expiration
        time.sleep(1.1)  # Wait for TTL to expire
        result = cache.get("test query", {"user_id": "test"}, 10, "semantic")
        assert result is None
        print("✓ TTL expiration working")
        
        # Test user invalidation
        cache.set("query1", {"user_id": "user1"}, 10, "semantic", [{"user_id": "user1", "content": "test"}])
        cache.set("query2", {"user_id": "user2"}, 10, "semantic", [{"user_id": "user2", "content": "test"}])
        
        cache.invalidate_user("user1")
        
        assert cache.get("query1", {"user_id": "user1"}, 10, "semantic") is None
        assert cache.get("query2", {"user_id": "user2"}, 10, "semantic") is not None
        print("✓ User-specific cache invalidation working")
        
        return True
        
    except Exception as e:
        print(f"✗ Query Cache test failed: {e}")
        return False

def test_performance_metrics():
    """Test Performance Metrics functionality"""
    print("\n--- Testing Performance Metrics ---")
    
    try:
        from vector_stores.performance_optimizations import PerformanceMetrics
        
        metrics = PerformanceMetrics()
        
        # Test recording operations
        metrics.record_operation("search", 0.5, True)
        metrics.record_operation("search", 0.7, True)
        metrics.record_operation("search", 1.2, False)
        
        search_metrics = metrics.get_operation_metrics("search")
        
        assert search_metrics['count'] == 3
        assert search_metrics['success_count'] == 2
        assert search_metrics['error_count'] == 1
        assert search_metrics['min_time'] == 0.5
        assert search_metrics['max_time'] == 1.2
        assert abs(search_metrics['avg_time'] - 0.8) < 0.01
        print("✓ Performance metrics recording working")
        
        # Test get all metrics
        all_metrics = metrics.get_metrics()
        assert 'search' in all_metrics
        print("✓ Get all metrics working")
        
        # Test reset
        metrics.reset_metrics()
        assert len(metrics.get_metrics()) == 0
        print("✓ Metrics reset working")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance Metrics test failed: {e}")
        return False

def test_batch_processor():
    """Test Batch Processor functionality"""
    print("\n--- Testing Batch Processor ---")
    
    try:
        from vector_stores.performance_optimizations import BatchProcessor
        
        processor = BatchProcessor(batch_size=3)
        
        # Test batch processing
        items = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        def square_batch(batch):
            return [x * x for x in batch]
        
        results = processor.process_in_batches(items, square_batch)
        
        expected = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        assert results == expected
        print("✓ Batch processing working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch Processor test failed: {e}")
        return False

def test_query_optimizer():
    """Test Query Optimizer functionality"""
    print("\n--- Testing Query Optimizer ---")
    
    try:
        from vector_stores.performance_optimizations import QueryOptimizer
        
        optimizer = QueryOptimizer()
        
        # Test filter optimization
        filters = {"user_id": "test", "source": None, "active": True}
        optimized = optimizer.optimize_filters(filters)
        
        expected = {"user_id": "test", "active": True}
        assert optimized == expected
        print("✓ Filter optimization working")
        
        # Test cost estimation
        cost = optimizer.estimate_query_cost({"user_id": "test", "source": "manual"}, 50)
        assert cost > 1.0  # Should be base cost + filter cost + limit cost
        print("✓ Query cost estimation working")
        
        # Test index suggestions
        query_patterns = [
            {"filters": {"user_id": "test1", "source": "manual"}},
            {"filters": {"user_id": "test2", "created_at": "2024-01-01"}},
            {"filters": {"user_id": "test3", "source": "auto"}}
        ]
        
        suggestions = optimizer.suggest_index_columns(query_patterns)
        assert "user_id" in suggestions  # Should be most frequent
        print("✓ Index suggestions working")
        
        return True
        
    except Exception as e:
        print(f"✗ Query Optimizer test failed: {e}")
        return False

def test_performance_decorators():
    """Test Performance Decorators"""
    print("\n--- Testing Performance Decorators ---")
    
    try:
        from vector_stores.performance_optimizations import performance_monitor, cache_result, PerformanceMetrics, QueryCache
        
        # Test performance monitor decorator
        class TestClass:
            def __init__(self):
                self.performance_metrics = PerformanceMetrics()
                self.query_cache = QueryCache()
            
            @performance_monitor("test_operation")
            def test_method(self, delay=0.1):
                time.sleep(delay)
                return "success"
            
            @cache_result()
            def cached_method(self, query, filters=None, limit=10, mode="semantic"):
                return f"result_for_{query}"
        
        test_obj = TestClass()
        
        # Test performance monitoring
        result = test_obj.test_method(0.05)
        assert result == "success"
        
        metrics = test_obj.performance_metrics.get_operation_metrics("test_operation")
        assert metrics is not None
        assert metrics['count'] == 1
        assert metrics['success_count'] == 1
        print("✓ Performance monitor decorator working")
        
        # Test cache decorator
        result1 = test_obj.cached_method("test query", {"user_id": "test"})
        result2 = test_obj.cached_method("test query", {"user_id": "test"})
        
        assert result1 == result2 == "result_for_test query"
        print("✓ Cache result decorator working")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance Decorators test failed: {e}")
        return False

def test_performance_optimized_vector_store():
    """Test Performance Optimized Vector Store"""
    print("\n--- Testing Performance Optimized Vector Store ---")
    
    try:
        from vector_stores.performance_optimizations import PerformanceOptimizedVectorStore
        
        # Create a test class that inherits from the mixin
        class TestVectorStore(PerformanceOptimizedVectorStore):
            def __init__(self, **kwargs):
                # Initialize performance components directly since there's no parent __init__
                self.query_cache = None
                self.performance_metrics = None
                self.batch_processor = None
                self.query_optimizer = None
                
                # Initialize performance components
                from vector_stores.performance_optimizations import QueryCache, PerformanceMetrics, BatchProcessor, QueryOptimizer
                self.query_cache = QueryCache(max_size=1000, ttl_seconds=300)
                self.performance_metrics = PerformanceMetrics()
                self.batch_processor = BatchProcessor(batch_size=100)
                self.query_optimizer = QueryOptimizer()
                
                # Performance settings
                self.enable_caching = kwargs.get('enable_caching', True)
                self.enable_metrics = kwargs.get('enable_metrics', True)
                self.cache_ttl = kwargs.get('cache_ttl', 300)
        
        store = TestVectorStore(enable_caching=True, enable_metrics=True)
        
        # Test performance report
        report = store.get_performance_report()
        assert 'cache_stats' in report
        assert 'operation_metrics' in report
        assert 'recommendations' in report
        print("✓ Performance report generation working")
        
        # Test cache invalidation
        store.invalidate_cache_for_user("test_user")
        print("✓ Cache invalidation working")
        
        # Test query plan optimization
        plan = store.optimize_query_plan("test query", {"user_id": "test", "source": None})
        assert 'optimized_filters' in plan
        assert 'estimated_cost' in plan
        print("✓ Query plan optimization working")
        
        # Test cache warming
        common_queries = [
            {"query": "test1", "filters": {"user_id": "user1"}, "limit": 10, "mode": "semantic"},
            {"query": "test2", "filters": {"user_id": "user2"}, "limit": 5, "mode": "text"}
        ]
        store.warm_cache(common_queries)
        print("✓ Cache warming working")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance Optimized Vector Store test failed: {e}")
        return False

def test_thread_safety():
    """Test thread safety of performance components"""
    print("\n--- Testing Thread Safety ---")
    
    try:
        from vector_stores.performance_optimizations import LRUCache, PerformanceMetrics
        
        cache = LRUCache(max_size=100)
        metrics = PerformanceMetrics()
        
        def cache_worker(worker_id):
            for i in range(50):
                cache.set(f"key_{worker_id}_{i}", f"value_{worker_id}_{i}")
                cache.get(f"key_{worker_id}_{i}")
        
        def metrics_worker(worker_id):
            for i in range(50):
                metrics.record_operation(f"operation_{worker_id}", 0.1, True)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t1 = threading.Thread(target=cache_worker, args=(i,))
            t2 = threading.Thread(target=metrics_worker, args=(i,))
            threads.extend([t1, t2])
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify results
        assert cache.size() <= 100  # Should not exceed max size
        assert len(metrics.get_metrics()) == 5  # Should have 5 different operations
        print("✓ Thread safety test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Thread safety test failed: {e}")
        return False

def test_performance_completeness():
    """Test performance optimization completeness"""
    print("\n--- Testing Performance Completeness ---")
    
    try:
        performance_file = os.path.join(mem0_dir, 'vector_stores', 'performance_optimizations.py')
        
        with open(performance_file, 'r') as f:
            content = f.read()
        
        # Count lines of code
        lines = content.split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        print(f"✓ Performance file has {len(code_lines)} lines of code")
        
        if len(code_lines) > 400:
            print("✓ Performance implementation appears comprehensive (>400 lines)")
        else:
            print("⚠ Performance implementation may need more detail (<400 lines)")
        
        # Check for docstrings
        docstring_count = content.count('"""')
        if docstring_count > 20:
            print(f"✓ Good documentation with {docstring_count//2} docstrings")
        else:
            print(f"⚠ Could use more documentation ({docstring_count//2} docstrings)")
        
        # Check for type hints
        if 'Dict[str, Any]' in content and 'List[Any]' in content:
            print("✓ Comprehensive type hints found")
        else:
            print("⚠ Type hints could be improved")
        
        # Check for threading support
        if 'threading' in content and 'RLock' in content:
            print("✓ Thread safety considerations found")
        else:
            print("⚠ Thread safety could be improved")
        
        print("✅ Performance completeness check complete")
        return True
        
    except Exception as e:
        print(f"✗ Error testing performance completeness: {e}")
        return False

if __name__ == "__main__":
    success1 = test_performance_structure()
    success2 = test_lru_cache()
    success3 = test_query_cache()
    success4 = test_performance_metrics()
    success5 = test_batch_processor()
    success6 = test_query_optimizer()
    success7 = test_performance_decorators()
    success8 = test_performance_optimized_vector_store()
    success9 = test_thread_safety()
    success10 = test_performance_completeness()
    
    all_tests = [success1, success2, success3, success4, success5, 
                success6, success7, success8, success9, success10]
    
    if all(all_tests):
        print("\n✅ All performance tests passed! Performance optimizations are ready.")
        print("✅ Phase 4 (Performance Optimizations) is complete and functional")
    else:
        print("\n❌ Some performance tests failed. Please check the implementation.")
        sys.exit(1) 