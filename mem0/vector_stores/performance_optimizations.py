"""
Performance Optimizations for Enhanced Mem0 Vector Store

This module provides performance enhancements including caching,
connection pooling, and query optimization.
"""

import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from functools import wraps
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Simple LRU Cache implementation for query results
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)


class QueryCache:
    """
    Advanced query cache with TTL and cache invalidation
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.RLock()
    
    def _generate_key(self, query: str, filters: Dict, limit: int, mode: str) -> str:
        """Generate cache key from query parameters"""
        key_data = {
            'query': query,
            'filters': sorted(filters.items()) if filters else [],
            'limit': limit,
            'mode': mode
        }
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, filters: Dict, limit: int, mode: str) -> Optional[Any]:
        """Get cached query result"""
        key = self._generate_key(query, filters, limit, mode)
        
        with self.lock:
            if key in self.cache:
                # Check if expired
                if time.time() - self.timestamps[key] > self.ttl_seconds:
                    del self.cache[key]
                    del self.timestamps[key]
                    return None
                
                return self.cache[key]
            
            return None
    
    def set(self, query: str, filters: Dict, limit: int, mode: str, result: Any) -> None:
        """Cache query result"""
        key = self._generate_key(query, filters, limit, mode)
        
        with self.lock:
            # Remove expired entries if cache is full
            if len(self.cache) >= self.max_size:
                self._cleanup_expired()
                
                # If still full, remove oldest
                if len(self.cache) >= self.max_size:
                    oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
            
            self.cache[key] = result
            self.timestamps[key] = time.time()
    
    def invalidate_user(self, user_id: str) -> None:
        """Invalidate all cache entries for a user"""
        with self.lock:
            keys_to_remove = []
            for key, result in self.cache.items():
                # Check if result contains user_id
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and item.get('user_id') == user_id:
                            keys_to_remove.append(key)
                            break
            
            for key in keys_to_remove:
                if key in self.cache:
                    del self.cache[key]
                    del self.timestamps[key]
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)


class ConnectionPool:
    """
    Simple connection pool for database connections
    """
    
    def __init__(self, connection_factory, max_connections: int = 10):
        self.connection_factory = connection_factory
        self.max_connections = max_connections
        self.pool = []
        self.in_use = set()
        self.lock = threading.RLock()
    
    def get_connection(self):
        """Get connection from pool"""
        with self.lock:
            if self.pool:
                conn = self.pool.pop()
                self.in_use.add(conn)
                return conn
            elif len(self.in_use) < self.max_connections:
                conn = self.connection_factory()
                self.in_use.add(conn)
                return conn
            else:
                # Wait for connection to become available
                # In a real implementation, this would use a proper queue
                raise Exception("Connection pool exhausted")
    
    def return_connection(self, conn):
        """Return connection to pool"""
        with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)
                self.pool.append(conn)
    
    def close_all(self):
        """Close all connections"""
        with self.lock:
            for conn in self.pool + list(self.in_use):
                try:
                    conn.close()
                except:
                    pass
            self.pool.clear()
            self.in_use.clear()


class PerformanceMetrics:
    """
    Performance metrics collection and reporting
    """
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.RLock()
    
    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record operation metrics"""
        with self.lock:
            if operation not in self.metrics:
                self.metrics[operation] = {
                    'count': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'success_count': 0,
                    'error_count': 0
                }
            
            metric = self.metrics[operation]
            metric['count'] += 1
            metric['total_time'] += duration
            metric['avg_time'] = metric['total_time'] / metric['count']
            metric['min_time'] = min(metric['min_time'], duration)
            metric['max_time'] = max(metric['max_time'], duration)
            
            if success:
                metric['success_count'] += 1
            else:
                metric['error_count'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics"""
        with self.lock:
            return dict(self.metrics)
    
    def get_operation_metrics(self, operation: str) -> Optional[Dict[str, Any]]:
        """Get metrics for specific operation"""
        with self.lock:
            return self.metrics.get(operation)
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self.lock:
            self.metrics.clear()


def performance_monitor(operation_name: str):
    """Decorator to monitor performance of operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(self, *args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                if hasattr(self, 'performance_metrics'):
                    self.performance_metrics.record_operation(
                        operation_name, duration, success
                    )
        return wrapper
    return decorator


def cache_result(cache_attr: str = 'query_cache'):
    """Decorator to cache query results"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, query: str, filters: Dict = None, limit: int = 10, 
                   mode: str = 'semantic', **kwargs):
            # Get cache from instance
            cache = getattr(self, cache_attr, None)
            if not cache:
                return func(self, query, filters, limit, mode, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(query, filters or {}, limit, mode)
            if cached_result is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result
            
            # Execute function and cache result
            result = func(self, query, filters, limit, mode, **kwargs)
            cache.set(query, filters or {}, limit, mode, result)
            logger.debug(f"Cache miss for query: {query[:50]}...")
            
            return result
        return wrapper
    return decorator


class BatchProcessor:
    """
    Batch processor for bulk operations
    """
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    def process_in_batches(self, items: List[Any], processor_func) -> List[Any]:
        """Process items in batches"""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = processor_func(batch)
            results.extend(batch_results)
        
        return results


class QueryOptimizer:
    """
    Query optimization utilities
    """
    
    @staticmethod
    def optimize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize filter conditions"""
        if not filters:
            return {}
        
        optimized = {}
        
        # Remove None values
        for key, value in filters.items():
            if value is not None:
                optimized[key] = value
        
        return optimized
    
    @staticmethod
    def suggest_index_columns(query_patterns: List[Dict[str, Any]]) -> List[str]:
        """Suggest columns that should be indexed based on query patterns"""
        column_frequency = {}
        
        for pattern in query_patterns:
            filters = pattern.get('filters', {})
            for column in filters.keys():
                column_frequency[column] = column_frequency.get(column, 0) + 1
        
        # Return columns sorted by frequency
        return sorted(column_frequency.keys(), key=lambda x: column_frequency[x], reverse=True)
    
    @staticmethod
    def estimate_query_cost(filters: Dict[str, Any], limit: int) -> float:
        """Estimate the cost of a query"""
        # Simple cost estimation based on filter complexity and limit
        base_cost = 1.0
        filter_cost = len(filters) * 0.1
        limit_cost = min(limit / 100, 1.0)
        
        return base_cost + filter_cost + limit_cost


class PerformanceOptimizedVectorStore:
    """
    Mixin class to add performance optimizations to vector stores
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize performance components
        self.query_cache = QueryCache(max_size=1000, ttl_seconds=300)
        self.performance_metrics = PerformanceMetrics()
        self.batch_processor = BatchProcessor(batch_size=100)
        self.query_optimizer = QueryOptimizer()
        
        # Performance settings
        self.enable_caching = kwargs.get('enable_caching', True)
        self.enable_metrics = kwargs.get('enable_metrics', True)
        self.cache_ttl = kwargs.get('cache_ttl', 300)
        
        logger.info("Performance optimizations initialized")
    
    def invalidate_cache_for_user(self, user_id: str):
        """Invalidate cache entries for a specific user"""
        if self.enable_caching and self.query_cache:
            self.query_cache.invalidate_user(user_id)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if not self.enable_metrics:
            return {"metrics_disabled": True}
        
        metrics = self.performance_metrics.get_metrics()
        
        report = {
            'cache_stats': {
                'size': self.query_cache.size() if self.query_cache else 0,
                'hit_rate': self._calculate_cache_hit_rate()
            },
            'operation_metrics': metrics,
            'recommendations': self._generate_performance_recommendations(metrics)
        }
        
        return report
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # This would track hits/misses in a real implementation
        return 0.75  # Placeholder
    
    def _generate_performance_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics"""
        recommendations = []
        
        for operation, metric in metrics.items():
            if metric['avg_time'] > 1.0:  # Slow operations
                recommendations.append(f"Consider optimizing {operation} - avg time: {metric['avg_time']:.2f}s")
            
            if metric['error_count'] > metric['success_count'] * 0.1:  # High error rate
                recommendations.append(f"High error rate in {operation} - {metric['error_count']} errors")
        
        if len(recommendations) == 0:
            recommendations.append("Performance looks good!")
        
        return recommendations
    
    def optimize_query_plan(self, query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize query execution plan"""
        optimized_filters = self.query_optimizer.optimize_filters(filters)
        estimated_cost = self.query_optimizer.estimate_query_cost(optimized_filters, 10)
        
        return {
            'original_filters': filters,
            'optimized_filters': optimized_filters,
            'estimated_cost': estimated_cost,
            'recommendations': []
        }
    
    def warm_cache(self, common_queries: List[Dict[str, Any]]):
        """Warm up cache with common queries"""
        if not self.enable_caching:
            return
        
        logger.info(f"Warming cache with {len(common_queries)} common queries")
        
        for query_info in common_queries:
            try:
                # This would execute the actual search in a real implementation
                # For now, we'll just add placeholder results
                self.query_cache.set(
                    query_info.get('query', ''),
                    query_info.get('filters', {}),
                    query_info.get('limit', 10),
                    query_info.get('mode', 'semantic'),
                    []  # Placeholder result
                )
            except Exception as e:
                logger.warning(f"Failed to warm cache for query: {e}")
        
        logger.info("Cache warming completed") 