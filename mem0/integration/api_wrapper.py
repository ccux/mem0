"""
API Wrapper for Enhanced Mem0 Integration

This module provides a simplified API interface for the Cognition Suite
to interact with the enhanced Mem0 functionality.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
from functools import wraps

from mem0.integration.cognition_suite_adapter import CognitionSuiteMemoryAdapter
from mem0.configs.base import MemoryConfig

logger = logging.getLogger(__name__)


def handle_errors(func):
    """Decorator to handle errors consistently across API methods"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.utcnow().isoformat()
            }
    return wrapper


class EnhancedMem0API:
    """
    Simplified API wrapper for enhanced Mem0 functionality
    
    This class provides a clean, easy-to-use interface for the Cognition Suite
    to interact with all enhanced memory capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced Mem0 API
        
        Args:
            config: Optional configuration dictionary
        """
        try:
            # Convert config dict to MemoryConfig if provided
            if config:
                memory_config = MemoryConfig(**config)
            else:
                memory_config = MemoryConfig()
            
            self.adapter = CognitionSuiteMemoryAdapter(memory_config)
            self.version = "1.0.0"
            
            logger.info("EnhancedMem0API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedMem0API: {e}")
            raise

    @handle_errors
    def add_memory(
        self,
        content: str,
        user_id: str,
        source: str = 'manual',
        document_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a new memory
        
        Args:
            content: The memory content
            user_id: User identifier
            source: Source of the memory (manual, chat, document, auto-extraction, system)
            document_id: Optional document ID
            chat_id: Optional chat ID
            context: Optional context information
            metadata: Optional additional metadata
            
        Returns:
            Dict with success status and memory details
        """
        result = self.adapter.add_memory_with_context(
            content=content,
            user_id=user_id,
            source=source,
            document_id=document_id,
            chat_id=chat_id,
            context=context,
            metadata=metadata
        )
        
        return {
            'success': True,
            'data': result,
            'timestamp': datetime.utcnow().isoformat()
        }

    @handle_errors
    def search_memories(
        self,
        query: str,
        user_id: str,
        mode: str = 'hybrid',
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_analytics: bool = False
    ) -> Dict[str, Any]:
        """
        Search memories with enhanced capabilities
        
        Args:
            query: Search query
            user_id: User identifier
            mode: Search mode (semantic, text, hybrid, fallback)
            limit: Maximum number of results
            filters: Optional filters
            include_analytics: Whether to include search analytics
            
        Returns:
            Dict with search results and metadata
        """
        result = self.adapter.search_memories_enhanced(
            query=query,
            user_id=user_id,
            search_mode=mode,
            limit=limit,
            filters=filters,
            include_analytics=include_analytics
        )
        
        return {
            'success': True,
            'data': result,
            'timestamp': datetime.utcnow().isoformat()
        }

    @handle_errors
    def get_user_dashboard(
        self,
        user_id: str,
        include_insights: bool = True,
        include_clusters: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive user memory dashboard
        
        Args:
            user_id: User identifier
            include_insights: Whether to include memory insights
            include_clusters: Whether to include memory clusters
            
        Returns:
            Dict with dashboard data
        """
        result = self.adapter.get_user_memory_dashboard(
            user_id=user_id,
            include_insights=include_insights,
            include_clusters=include_clusters
        )
        
        return {
            'success': True,
            'data': result,
            'timestamp': datetime.utcnow().isoformat()
        }

    @handle_errors
    def get_document_analysis(
        self,
        document_id: str,
        user_id: str,
        include_similar: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive document memory analysis
        
        Args:
            document_id: Document identifier
            user_id: User identifier
            include_similar: Whether to include similar memories
            
        Returns:
            Dict with document analysis
        """
        result = self.adapter.get_document_memory_analysis(
            document_id=document_id,
            user_id=user_id,
            include_similar=include_similar
        )
        
        return {
            'success': True,
            'data': result,
            'timestamp': datetime.utcnow().isoformat()
        }

    @handle_errors
    def bulk_operations(
        self,
        operations: List[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Execute bulk memory operations
        
        Args:
            operations: List of operations to execute
            user_id: User identifier
            
        Returns:
            Dict with operation results
        """
        result = self.adapter.bulk_memory_management(
            operations=operations,
            user_id=user_id
        )
        
        return {
            'success': True,
            'data': result,
            'timestamp': datetime.utcnow().isoformat()
        }

    @handle_errors
    def get_recommendations(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get personalized memory recommendations
        
        Args:
            user_id: User identifier
            context: Optional context information
            
        Returns:
            Dict with recommendations
        """
        result = self.adapter.get_memory_recommendations(
            user_id=user_id,
            context=context
        )
        
        return {
            'success': True,
            'data': result,
            'timestamp': datetime.utcnow().isoformat()
        }

    @handle_errors
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get basic memory statistics for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with memory statistics
        """
        result = self.adapter.enhanced_memory.get_memory_stats(user_id=user_id)
        
        return {
            'success': True,
            'data': result,
            'timestamp': datetime.utcnow().isoformat()
        }

    @handle_errors
    def count_memories(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Count memories for a user
        
        Args:
            user_id: User identifier
            filters: Optional filters
            
        Returns:
            Dict with memory count
        """
        count = self.adapter.enhanced_memory.count_user_memories(
            user_id=user_id,
            filters=filters
        )
        
        return {
            'success': True,
            'data': {'count': count, 'user_id': user_id},
            'timestamp': datetime.utcnow().isoformat()
        }

    @handle_errors
    def find_similar_memories(
        self,
        memory_id: str,
        user_id: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Find memories similar to a given memory
        
        Args:
            memory_id: Reference memory ID
            user_id: User identifier
            limit: Maximum number of similar memories
            
        Returns:
            Dict with similar memories
        """
        result = self.adapter.enhanced_memory.search_similar_memories(
            memory_id=memory_id,
            user_id=user_id,
            limit=limit
        )
        
        return {
            'success': True,
            'data': {
                'similar_memories': result,
                'reference_memory_id': memory_id,
                'user_id': user_id
            },
            'timestamp': datetime.utcnow().isoformat()
        }

    @handle_errors
    def get_memory_clusters(
        self,
        user_id: str,
        num_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        Get memory clusters for topic analysis
        
        Args:
            user_id: User identifier
            num_clusters: Number of clusters to generate
            
        Returns:
            Dict with memory clusters
        """
        result = self.adapter.enhanced_memory.get_memory_clusters(
            user_id=user_id,
            num_clusters=num_clusters
        )
        
        return {
            'success': True,
            'data': result,
            'timestamp': datetime.utcnow().isoformat()
        }

    @handle_errors
    def get_latest_memories(
        self,
        user_id: str,
        limit: int = 10,
        source_priority: bool = True
    ) -> Dict[str, Any]:
        """
        Get latest memories for a user
        
        Args:
            user_id: User identifier
            limit: Maximum number of memories
            source_priority: Whether to prioritize by source
            
        Returns:
            Dict with latest memories
        """
        result = self.adapter.enhanced_memory.get_latest_memories(
            user_id=user_id,
            limit=limit,
            source_priority=source_priority
        )
        
        return {
            'success': True,
            'data': {
                'memories': result,
                'user_id': user_id,
                'limit': limit
            },
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_api_info(self) -> Dict[str, Any]:
        """
        Get API information and capabilities
        
        Returns:
            Dict with API information
        """
        return {
            'api_version': self.version,
            'enhanced_features': [
                'hybrid_search',
                'semantic_fallback',
                'memory_analytics',
                'memory_insights',
                'memory_clustering',
                'bulk_operations',
                'personalized_recommendations',
                'document_analysis',
                'user_dashboard'
            ],
            'search_modes': ['semantic', 'text', 'hybrid', 'fallback'],
            'supported_sources': ['manual', 'chat', 'document', 'auto-extraction', 'system'],
            'timestamp': datetime.utcnow().isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the API
        
        Returns:
            Dict with health status
        """
        try:
            # Try to access the adapter
            adapter_status = "healthy" if self.adapter else "unhealthy"
            
            # Try to access the enhanced memory
            memory_status = "healthy" if self.adapter.enhanced_memory else "unhealthy"
            
            overall_status = "healthy" if adapter_status == "healthy" and memory_status == "healthy" else "unhealthy"
            
            return {
                'status': overall_status,
                'components': {
                    'adapter': adapter_status,
                    'enhanced_memory': memory_status
                },
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }


# Convenience functions for common operations
def create_api(config: Optional[Dict[str, Any]] = None) -> EnhancedMem0API:
    """
    Create and return an EnhancedMem0API instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        EnhancedMem0API instance
    """
    return EnhancedMem0API(config)


def quick_search(
    query: str,
    user_id: str,
    api: Optional[EnhancedMem0API] = None,
    mode: str = 'hybrid'
) -> Dict[str, Any]:
    """
    Quick search function for simple use cases
    
    Args:
        query: Search query
        user_id: User identifier
        api: Optional API instance (will create one if not provided)
        mode: Search mode
        
    Returns:
        Search results
    """
    if api is None:
        api = create_api()
    
    return api.search_memories(query, user_id, mode=mode)


def quick_add(
    content: str,
    user_id: str,
    source: str = 'manual',
    api: Optional[EnhancedMem0API] = None
) -> Dict[str, Any]:
    """
    Quick add function for simple use cases
    
    Args:
        content: Memory content
        user_id: User identifier
        source: Memory source
        api: Optional API instance (will create one if not provided)
        
    Returns:
        Add result
    """
    if api is None:
        api = create_api()
    
    return api.add_memory(content, user_id, source=source) 