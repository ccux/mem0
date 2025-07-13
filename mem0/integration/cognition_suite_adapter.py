"""
Cognition Suite Integration Adapter for Enhanced Mem0

This module provides a bridge between the enhanced Mem0 components
and the Cognition Suite's existing memory infrastructure.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

from mem0.memory.enhanced_memory import EnhancedMemory
from mem0.vector_stores.base import SearchMode, SortOrder
from mem0.configs.base import MemoryConfig

logger = logging.getLogger(__name__)


class CognitionSuiteMemoryAdapter:
    """
    Adapter class that integrates enhanced Mem0 functionality 
    with the Cognition Suite's memory system
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the adapter with enhanced memory capabilities"""
        self.config = config or MemoryConfig()
        self.enhanced_memory = EnhancedMemory(self.config)
        
        # Cognition Suite specific mappings
        self.source_priority_map = {
            'manual': 1,           # Highest priority - user-created
            'chat': 2,             # High priority - chat interactions
            'document': 3,         # Medium priority - document extraction
            'auto-extraction': 4,  # Lower priority - automatic extraction
            'system': 5           # Lowest priority - system generated
        }
        
        logger.info("CognitionSuiteMemoryAdapter initialized with enhanced capabilities")

    def add_memory_with_context(
        self,
        content: str,
        user_id: str,
        source: str = 'manual',
        context: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a memory with Cognition Suite context
        
        Args:
            content: Memory content
            user_id: User identifier
            source: Memory source (manual, chat, document, auto-extraction, system)
            context: Additional context information
            document_id: Associated document ID
            chat_id: Associated chat ID
            metadata: Additional metadata
            
        Returns:
            Dict containing memory ID and creation details
        """
        try:
            # Build enhanced metadata
            enhanced_metadata = {
                'source': source,
                'source_priority': self.source_priority_map.get(source, 5),
                'created_at': datetime.utcnow().isoformat(),
                'context': context or {},
                'integration_version': '1.0.0'
            }
            
            if document_id:
                enhanced_metadata['document_id'] = document_id
            if chat_id:
                enhanced_metadata['chat_id'] = chat_id
            if metadata:
                enhanced_metadata.update(metadata)
            
            # Add memory using enhanced memory system
            result = self.enhanced_memory.add(
                messages=content,
                user_id=user_id,
                metadata=enhanced_metadata
            )
            
            logger.info(f"Memory added for user {user_id} from source {source}")
            return {
                'memory_id': result,
                'user_id': user_id,
                'source': source,
                'created_at': enhanced_metadata['created_at']
            }
            
        except Exception as e:
            logger.error(f"Error adding memory with context: {e}")
            raise

    def search_memories_enhanced(
        self,
        query: str,
        user_id: str,
        search_mode: str = 'hybrid',
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_analytics: bool = False
    ) -> Dict[str, Any]:
        """
        Enhanced memory search with Cognition Suite optimizations
        
        Args:
            query: Search query
            user_id: User identifier
            search_mode: 'semantic', 'text', 'hybrid', or 'fallback'
            limit: Maximum number of results
            filters: Additional search filters
            include_analytics: Whether to include search analytics
            
        Returns:
            Dict containing search results and optional analytics
        """
        try:
            # Choose search method based on mode
            if search_mode == 'hybrid':
                results = self.enhanced_memory.hybrid_search(
                    query, user_id=user_id, limit=limit, filters=filters
                )
            elif search_mode == 'fallback':
                results = self.enhanced_memory.search_with_fallback(
                    query, user_id=user_id, limit=limit, filters=filters
                )
            else:
                # Use the base search for semantic/text modes
                results = self.enhanced_memory.search(
                    query, user_id=user_id, limit=limit, filters=filters
                )
            
            # Format results for Cognition Suite
            formatted_results = self._format_search_results_for_cognition_suite(results)
            
            response = {
                'results': formatted_results,
                'total_count': len(formatted_results),
                'search_mode': search_mode,
                'query': query
            }
            
            # Add analytics if requested
            if include_analytics:
                response['analytics'] = self.enhanced_memory.search_with_analytics(
                    query, user_id=user_id
                )
            
            logger.info(f"Enhanced search completed for user {user_id}, mode: {search_mode}")
            return response
            
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            raise

    def get_user_memory_dashboard(
        self,
        user_id: str,
        include_insights: bool = True,
        include_clusters: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive memory dashboard for a user
        
        Args:
            user_id: User identifier
            include_insights: Whether to include memory insights
            include_clusters: Whether to include memory clusters
            
        Returns:
            Dict containing dashboard data
        """
        try:
            dashboard = {
                'user_id': user_id,
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Get basic statistics
            dashboard['stats'] = self.enhanced_memory.get_memory_stats(user_id=user_id)
            
            # Get recent memories
            dashboard['recent_memories'] = self.enhanced_memory.get_latest_memories(
                user_id=user_id, limit=10
            )
            
            # Get memory count
            dashboard['total_memories'] = self.enhanced_memory.count_user_memories(
                user_id=user_id
            )
            
            # Add insights if requested
            if include_insights:
                dashboard['insights'] = self.enhanced_memory.get_memory_insights(
                    user_id=user_id
                )
            
            # Add clusters if requested
            if include_clusters:
                dashboard['clusters'] = self.enhanced_memory.get_memory_clusters(
                    user_id=user_id, num_clusters=5
                )
            
            logger.info(f"Memory dashboard generated for user {user_id}")
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating memory dashboard: {e}")
            raise

    def get_document_memory_analysis(
        self,
        document_id: str,
        user_id: str,
        include_similar: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive analysis of memories related to a document
        
        Args:
            document_id: Document identifier
            user_id: User identifier
            include_similar: Whether to include similar memories
            
        Returns:
            Dict containing document memory analysis
        """
        try:
            analysis = {
                'document_id': document_id,
                'user_id': user_id,
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Get document memories
            document_memories = self.enhanced_memory.get_document_memories(
                document_id, user_id=user_id
            )
            
            analysis['document_memories'] = document_memories
            analysis['memory_count'] = len(document_memories)
            
            # Get analytics for document memories
            if document_memories:
                filters = {'document_id': document_id}
                analysis['analytics'] = self.enhanced_memory.get_memory_analytics(
                    user_id=user_id
                )
                
                # Find similar memories if requested
                if include_similar and document_memories:
                    first_memory = document_memories[0]
                    analysis['similar_memories'] = self.enhanced_memory.search_similar_memories(
                        first_memory['id'], user_id=user_id, limit=5
                    )
            
            logger.info(f"Document memory analysis completed for document {document_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing document memories: {e}")
            raise

    def bulk_memory_management(
        self,
        operations: List[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Execute bulk memory operations for Cognition Suite
        
        Args:
            operations: List of memory operations
            user_id: User identifier
            
        Returns:
            Dict containing operation results
        """
        try:
            # Add user_id to all operations that need it
            enhanced_operations = []
            for operation in operations:
                enhanced_op = operation.copy()
                if 'params' in enhanced_op:
                    enhanced_op['params']['user_id'] = user_id
                enhanced_operations.append(enhanced_op)
            
            # Execute bulk operations
            results = self.enhanced_memory.bulk_memory_operations(enhanced_operations)
            
            # Analyze results
            success_count = len([r for r in results if r.get('type') != 'error'])
            error_count = len([r for r in results if r.get('type') == 'error'])
            
            response = {
                'user_id': user_id,
                'total_operations': len(operations),
                'successful_operations': success_count,
                'failed_operations': error_count,
                'results': results,
                'executed_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Bulk operations completed for user {user_id}: {success_count} success, {error_count} errors")
            return response
            
        except Exception as e:
            logger.error(f"Error in bulk memory management: {e}")
            raise

    def get_memory_recommendations(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get personalized memory recommendations for a user
        
        Args:
            user_id: User identifier
            context: Current context (e.g., current document, chat)
            
        Returns:
            Dict containing recommendations
        """
        try:
            # Get memory insights
            insights = self.enhanced_memory.get_memory_insights(user_id=user_id)
            
            # Get analytics
            analytics = self.enhanced_memory.get_memory_analytics(user_id=user_id)
            
            recommendations = {
                'user_id': user_id,
                'generated_at': datetime.utcnow().isoformat(),
                'context': context or {},
                'general_recommendations': insights.get('recommendations', []),
                'memory_optimization': self._generate_memory_optimization_tips(analytics),
                'search_suggestions': self._generate_search_suggestions(analytics),
                'content_suggestions': self._generate_content_suggestions(insights, context)
            }
            
            logger.info(f"Memory recommendations generated for user {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating memory recommendations: {e}")
            raise

    def _format_search_results_for_cognition_suite(
        self, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format search results for Cognition Suite compatibility"""
        formatted_results = []
        
        for result in results:
            formatted_result = {
                'id': result.get('id'),
                'content': result.get('memory', ''),
                'user_id': result.get('user_id'),
                'source': result.get('metadata', {}).get('source', 'unknown'),
                'created_at': result.get('created_at'),
                'updated_at': result.get('updated_at'),
                'score': result.get('score', 0.0),
                'search_mode': result.get('search_mode', 'unknown'),
                'metadata': result.get('metadata', {}),
                'context': result.get('metadata', {}).get('context', {})
            }
            
            # Add document and chat IDs if available
            metadata = result.get('metadata', {})
            if 'document_id' in metadata:
                formatted_result['document_id'] = metadata['document_id']
            if 'chat_id' in metadata:
                formatted_result['chat_id'] = metadata['chat_id']
            
            formatted_results.append(formatted_result)
        
        return formatted_results

    def _generate_memory_optimization_tips(
        self, 
        analytics: Dict[str, Any]
    ) -> List[str]:
        """Generate memory optimization tips based on analytics"""
        tips = []
        
        try:
            stats = analytics.get('basic_stats', {})
            total_memories = stats.get('total_memories', 0)
            avg_length = stats.get('avg_memory_length', 0)
            
            if total_memories < 20:
                tips.append("Add more memories to improve search accuracy and personalization")
            
            if avg_length < 100:
                tips.append("Consider adding more detailed memories for better context")
            
            source_dist = analytics.get('source_distribution', {})
            manual_count = source_dist.get('manual', 0)
            auto_count = source_dist.get('auto-extraction', 0)
            
            if manual_count < auto_count * 0.3:
                tips.append("Add more manual memories to improve personalization")
            
            return tips
            
        except Exception as e:
            logger.error(f"Error generating optimization tips: {e}")
            return ["Unable to generate optimization tips at this time"]

    def _generate_search_suggestions(
        self, 
        analytics: Dict[str, Any]
    ) -> List[str]:
        """Generate search suggestions based on analytics"""
        suggestions = []
        
        try:
            top_keywords = analytics.get('top_keywords', [])
            
            if top_keywords:
                suggestions.append(f"Try searching for: {', '.join([kw['keyword'] for kw in top_keywords[:3]])}")
            
            suggestions.extend([
                "Use hybrid search for best results",
                "Try searching for recent topics or projects",
                "Search for specific document names or types"
            ])
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating search suggestions: {e}")
            return ["Use the search function to find relevant memories"]

    def _generate_content_suggestions(
        self, 
        insights: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate content suggestions based on insights and context"""
        suggestions = []
        
        try:
            total_memories = insights.get('total_memories', 0)
            
            if total_memories == 0:
                suggestions.append("Start by adding your first memory about current projects or goals")
            elif total_memories < 5:
                suggestions.append("Add memories about your recent work, meetings, or important decisions")
            else:
                suggestions.append("Consider adding memories about lessons learned or future plans")
            
            # Context-based suggestions
            if context:
                if context.get('document_type') == 'meeting_notes':
                    suggestions.append("Add key decisions and action items from this meeting")
                elif context.get('document_type') == 'project_doc':
                    suggestions.append("Add project milestones and important requirements")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating content suggestions: {e}")
            return ["Add memories about important information you want to remember"] 