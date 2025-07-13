import logging
from typing import Dict, List, Optional, Any, Tuple
from mem0.memory.main import Memory
from mem0.vector_stores.base import SearchMode, SortOrder

logger = logging.getLogger(__name__)


class EnhancedMemory(Memory):
    """Enhanced Memory class with advanced querying capabilities"""

    def search_with_fallback(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Search with automatic fallback to text search
        """
        # Build filters for search
        search_filters = self._build_search_filters(user_id, agent_id, run_id, filters)
        
        # Try semantic search first
        try:
            results = self.vector_store.advanced_search(
                query,
                search_filters,
                mode=SearchMode.SEMANTIC,
                limit=limit,
                **kwargs
            )
            
            # If no results, fallback to text search
            if not results:
                logger.info(f"Semantic search returned no results for '{query}', falling back to text search")
                results = self.vector_store.advanced_search(
                    query,
                    search_filters,
                    mode=SearchMode.TEXT,
                    limit=limit,
                    **kwargs
                )
            
            return self._format_search_results(results)
            
        except Exception as e:
            logger.error(f"Error in search_with_fallback: {e}")
            # Fallback to original search method
            return self.search(query, user_id, agent_id, run_id, limit, filters)

    def hybrid_search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Hybrid search combining semantic and text search
        """
        # Build filters for search
        search_filters = self._build_search_filters(user_id, agent_id, run_id, filters)
        
        try:
            results = self.vector_store.advanced_search(
                query,
                search_filters,
                mode=SearchMode.HYBRID,
                limit=limit,
                **kwargs
            )
            
            return self._format_search_results(results)
            
        except Exception as e:
            logger.error(f"Error in hybrid_search: {e}")
            # Fallback to original search method
            return self.search(query, user_id, agent_id, run_id, limit, filters)

    def get_memory_stats(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            # For now, we'll use user_id as the primary identifier
            # In a full implementation, we'd combine all filters
            stats_user_id = user_id or agent_id or run_id
            
            if not stats_user_id:
                raise ValueError("At least one of user_id, agent_id, or run_id must be provided")
            
            return self.vector_store.aggregate_stats(stats_user_id)
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            # Fallback to basic count
            filters = self._build_search_filters(user_id, agent_id, run_id)
            count = self.vector_store.count_memories(filters)
            return {
                'total_memories': count,
                'method': 'fallback_count'
            }

    def get_memories_with_complex_sorting(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        sort_criteria: Optional[List[Tuple[str, SortOrder]]] = None,
        filters: Optional[Dict] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get memories with complex sorting logic"""
        try:
            # Combine filters
            search_filters = self._build_search_filters(user_id, agent_id, run_id, filters)
            
            results = self.vector_store.list_with_sorting(
                search_filters,
                sort_criteria,
                limit,
                offset
            )
            
            return self._format_memory_results(results)
            
        except Exception as e:
            logger.error(f"Error in get_memories_with_complex_sorting: {e}")
            # Fallback to get_all
            return self.get_all(user_id, agent_id, run_id, limit)

    def count_user_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> int:
        """Count memories for a user with optional filters"""
        try:
            search_filters = self._build_search_filters(user_id, agent_id, run_id, filters)
            return self.vector_store.count_memories(search_filters)
            
        except Exception as e:
            logger.error(f"Error counting memories: {e}")
            # Fallback to length of get_all
            memories = self.get_all(user_id, agent_id, run_id, limit=10000)
            return len(memories)

    def get_document_memories(
        self,
        document_id: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get all memories for a specific document"""
        try:
            search_filters = self._build_search_filters(user_id, agent_id, run_id)
            search_filters["document_id"] = document_id
            
            results = self.vector_store.list_with_sorting(
                search_filters,
                sort_by=[("created_at", SortOrder.DESC)],
                limit=limit
            )
            
            return self._format_memory_results(results)
            
        except Exception as e:
            logger.error(f"Error getting document memories: {e}")
            # Fallback to search
            return self.search(f"document:{document_id}", user_id, agent_id, run_id, limit)

    def search_with_analytics(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        aggregate_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Search with metadata aggregation for analytics"""
        try:
            search_filters = self._build_search_filters(user_id, agent_id, run_id)
            
            results = self.vector_store.search_with_metadata_aggregation(
                query,
                search_filters,
                aggregate_fields
            )
            
            # Format results
            formatted_results = self._format_search_results(results.get('results', []))
            
            return {
                'results': formatted_results,
                'aggregations': results.get('aggregations', {}),
                'total_count': results.get('total_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in search_with_analytics: {e}")
            # Fallback to regular search
            search_results = self.search(query, user_id, agent_id, run_id)
            return {
                'results': search_results,
                'aggregations': {},
                'total_count': len(search_results)
            }

    def get_latest_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 10,
        source_priority: bool = True
    ) -> List[Dict]:
        """Get latest memories with optional source prioritization"""
        try:
            sort_criteria = []
            if source_priority:
                sort_criteria.append(("source_priority", SortOrder.ASC))

            sort_criteria.extend([
                ("created_at", SortOrder.DESC),
                ("updated_at", SortOrder.DESC)
            ])

            return self.get_memories_with_complex_sorting(
                user_id,
                agent_id,
                run_id,
                sort_criteria,
                limit=limit
            )
            
        except Exception as e:
            logger.error(f"Error getting latest memories: {e}")
            # Fallback to get_all
            return self.get_all(user_id, agent_id, run_id, limit)

    def bulk_memory_operations(self, operations: List[Dict]) -> List[Dict]:
        """Execute multiple memory operations in a single transaction"""
        try:
            return self.vector_store.bulk_operations(operations)
            
        except Exception as e:
            logger.error(f"Error in bulk_memory_operations: {e}")
            # Fallback to individual operations
            results = []
            for operation in operations:
                try:
                    op_type = operation.get('type')
                    if op_type == 'add':
                        result = self.add(**operation.get('params', {}))
                        results.append({'type': 'add', 'result': result})
                    elif op_type == 'update':
                        result = self.update(operation['memory_id'], operation['data'])
                        results.append({'type': 'update', 'result': result})
                    elif op_type == 'delete':
                        result = self.delete(operation['memory_id'])
                        results.append({'type': 'delete', 'result': result})
                    else:
                        results.append({'type': 'error', 'message': f'Unknown operation: {op_type}'})
                except Exception as op_error:
                    results.append({'type': 'error', 'message': str(op_error)})
            
            return results

    def get_memory_analytics(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        time_range: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive memory analytics"""
        try:
            # Get basic stats
            stats = self.get_memory_stats(user_id, agent_id, run_id)
            
            # Get additional analytics
            search_filters = self._build_search_filters(user_id, agent_id, run_id)
            
            # Add time range filters if provided
            if time_range:
                if 'start_date' in time_range:
                    search_filters['created_after'] = time_range['start_date']
                if 'end_date' in time_range:
                    search_filters['created_before'] = time_range['end_date']
            
            # Get aggregations for different fields
            analytics = {
                'basic_stats': stats,
                'source_distribution': self._get_field_aggregation('source', search_filters),
                'memory_trends': self._get_memory_trends(search_filters),
                'top_keywords': self._get_top_keywords(search_filters),
                'memory_length_stats': self._get_memory_length_stats(search_filters)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting memory analytics: {e}")
            return {
                'basic_stats': self.get_memory_stats(user_id, agent_id, run_id),
                'source_distribution': {},
                'memory_trends': {},
                'top_keywords': [],
                'memory_length_stats': {}
            }

    def get_memory_insights(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get memory insights and patterns"""
        try:
            analytics = self.get_memory_analytics(user_id, agent_id, run_id)
            
            # Generate insights based on analytics
            insights = {
                'total_memories': analytics['basic_stats'].get('total_memories', 0),
                'most_active_source': self._get_most_active_source(analytics['source_distribution']),
                'memory_growth_trend': self._analyze_growth_trend(analytics['memory_trends']),
                'average_memory_length': analytics['basic_stats'].get('avg_memory_length', 0),
                'recommendations': self._generate_recommendations(analytics)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting memory insights: {e}")
            return {
                'total_memories': 0,
                'most_active_source': 'unknown',
                'memory_growth_trend': 'stable',
                'average_memory_length': 0,
                'recommendations': []
            }

    def search_similar_memories(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """Find memories similar to a given memory"""
        try:
            # Get the reference memory
            reference_memory = self.get(memory_id)
            if not reference_memory:
                return []
            
            # Use the memory content to search for similar memories
            query = reference_memory.get('memory', '')
            
            # Search for similar memories, excluding the reference memory
            results = self.search_with_fallback(
                query,
                user_id,
                agent_id,
                run_id,
                limit=limit + 1  # Get one extra to exclude the reference
            )
            
            # Filter out the reference memory
            similar_memories = [
                memory for memory in results 
                if memory.get('id') != memory_id
            ]
            
            return similar_memories[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []

    def get_memory_clusters(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        num_clusters: int = 5
    ) -> Dict[str, Any]:
        """Get memory clusters for topic analysis"""
        try:
            # This is a placeholder for clustering functionality
            # In a full implementation, this would use ML clustering algorithms
            
            memories = self.get_all(user_id, agent_id, run_id, limit=1000)
            
            # Simple clustering based on keywords (placeholder)
            clusters = self._simple_keyword_clustering(memories, num_clusters)
            
            return {
                'clusters': clusters,
                'total_memories': len(memories),
                'clustering_method': 'keyword_based'
            }
            
        except Exception as e:
            logger.error(f"Error getting memory clusters: {e}")
            return {
                'clusters': [],
                'total_memories': 0,
                'clustering_method': 'error'
            }

    def _get_field_aggregation(self, field: str, filters: Dict) -> Dict[str, int]:
        """Get aggregation for a specific field"""
        try:
            # This would use the vector store aggregation in a full implementation
            # For now, return a placeholder
            return {'manual': 10, 'auto-extraction': 25, 'document': 15}
        except Exception as e:
            logger.error(f"Error getting field aggregation for {field}: {e}")
            return {}

    def _get_memory_trends(self, filters: Dict) -> Dict[str, Any]:
        """Get memory creation trends over time"""
        try:
            # This would analyze memory creation patterns over time
            # For now, return a placeholder
            return {
                'daily_counts': {'2024-01-01': 5, '2024-01-02': 8, '2024-01-03': 12},
                'trend_direction': 'increasing'
            }
        except Exception as e:
            logger.error(f"Error getting memory trends: {e}")
            return {}

    def _get_top_keywords(self, filters: Dict) -> List[Dict[str, Any]]:
        """Get top keywords from memories"""
        try:
            # This would analyze memory content for common keywords
            # For now, return a placeholder
            return [
                {'keyword': 'project', 'count': 15},
                {'keyword': 'meeting', 'count': 12},
                {'keyword': 'deadline', 'count': 8}
            ]
        except Exception as e:
            logger.error(f"Error getting top keywords: {e}")
            return []

    def _get_memory_length_stats(self, filters: Dict) -> Dict[str, float]:
        """Get memory length statistics"""
        try:
            # This would analyze memory length patterns
            # For now, return a placeholder
            return {
                'average_length': 150.5,
                'median_length': 120.0,
                'max_length': 500,
                'min_length': 10
            }
        except Exception as e:
            logger.error(f"Error getting memory length stats: {e}")
            return {}

    def _get_most_active_source(self, source_distribution: Dict) -> str:
        """Get the most active memory source"""
        if not source_distribution:
            return 'unknown'
        
        return max(source_distribution, key=source_distribution.get)

    def _analyze_growth_trend(self, trends: Dict) -> str:
        """Analyze memory growth trend"""
        try:
            daily_counts = trends.get('daily_counts', {})
            if len(daily_counts) < 2:
                return 'stable'
            
            values = list(daily_counts.values())
            if values[-1] > values[0]:
                return 'increasing'
            elif values[-1] < values[0]:
                return 'decreasing'
            else:
                return 'stable'
        except Exception:
            return 'stable'

    def _generate_recommendations(self, analytics: Dict) -> List[str]:
        """Generate recommendations based on analytics"""
        recommendations = []
        
        try:
            total_memories = analytics['basic_stats'].get('total_memories', 0)
            
            if total_memories < 10:
                recommendations.append("Consider adding more memories to improve search accuracy")
            
            source_dist = analytics.get('source_distribution', {})
            if source_dist.get('manual', 0) < source_dist.get('auto-extraction', 0):
                recommendations.append("Consider adding more manual memories for better personalization")
            
            avg_length = analytics['basic_stats'].get('avg_memory_length', 0)
            if avg_length < 50:
                recommendations.append("Consider adding more detailed memories for better context")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations"]

    def _simple_keyword_clustering(self, memories: List[Dict], num_clusters: int) -> List[Dict]:
        """Simple keyword-based clustering (placeholder)"""
        try:
            # This is a very basic clustering implementation
            # In a full implementation, this would use proper ML clustering
            
            clusters = []
            for i in range(min(num_clusters, len(memories))):
                cluster = {
                    'id': f'cluster_{i}',
                    'name': f'Topic {i+1}',
                    'memories': memories[i:i+1],  # Just one memory per cluster for now
                    'keywords': ['placeholder', 'keyword'],
                    'size': 1
                }
                clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error in simple keyword clustering: {e}")
            return []

    def _build_search_filters(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        additional_filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Build search filters from parameters"""
        filters = {}
        
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id
            
        if additional_filters:
            filters.update(additional_filters)
            
        return filters

    def _format_search_results(self, results: List[Dict]) -> List[Dict]:
        """Format search results to match expected Memory format"""
        formatted_results = []
        
        for result in results:
            # Extract payload and format it
            payload = result.get('payload', {})
            
            formatted_result = {
                'id': result.get('id'),
                'memory': payload.get('memory', ''),
                'user_id': payload.get('user_id'),
                'agent_id': payload.get('agent_id'),
                'run_id': payload.get('run_id'),
                'created_at': payload.get('created_at'),
                'updated_at': payload.get('updated_at'),
                'metadata': payload.get('metadata', {}),
                'score': result.get('score', 0.0),
                'search_mode': result.get('search_mode', 'unknown')
            }
            
            formatted_results.append(formatted_result)
            
        return formatted_results

    def _format_memory_results(self, results: List[Dict]) -> List[Dict]:
        """Format memory results to match expected Memory format"""
        formatted_results = []
        
        for result in results:
            # Extract payload and format it
            payload = result.get('payload', {})
            
            formatted_result = {
                'id': result.get('id'),
                'memory': payload.get('memory', ''),
                'user_id': payload.get('user_id'),
                'agent_id': payload.get('agent_id'),
                'run_id': payload.get('run_id'),
                'created_at': payload.get('created_at'),
                'updated_at': payload.get('updated_at'),
                'metadata': payload.get('metadata', {})
            }
            
            formatted_results.append(formatted_result)
            
        return formatted_results 