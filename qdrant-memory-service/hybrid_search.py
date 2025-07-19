"""
Hybrid Search Service for combining multiple search modes.

This module provides semantic, text-based, and graph-enhanced search capabilities
for memory retrieval with configurable search modes and weighting.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from qdrant_memory_client import QdrantMemoryClient
from neo4j_graph_client import Neo4jGraphClient
from graph_extractor import GraphExtractor
from models import MemoryResponse, SearchMode, HybridSearchRequest

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Individual search result with metadata"""
    memory: MemoryResponse
    relevance_score: float
    search_type: str  # 'semantic', 'text', 'graph'
    match_reason: str

class HybridSearchService:
    """Hybrid search service combining multiple search strategies"""

    def __init__(self, qdrant_client: QdrantMemoryClient, graph_client: Neo4jGraphClient,
                 graph_extractor: GraphExtractor):
        self.qdrant_client = qdrant_client
        self.graph_client = graph_client
        self.graph_extractor = graph_extractor

        # Default search weights
        self.default_weights = {
            'semantic': 0.6,
            'text': 0.3,
            'graph': 0.1
        }

    def search(self, request: HybridSearchRequest) -> List[MemoryResponse]:
        """
        Perform hybrid search across multiple modalities.

        Args:
            request: Hybrid search request with mode and parameters

        Returns:
            List of ranked memory responses
        """
        try:
            search_mode = request.search_mode.mode.lower()

            if search_mode == 'semantic':
                return self._semantic_search(request)
            elif search_mode == 'text':
                return self._text_search(request)
            elif search_mode == 'graph':
                return self._graph_search(request)
            elif search_mode == 'hybrid':
                return self._hybrid_search(request)
            else:
                logger.warning(f"Unknown search mode: {search_mode}, defaulting to semantic")
                return self._semantic_search(request)

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    def _semantic_search(self, request: HybridSearchRequest) -> List[MemoryResponse]:
        """Perform semantic vector search"""
        try:
            memories = self.qdrant_client.search_memories_by_text(
                query=request.query,
                user_id=request.user_id,
                limit=request.limit
            )

            return memories

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def _text_search(self, request: HybridSearchRequest) -> List[MemoryResponse]:
        """Perform text-based search using keyword matching"""
        try:
            # Get user memories for text search (reduced limit for performance)
            all_memories = self.qdrant_client.get_memories(
                user_id=request.user_id,
                limit=200  # Reduced from 1000 for performance
            )

            if not all_memories:
                return []

            # Perform text matching
            query_terms = self._extract_search_terms(request.query)
            scored_memories = []

            for memory in all_memories:
                memory_text = memory.memory.lower()
                score = self._calculate_text_score(memory_text, query_terms)

                if score > 0.3:  # Only include reasonably good text matches
                    # Set text-based score
                    memory.score = score
                    scored_memories.append(memory)

                    # Early termination if we have enough good matches
                    if len(scored_memories) >= request.limit * 2:
                        break

            # Sort by text score and limit
            scored_memories.sort(key=lambda x: x.score, reverse=True)
            return scored_memories[:request.limit]

        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []

    def _graph_search(self, request: HybridSearchRequest) -> List[MemoryResponse]:
        """Perform graph-enhanced search"""
        try:
            if not self.graph_client.connected:
                logger.warning("Graph client not connected, falling back to semantic search")
                return self._semantic_search(request)

            # Extract entities from search query
            search_entities = self.graph_extractor.extract_search_entities(request.query)

            if not search_entities:
                # No entities found, use semantic search
                return self._semantic_search(request)

            # Find related entities in graph
            related_memory_ids = set()
            all_entities = self.graph_client.get_entities_by_user(request.user_id, limit=100)

            for entity in all_entities:
                entity_name = entity.get('name', '').lower()

                # Check if entity matches search terms
                for search_entity in search_entities:
                    if search_entity in entity_name or entity_name in search_entity:
                        # Find related entities
                        related_entities = self.graph_client.search_related_entities(
                            entity['id'], request.user_id, max_depth=2
                        )

                        # Collect memory IDs from related entities
                        for related in related_entities:
                            memory_id = related.get('memory_id')
                            if memory_id:
                                related_memory_ids.add(memory_id)

                        # Also add the original entity's memory
                        if entity.get('memory_id'):
                            related_memory_ids.add(entity['memory_id'])

            if not related_memory_ids:
                return self._semantic_search(request)

            # Get memories by IDs and combine with semantic search
            graph_memories = []
            all_memories = self.qdrant_client.get_memories(request.user_id, limit=1000)

            for memory in all_memories:
                if memory.id in related_memory_ids:
                    # Boost score for graph-found memories
                    memory.score = (memory.score or 0.5) * 1.2
                    graph_memories.append(memory)

            # Supplement with semantic search if needed
            if len(graph_memories) < request.limit:
                semantic_memories = self._semantic_search(request)

                # Add semantic memories not already in graph results
                existing_ids = {mem.id for mem in graph_memories}
                for sem_mem in semantic_memories:
                    if sem_mem.id not in existing_ids:
                        graph_memories.append(sem_mem)

            # Sort by score and limit
            graph_memories.sort(key=lambda x: x.score or 0, reverse=True)
            return graph_memories[:request.limit]

        except Exception as e:
            logger.error(f"Error in graph search: {e}")
            return self._semantic_search(request)

    def _hybrid_search(self, request: HybridSearchRequest) -> List[MemoryResponse]:
        """Perform weighted hybrid search combining all modes"""
        try:
            # Get weights from request or use defaults
            weights = request.search_mode.weights or self.default_weights

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

            # Start with semantic search (fastest and most reliable)
            search_results = {}

            if weights.get('semantic', 0) > 0:
                semantic_request = HybridSearchRequest(
                    query=request.query,
                    user_id=request.user_id,
                    search_mode=SearchMode(mode='semantic'),
                    limit=request.limit
                )
                search_results['semantic'] = self._semantic_search(semantic_request)

                # Early termination: if we have good semantic results, skip slower searches
                if search_results['semantic']:
                    best_semantic_score = max((r.score or 0) for r in search_results['semantic'])
                    if best_semantic_score > 0.7:  # If we have very good semantic matches
                        logger.info(f"Hybrid search: Early termination with high-quality semantic results (best score: {best_semantic_score})")
                        return search_results['semantic'][:request.limit]

            # Only do text search if semantic didn't give good results
            if weights.get('text', 0) > 0 and (not search_results.get('semantic') or len(search_results['semantic']) < request.limit // 2):
                text_request = HybridSearchRequest(
                    query=request.query,
                    user_id=request.user_id,
                    search_mode=SearchMode(mode='text'),
                    limit=request.limit // 2  # Reduced limit for performance
                )
                search_results['text'] = self._text_search(text_request)

            # Only do graph search if we have few results and graph is available
            if (weights.get('graph', 0) > 0 and
                self.graph_client.connected and
                sum(len(results) for results in search_results.values()) < request.limit):

                try:
                    graph_request = HybridSearchRequest(
                        query=request.query,
                        user_id=request.user_id,
                        search_mode=SearchMode(mode='graph'),
                        limit=request.limit // 3  # Very small limit for graph search
                    )
                    search_results['graph'] = self._graph_search(graph_request)
                except Exception as e:
                    logger.warning(f"Graph search failed, continuing without it: {e}")
                    weights['graph'] = 0

            # Combine and rank results
            combined_results = self._combine_search_results(search_results, weights)

            # Debug: print all scores before filtering
            logger.info(f"Hybrid search: all scores before threshold: {[float(getattr(r, 'score', 0) or 0) for r in combined_results]}")

            # Apply minimum similarity threshold to filter out low-quality results
            MIN_SIMILARITY_THRESHOLD = 0.5  # 50% threshold for meaningful matches
            filtered_results = [
                result for result in combined_results
                if getattr(result, 'score', None) is not None and float(result.score) >= MIN_SIMILARITY_THRESHOLD
            ]

            logger.info(f"Hybrid search: {len(combined_results)} total results, {len(filtered_results)} above threshold ({MIN_SIMILARITY_THRESHOLD})")

            # If no results pass the threshold, return top N with a low_confidence flag
            if not filtered_results:
                logger.warning(f"Hybrid search: No results above threshold {MIN_SIMILARITY_THRESHOLD}, returning top {request.limit} low-confidence results.")
                # Optionally, you could add a 'low_confidence' flag to the result objects
                for r in combined_results[:request.limit]:
                    setattr(r, 'low_confidence', True)
                return combined_results[:request.limit]

            return filtered_results[:request.limit]

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self._semantic_search(request)

    def _combine_search_results(self, search_results: Dict[str, List[MemoryResponse]],
                               weights: Dict[str, float]) -> List[MemoryResponse]:
        """Combine results from different search modes"""
        try:
            # Collect all unique memories with their scores from different modes
            memory_scores = {}

            for search_type, memories in search_results.items():
                weight = weights.get(search_type, 0)

                for i, memory in enumerate(memories):
                    memory_id = memory.id

                    # Calculate position-based score (higher for earlier positions)
                    position_score = 1.0 - (i / len(memories)) if memories else 0

                    # Use memory score if available, otherwise use position
                    base_score = memory.score if memory.score is not None else position_score

                    # Apply search type weight
                    weighted_score = base_score * weight

                    if memory_id not in memory_scores:
                        memory_scores[memory_id] = {
                            'memory': memory,
                            'total_score': 0,
                            'component_scores': {}
                        }

                    memory_scores[memory_id]['total_score'] += weighted_score
                    memory_scores[memory_id]['component_scores'][search_type] = weighted_score

            # Sort by combined score
            ranked_memories = sorted(
                memory_scores.values(),
                key=lambda x: x['total_score'],
                reverse=True
            )

            # Return memories with updated scores
            result_memories = []
            for item in ranked_memories:
                memory = item['memory']
                memory.score = item['total_score']
                result_memories.append(memory)

            return result_memories

        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return []

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms from query"""
        # Remove punctuation and convert to lowercase
        clean_query = re.sub(r'[^\w\s]', ' ', query.lower())

        # Split into terms and filter out short words
        terms = [term.strip() for term in clean_query.split() if len(term.strip()) > 2]

        return terms

    def _calculate_text_score(self, text: str, query_terms: List[str]) -> float:
        """Calculate text matching score"""
        if not query_terms:
            return 0

        text_lower = text.lower()
        matches = 0
        total_matches = 0

        for term in query_terms:
            # Count exact matches
            term_matches = text_lower.count(term)
            if term_matches > 0:
                matches += 1
                total_matches += term_matches

        # Calculate score based on:
        # 1. Percentage of query terms found
        # 2. Total number of matches (with diminishing returns)
        term_coverage = matches / len(query_terms)
        match_density = min(total_matches / len(query_terms), 2.0) / 2.0  # Cap at 2x

        return (term_coverage * 0.7) + (match_density * 0.3)

    def get_search_suggestions(self, partial_query: str, user_id: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on partial query"""
        try:
            if len(partial_query) < 2:
                return []

            # Get user's entities for suggestions
            if self.graph_client.connected:
                entities = self.graph_client.get_entities_by_user(user_id, limit=100)

                suggestions = []
                partial_lower = partial_query.lower()

                for entity in entities:
                    entity_name = entity.get('name', '').lower()

                    # Check if entity name contains the partial query
                    if partial_lower in entity_name:
                        suggestions.append(entity.get('name', ''))

                # Add some common search patterns
                common_patterns = [
                    f"{partial_query} information",
                    f"{partial_query} details",
                    f"about {partial_query}",
                    f"{partial_query} discussion"
                ]

                suggestions.extend(common_patterns)

                # Remove duplicates and limit
                unique_suggestions = list(dict.fromkeys(suggestions))
                return unique_suggestions[:limit]

            return []

        except Exception as e:
            logger.error(f"Error getting search suggestions: {e}")
            return []

    def explain_search_results(self, query: str, results: List[MemoryResponse],
                              search_mode: str) -> Dict[str, Any]:
        """Provide explanation for search results"""
        try:
            explanation = {
                'query': query,
                'search_mode': search_mode,
                'total_results': len(results),
                'explanations': []
            }

            for i, result in enumerate(results[:3]):  # Explain top 3 results
                result_explanation = {
                    'rank': i + 1,
                    'memory_id': result.id,
                    'score': result.score,
                    'reasoning': self._generate_result_reasoning(query, result, search_mode)
                }
                explanation['explanations'].append(result_explanation)

            return explanation

        except Exception as e:
            logger.error(f"Error explaining search results: {e}")
            return {'error': str(e)}

    def _generate_result_reasoning(self, query: str, result: MemoryResponse, search_mode: str) -> str:
        """Generate reasoning for why a result was returned"""
        try:
            if search_mode == 'semantic':
                return f"Semantically similar to your query with {result.score:.2f} similarity score"
            elif search_mode == 'text':
                query_terms = self._extract_search_terms(query)
                matched_terms = [term for term in query_terms if term in result.memory.lower()]
                if matched_terms:
                    return f"Contains keywords: {', '.join(matched_terms)}"
                return "Text-based match found"
            elif search_mode == 'graph':
                return "Related through entity connections in your knowledge graph"
            elif search_mode == 'hybrid':
                return f"Combined relevance score of {result.score:.2f} from multiple search methods"
            else:
                return "Relevant to your search query"

        except Exception as e:
            logger.warning(f"Error generating reasoning: {e}")
            return "Matched your search criteria"
