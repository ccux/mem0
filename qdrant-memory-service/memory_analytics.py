"""
Memory Analytics and Clustering Service for advanced memory analysis.

This module provides clustering, analytics, and insights for memory data.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import json
import re

from models import MemoryCluster, MemoryAnalyticsResponse

logger = logging.getLogger(__name__)

class MemoryAnalytics:
    """Advanced analytics and clustering for memory data"""

    def __init__(self, gemini_client=None):
        self.gemini_client = gemini_client
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )

    def _generate_gemini_response(self, prompt: str) -> str:
        """Generate response using Gemini LLM"""
        try:
            if self.gemini_client:
                from google import genai
                model = self.gemini_client.client.models.generate_content(
                    model=self.gemini_client.llm_model,
                    contents=prompt
                )
                return model.text if model else ""
            return ""
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return ""

    def analyze_memories(self, memories: List[Dict[str, Any]], user_id: str) -> MemoryAnalyticsResponse:
        """
        Perform comprehensive analysis of user memories.

        Args:
            memories: List of memory objects
            user_id: User ID for context

        Returns:
            Comprehensive analytics response
        """
        try:
            if not memories:
                return MemoryAnalyticsResponse(
                    total_memories=0,
                    memory_clusters=[],
                    temporal_distribution={},
                    category_distribution={},
                    average_similarity_scores={},
                    insights=["No memories to analyze"]
                )

            logger.info(f"Analyzing {len(memories)} memories for user {user_id}")

            # Extract memory texts and metadata
            memory_texts = [mem.get('memory', '') for mem in memories]

            # Perform clustering
            clusters = self._cluster_memories(memory_texts, memories)

            # Temporal analysis
            temporal_dist = self._analyze_temporal_distribution(memories)

            # Category analysis
            category_dist = self._analyze_category_distribution(memories)

            # Similarity analysis
            similarity_scores = self._analyze_similarity_scores(memories)

            # Generate insights
            insights = self._generate_insights(memories, clusters, temporal_dist, category_dist)

            return MemoryAnalyticsResponse(
                total_memories=len(memories),
                memory_clusters=clusters,
                temporal_distribution=temporal_dist,
                category_distribution=category_dist,
                average_similarity_scores=similarity_scores,
                insights=insights
            )

        except Exception as e:
            logger.error(f"Error analyzing memories: {e}")
            return MemoryAnalyticsResponse(
                total_memories=len(memories) if memories else 0,
                memory_clusters=[],
                temporal_distribution={},
                category_distribution={},
                average_similarity_scores={},
                insights=[f"Error during analysis: {str(e)}"]
            )

    def _cluster_memories(self, memory_texts: List[str], memories: List[Dict[str, Any]], max_clusters: int = 5) -> List[MemoryCluster]:
        """Cluster memories by content similarity"""
        try:
            if len(memory_texts) < 2:
                return []

            # Vectorize memory texts
            try:
                tfidf_matrix = self.vectorizer.fit_transform(memory_texts)
            except Exception as e:
                logger.warning(f"TF-IDF vectorization failed: {e}")
                return []

            # Determine optimal number of clusters
            n_clusters = min(max_clusters, len(memory_texts) // 2, 10)
            if n_clusters < 2:
                return []

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)

            # Build clusters
            clusters = []
            feature_names = self.vectorizer.get_feature_names_out()

            for cluster_id in range(n_clusters):
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]

                if not cluster_indices:
                    continue

                # Get representative memories (up to 3)
                cluster_memories = [memory_texts[i] for i in cluster_indices]
                representative_memories = cluster_memories[:3]

                # Extract keywords for this cluster
                cluster_center = kmeans.cluster_centers_[cluster_id]
                top_indices = cluster_center.argsort()[-10:][::-1]  # Top 10 features
                keywords = [feature_names[i] for i in top_indices]

                # Generate topic name
                topic = self._generate_cluster_topic(cluster_memories, keywords)

                cluster = MemoryCluster(
                    cluster_id=cluster_id,
                    topic=topic,
                    memory_count=len(cluster_indices),
                    representative_memories=representative_memories,
                    keywords=keywords[:5]  # Top 5 keywords
                )
                clusters.append(cluster)

            # Sort clusters by size (largest first)
            clusters.sort(key=lambda x: x.memory_count, reverse=True)

            return clusters

        except Exception as e:
            logger.error(f"Error clustering memories: {e}")
            return []

    def _analyze_temporal_distribution(self, memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze when memories were created"""
        try:
            temporal_dist = defaultdict(int)

            for memory in memories:
                created_at = memory.get('created_at')
                if created_at:
                    try:
                        if isinstance(created_at, str):
                            date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        else:
                            date = created_at

                        # Group by date (YYYY-MM-DD)
                        date_key = date.strftime('%Y-%m-%d')
                        temporal_dist[date_key] += 1

                    except Exception as e:
                        logger.warning(f"Error parsing date {created_at}: {e}")

            return dict(temporal_dist)

        except Exception as e:
            logger.error(f"Error analyzing temporal distribution: {e}")
            return {}

    def _analyze_category_distribution(self, memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze memory categories"""
        try:
            category_dist = defaultdict(int)

            for memory in memories:
                metadata = memory.get('metadata', {})

                # Try different category fields
                category = (
                    metadata.get('category') or
                    metadata.get('source') or
                    metadata.get('type') or
                    'uncategorized'
                )

                category_dist[str(category)] += 1

            return dict(category_dist)

        except Exception as e:
            logger.error(f"Error analyzing category distribution: {e}")
            return {}

    def _analyze_similarity_scores(self, memories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze similarity score distributions"""
        try:
            scores_by_category = defaultdict(list)

            for memory in memories:
                score = memory.get('score')
                if score is not None:
                    metadata = memory.get('metadata', {})
                    category = metadata.get('category', 'uncategorized')
                    scores_by_category[str(category)].append(float(score))

            # Calculate averages
            avg_scores = {}
            for category, scores in scores_by_category.items():
                if scores:
                    avg_scores[category] = sum(scores) / len(scores)

            return avg_scores

        except Exception as e:
            logger.error(f"Error analyzing similarity scores: {e}")
            return {}

    def _generate_cluster_topic(self, cluster_memories: List[str], keywords: List[str]) -> str:
        """Generate a topic name for a memory cluster"""
        try:
            if self.gemini_client:
                # Use LLM to generate topic
                prompt = f"""
Analyze these related memories and generate a short, descriptive topic name (2-4 words):

Memories:
{chr(10).join(cluster_memories[:3])}

Keywords: {', '.join(keywords[:5])}

Generate a concise topic name that captures the main theme:
"""
                response = self._generate_gemini_response(prompt)
                if response:
                    # Extract topic from response
                    topic = response.strip().strip('"').strip("'")
                    if len(topic) < 50 and topic:
                        return topic

            # Fallback: Use top keywords
            if keywords:
                return ' '.join(keywords[:2]).title()

            return "Miscellaneous"

        except Exception as e:
            logger.warning(f"Error generating cluster topic: {e}")
            return "Unknown Topic"

    def _generate_insights(self, memories: List[Dict[str, Any]], clusters: List[MemoryCluster],
                          temporal_dist: Dict[str, int], category_dist: Dict[str, int]) -> List[str]:
        """Generate insights about the user's memory patterns"""
        try:
            insights = []

            # Memory volume insights
            total_memories = len(memories)
            if total_memories > 50:
                insights.append(f"You have an extensive memory base with {total_memories} memories")
            elif total_memories > 20:
                insights.append(f"You have a growing memory collection with {total_memories} memories")
            else:
                insights.append(f"Your memory collection is growing with {total_memories} memories")

            # Clustering insights
            if clusters:
                largest_cluster = max(clusters, key=lambda x: x.memory_count)
                insights.append(f"Your most frequent topic is '{largest_cluster.topic}' with {largest_cluster.memory_count} related memories")

                if len(clusters) >= 3:
                    insights.append(f"Your memories span {len(clusters)} distinct topics, showing diverse interests")

            # Temporal insights
            if temporal_dist:
                dates = list(temporal_dist.keys())
                if len(dates) > 1:
                    dates.sort()
                    recent_date = dates[-1]
                    recent_count = temporal_dist[recent_date]
                    insights.append(f"Most recent activity: {recent_count} memories on {recent_date}")

                    # Activity pattern
                    if len(dates) >= 7:
                        weekly_activity = sum(temporal_dist.values()) / len(dates) * 7
                        insights.append(f"Average weekly memory creation: {weekly_activity:.1f} memories")

            # Category insights
            if category_dist:
                top_category = max(category_dist.items(), key=lambda x: x[1])
                insights.append(f"Most common memory source: '{top_category[0]}' ({top_category[1]} memories)")

            # Diversity insights
            unique_sources = len(set(mem.get('metadata', {}).get('source', 'unknown') for mem in memories))
            if unique_sources > 3:
                insights.append(f"Your memories come from {unique_sources} different sources, showing varied learning")

            return insights[:6]  # Limit to 6 insights

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return ["Analysis completed successfully"]

    def find_similar_memories(self, target_memory: str, memories: List[Dict[str, Any]],
                            threshold: float = 0.3, limit: int = 5) -> List[Dict[str, Any]]:
        """Find memories similar to a target memory"""
        try:
            if not memories:
                return []

            memory_texts = [mem.get('memory', '') for mem in memories]
            all_texts = memory_texts + [target_memory]

            # Vectorize all texts
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)

            # Calculate similarity with target (last vector)
            target_vector = tfidf_matrix[-1]
            similarities = (tfidf_matrix[:-1] * target_vector.T).toarray().flatten()

            # Find similar memories above threshold
            similar_indices = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    similar_indices.append((i, similarity))

            # Sort by similarity and limit results
            similar_indices.sort(key=lambda x: x[1], reverse=True)
            similar_indices = similar_indices[:limit]

            # Return similar memories with scores
            similar_memories = []
            for idx, similarity in similar_indices:
                memory = memories[idx].copy()
                memory['similarity_score'] = float(similarity)
                similar_memories.append(memory)

            return similar_memories

        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []

    def get_memory_trends(self, memories: List[Dict[str, Any]], days: int = 30) -> Dict[str, Any]:
        """Analyze memory creation trends over time"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            recent_memories = []

            for memory in memories:
                created_at = memory.get('created_at')
                if created_at:
                    try:
                        if isinstance(created_at, str):
                            date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        else:
                            date = created_at

                        if date >= cutoff_date:
                            recent_memories.append(memory)
                    except:
                        continue

            # Calculate trends
            total_recent = len(recent_memories)
            daily_average = total_recent / days if days > 0 else 0

            # Category trends
            category_counts = defaultdict(int)
            for memory in recent_memories:
                category = memory.get('metadata', {}).get('category', 'uncategorized')
                category_counts[category] += 1

            return {
                'period_days': days,
                'total_memories': total_recent,
                'daily_average': daily_average,
                'category_trends': dict(category_counts),
                'trend_direction': 'increasing' if daily_average > 1 else 'stable' if daily_average > 0.5 else 'decreasing'
            }

        except Exception as e:
            logger.error(f"Error analyzing memory trends: {e}")
            return {
                'period_days': days,
                'total_memories': 0,
                'daily_average': 0,
                'category_trends': {},
                'trend_direction': 'unknown'
            }
