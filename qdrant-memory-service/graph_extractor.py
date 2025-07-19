"""
Graph Extractor for extracting entities and relationships from memory content.

This module uses LLMs to analyze memory content and extract structured
entities and relationships for graph memory functionality.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from neo4j_graph_client import GraphEntity, GraphRelationship
import uuid
from google import genai

logger = logging.getLogger(__name__)

class GraphExtractor:
    """Extracts entities and relationships from memory content using LLMs"""

    def __init__(self, gemini_client):
        self.gemini_client = gemini_client

    def _generate_gemini_response(self, prompt: str) -> str:
        """Generate response using Gemini LLM"""
        try:
            model = self.gemini_client.client.models.generate_content(
                model=self.gemini_client.llm_model,
                contents=prompt
            )
            return model.text if model else ""
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return ""

    def extract_entities_and_relationships(self, memory_content: str, user_id: str) -> Tuple[List[GraphEntity], List[GraphRelationship]]:
        """
        Extract entities and relationships from memory content.

        Args:
            memory_content: The memory text to analyze
            user_id: The user ID for context

        Returns:
            Tuple of (entities, relationships)
        """
        try:
            # Use Gemini to extract structured data
            extraction_prompt = self._build_extraction_prompt(memory_content)
            response = self._generate_gemini_response(extraction_prompt)

            if not response:
                logger.warning("No response from Gemini for entity extraction")
                return [], []

            # Parse the response to extract entities and relationships
            entities, relationships = self._parse_extraction_response(response, user_id)

            logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
            return entities, relationships

        except Exception as e:
            logger.error(f"Error extracting entities and relationships: {e}")
            return [], []

    def _build_extraction_prompt(self, memory_content: str) -> str:
        """Build the prompt for entity and relationship extraction"""
        return f"""
You are an expert at extracting entities and relationships from text for knowledge graph construction.

Analyze the following memory content and extract:
1. Entities (people, places, organizations, concepts, objects, etc.)
2. Relationships between these entities

Memory Content: "{memory_content}"

IMPORTANT INSTRUCTIONS:
- Extract only concrete, specific entities mentioned in the text
- Avoid generic concepts unless they are specifically important
- For people, use their actual names if mentioned
- For places, be as specific as possible
- Focus on meaningful relationships, not trivial ones
- Each entity should have a clear type (Person, Location, Organization, Concept, Object, etc.)

Return your response in this EXACT JSON format:
{{
    "entities": [
        {{
            "id": "unique_identifier",
            "name": "Entity Name",
            "type": "EntityType",
            "properties": {{
                "description": "brief description",
                "context": "context from memory"
            }}
        }}
    ],
    "relationships": [
        {{
            "source_id": "entity1_id",
            "target_id": "entity2_id",
            "relationship_type": "RELATIONSHIP_TYPE",
            "properties": {{
                "description": "relationship description",
                "strength": "high|medium|low"
            }}
        }}
    ]
}}

Ensure all entity IDs are unique and all relationship source_id and target_id values match entity IDs.
Use relationship types like: LIVES_IN, WORKS_AT, KNOWS, LIKES, DISLIKES, OWNS, RELATED_TO, PART_OF, etc.
"""

    def _parse_extraction_response(self, response: str, user_id: str) -> Tuple[List[GraphEntity], List[GraphRelationship]]:
        """Parse the LLM response to extract entities and relationships"""
        entities = []
        relationships = []

        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in extraction response")
                return entities, relationships

            json_str = json_match.group()
            data = json.loads(json_str)

            # Extract entities
            if "entities" in data:
                for entity_data in data["entities"]:
                    try:
                        entity = GraphEntity(
                            id=self._clean_entity_id(entity_data.get("id", str(uuid.uuid4()))),
                            name=entity_data.get("name", "Unknown"),
                            type=entity_data.get("type", "Unknown"),
                            properties=entity_data.get("properties", {})
                        )
                        entities.append(entity)
                    except Exception as e:
                        logger.warning(f"Error parsing entity {entity_data}: {e}")

            # Extract relationships
            if "relationships" in data:
                entity_ids = {entity.id for entity in entities}

                for rel_data in data["relationships"]:
                    try:
                        source_id = self._clean_entity_id(rel_data.get("source_id", ""))
                        target_id = self._clean_entity_id(rel_data.get("target_id", ""))

                        # Only create relationships between extracted entities
                        if source_id in entity_ids and target_id in entity_ids:
                            relationship = GraphRelationship(
                                source_id=source_id,
                                target_id=target_id,
                                relationship_type=rel_data.get("relationship_type", "RELATED_TO"),
                                properties=rel_data.get("properties", {})
                            )
                            relationships.append(relationship)
                        else:
                            logger.warning(f"Skipping relationship with missing entities: {source_id} -> {target_id}")
                    except Exception as e:
                        logger.warning(f"Error parsing relationship {rel_data}: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in extraction response: {e}")
            logger.debug(f"Response content: {response}")
        except Exception as e:
            logger.error(f"Unexpected error parsing extraction response: {e}")

        return entities, relationships

    def _clean_entity_id(self, entity_id: str) -> str:
        """Clean and normalize entity ID"""
        if not entity_id:
            return str(uuid.uuid4())

        # Remove special characters and spaces, convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', str(entity_id).lower())
        cleaned = re.sub(r'_+', '_', cleaned)  # Remove multiple underscores
        cleaned = cleaned.strip('_')  # Remove leading/trailing underscores

        if not cleaned:
            return str(uuid.uuid4())

        return cleaned

    def extract_search_entities(self, search_query: str) -> List[str]:
        """
        Extract entity names/types from a search query for graph-enhanced search.

        Args:
            search_query: The search query text

        Returns:
            List of entity identifiers to search for in the graph
        """
        try:
            extraction_prompt = f"""
Analyze this search query and identify any entities (people, places, organizations, concepts) that might be relevant for searching a knowledge graph.

Search Query: "{search_query}"

Extract entity names, types, or identifiers that could help find related information in a knowledge graph.
Focus on concrete nouns, proper names, and specific concepts.

Return a simple JSON list of entity names/identifiers:
["entity1", "entity2", "entity3"]

If no clear entities are found, return an empty list: []
"""

            response = self._generate_gemini_response(extraction_prompt)

            if response:
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    entities = json.loads(json_match.group())
                    # Clean and validate entity names
                    cleaned_entities = []
                    for entity in entities:
                        if isinstance(entity, str) and len(entity.strip()) > 1:
                            cleaned_entities.append(entity.strip().lower())
                    return cleaned_entities

            return []

        except Exception as e:
            logger.error(f"Error extracting search entities: {e}")
            return []
