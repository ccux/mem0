"""
Neo4j Graph Client for Graph Memory functionality in Cognition Suite.

This module provides the Neo4j integration for storing and querying
entity relationships and graph-based memories.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase, Transaction
from pydantic import BaseModel
import json
import time

logger = logging.getLogger(__name__)

class GraphEntity(BaseModel):
    """Represents a graph entity (node)"""
    id: str
    name: str
    type: str  # Person, Location, Concept, etc.
    properties: Dict[str, Any] = {}

class GraphRelationship(BaseModel):
    """Represents a graph relationship (edge)"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = {}

class Neo4jGraphClient:
    """Neo4j client for graph memory operations"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.connected = False

    def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            self.connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")

            # Create indexes for better performance
            self._create_indexes()

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Disconnected from Neo4j")

    def _create_indexes(self):
        """Create necessary indexes for optimal performance"""
        indexes = [
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_user_index IF NOT EXISTS FOR (e:Entity) ON (e.user_id)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX memory_id_index IF NOT EXISTS FOR (m:Memory) ON (m.memory_id)",
            "CREATE INDEX memory_user_index IF NOT EXISTS FOR (m:Memory) ON (m.user_id)"
        ]

        with self.driver.session(database=self.database) as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")

    def add_entity(self, entity: GraphEntity, user_id: str, memory_id: str) -> bool:
        """Add or update an entity in the graph"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MERGE (e:Entity {id: $entity_id, user_id: $user_id})
                SET e.name = $name,
                    e.type = $type,
                    e.properties = $properties,
                    e.memory_id = $memory_id,
                    e.updated_at = datetime()
                WITH e
                WHERE e.created_at IS NULL
                SET e.created_at = datetime()
                RETURN e
                """

                result = session.run(query, {
                    "entity_id": entity.id,
                    "user_id": user_id,
                    "name": entity.name,
                    "type": entity.type,
                    "properties": json.dumps(entity.properties),
                    "memory_id": memory_id
                })

                return result.single() is not None

        except Exception as e:
            logger.error(f"Error adding entity {entity.id}: {e}")
            return False

    def add_relationship(self, relationship: GraphRelationship, user_id: str, memory_id: str) -> bool:
        """Add a relationship between entities"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (source:Entity {id: $source_id, user_id: $user_id})
                MATCH (target:Entity {id: $target_id, user_id: $user_id})
                MERGE (source)-[r:RELATES {type: $rel_type}]->(target)
                SET r.properties = $properties,
                    r.memory_id = $memory_id,
                    r.updated_at = datetime()
                WITH r
                WHERE r.created_at IS NULL
                SET r.created_at = datetime()
                RETURN r
                """

                result = session.run(query, {
                    "source_id": relationship.source_id,
                    "target_id": relationship.target_id,
                    "user_id": user_id,
                    "rel_type": relationship.relationship_type,
                    "properties": json.dumps(relationship.properties),
                    "memory_id": memory_id
                })

                return result.single() is not None

        except Exception as e:
            logger.error(f"Error adding relationship {relationship.source_id}->{relationship.target_id}: {e}")
            return False

    def get_entities_by_user(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all entities for a user"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (e:Entity {user_id: $user_id})
                RETURN e.id as id, e.name as name, e.type as type,
                       e.properties as properties, e.created_at as created_at,
                       e.updated_at as updated_at, e.memory_id as memory_id
                ORDER BY e.updated_at DESC
                LIMIT $limit
                """

                result = session.run(query, {"user_id": user_id, "limit": limit})

                entities = []
                for record in result:
                    entity = dict(record)

                    # Convert Neo4j datetime to Python datetime
                    for field in ["created_at", "updated_at"]:
                        if entity.get(field) and hasattr(entity[field], 'to_native'):
                            entity[field] = entity[field].to_native()

                    if entity["properties"]:
                        try:
                            entity["properties"] = json.loads(entity["properties"])
                        except:
                            entity["properties"] = {}

                    # Add user_id field
                    entity["user_id"] = user_id

                    entities.append(entity)

                return entities

        except Exception as e:
            logger.error(f"Error getting entities for user {user_id}: {e}")
            return []

    def search_related_entities(self, entity_id: str, user_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find entities related to a given entity"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (start:Entity {id: $entity_id, user_id: $user_id})
                MATCH (start)-[r*1..$max_depth]-(related:Entity)
                WHERE related.user_id = $user_id
                RETURN DISTINCT related.id as id, related.name as name,
                       related.type as type, related.properties as properties,
                       length(r) as distance
                ORDER BY distance, related.updated_at DESC
                LIMIT 20
                """

                result = session.run(query, {
                    "entity_id": entity_id,
                    "user_id": user_id,
                    "max_depth": max_depth
                })

                related = []
                for record in result:
                    entity = dict(record)
                    if entity["properties"]:
                        try:
                            entity["properties"] = json.loads(entity["properties"])
                        except:
                            entity["properties"] = {}
                    related.append(entity)

                return related

        except Exception as e:
            logger.error(f"Error searching related entities for {entity_id}: {e}")
            return []

    def get_entity_relationships(self, entity_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for an entity"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (e:Entity {id: $entity_id, user_id: $user_id})
                MATCH (e)-[r]-(other:Entity)
                WHERE other.user_id = $user_id
                RETURN r.type as relationship_type, r.properties as properties,
                       other.id as other_id, other.name as other_name,
                       other.type as other_type,
                       CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END as direction
                """

                result = session.run(query, {"entity_id": entity_id, "user_id": user_id})

                relationships = []
                for record in result:
                    rel = dict(record)
                    if rel["properties"]:
                        try:
                            rel["properties"] = json.loads(rel["properties"])
                        except:
                            rel["properties"] = {}
                    relationships.append(rel)

                return relationships

        except Exception as e:
            logger.error(f"Error getting relationships for entity {entity_id}: {e}")
            return []

    def delete_memory_graph_data(self, memory_id: str, user_id: str):
        """Delete all graph data associated with a memory"""
        try:
            with self.driver.session(database=self.database) as session:
                # Delete relationships first
                session.run("""
                    MATCH ()-[r:RELATES]-()
                    WHERE r.memory_id = $memory_id
                    DELETE r
                """, {"memory_id": memory_id})

                # Delete entities that only belong to this memory
                session.run("""
                    MATCH (e:Entity {memory_id: $memory_id, user_id: $user_id})
                    WHERE NOT EXISTS {
                        MATCH ()-[r:RELATES]-()
                        WHERE r.memory_id <> $memory_id AND (startNode(r) = e OR endNode(r) = e)
                    }
                    DELETE e
                """, {"memory_id": memory_id, "user_id": user_id})

        except Exception as e:
            logger.error(f"Error deleting graph data for memory {memory_id}: {e}")

    def get_graph_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary statistics for user's graph"""
        try:
            with self.driver.session(database=self.database) as session:
                # Count entities by type
                entity_query = """
                MATCH (e:Entity {user_id: $user_id})
                RETURN e.type as type, count(e) as count
                ORDER BY count DESC
                """

                entity_result = session.run(entity_query, {"user_id": user_id})
                entity_stats = {record["type"]: record["count"] for record in entity_result}

                # Count relationships by type
                rel_query = """
                MATCH ()-[r:RELATES]-()
                WHERE EXISTS {
                    MATCH (e:Entity {user_id: $user_id})
                    WHERE startNode(r) = e OR endNode(r) = e
                }
                RETURN r.type as type, count(r) as count
                ORDER BY count DESC
                """

                rel_result = session.run(rel_query, {"user_id": user_id})
                relationship_stats = {record["type"]: record["count"] for record in rel_result}

                # Total counts
                total_entities = sum(entity_stats.values())
                total_relationships = sum(relationship_stats.values())

                return {
                    "total_entities": total_entities,
                    "total_relationships": total_relationships,
                    "entity_types": entity_stats,
                    "relationship_types": relationship_stats
                }

        except Exception as e:
            logger.error(f"Error getting graph summary for user {user_id}: {e}")
            return {
                "total_entities": 0,
                "total_relationships": 0,
                "entity_types": {},
                "relationship_types": {}
            }
