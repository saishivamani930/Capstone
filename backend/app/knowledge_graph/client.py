import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables from backend/.env file
load_dotenv()

class Neo4jClient:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password123")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def sync_patient_nlp(self, patient_id: str, entities: list, relations: list):
        """
        Inserts extracted entities and relationships from Member 1 into Neo4j.
        """
        query = """
        MERGE (p:Patient {id: $patient_id})
        WITH p
        UNWIND $entities AS e
        MERGE (n:Entity {id: e.id})
        SET n.text = e.text,
            n.category = e.category,
            n.status = e.status,
            n.severity = e.severity,
            n.duration = e.duration,
            n.wikidata_id = COALESCE(e.wikidata_id, "")
        
        MERGE (p)-[r:HAS_ENTITY]->(n)
        SET r.status = e.status
        """
        
        rel_query = """
        UNWIND $relations AS rel
        MATCH (a:Entity {id: rel.source_id})
        MATCH (b:Entity {id: rel.target_id})
        MERGE (a)-[r:RELATED_TO {type: rel.relation_type}]->(b)
        """
        
        with self.driver.session() as session:
            session.run(query, patient_id=patient_id, entities=entities)
            if relations:
                session.run(rel_query, relations=relations)

    def get_patient_symptoms(self, patient_id: str) -> list:
        """
        Returns all symptoms marked as 'present' for a patient.
        """
        query = """
        MATCH (p:Patient {id: $patient_id})-[:HAS_ENTITY]->(e:Entity)
        WHERE e.category = 'symptom' AND e.status = 'present'
        RETURN e.text AS symptom_name, e.wikidata_id AS qid
        """
        with self.driver.session() as session:
            result = session.run(query, patient_id=patient_id)
            return [record.data() for record in result]