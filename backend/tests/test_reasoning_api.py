import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api import reasoning

client = TestClient(app)


def test_reasoning_process_endpoint(monkeypatch):
    # Mock Neo4j Client so test runs without needing local Neo4j active
    monkeypatch.setattr(
        reasoning.neo4j_client,
        "sync_patient_nlp",
        lambda *args, **kwargs: None,
    )

    # Mock Wikidata QID lookup
    async def mock_get_qid(entity_name):
        return "Q12140"

    # Mock Wikidata Differential Diagnosis query
    async def mock_find_candidate_diseases(qids):
        return [
            {
                "wikidata_id": "Q190805",
                "disease_name": "heart disease",
                "match_count": 2,
            }
        ]

    monkeypatch.setattr(reasoning.wiki_client, "get_qid_for_entity", mock_get_qid)
    monkeypatch.setattr(reasoning.wiki_client, "find_candidate_diseases", mock_find_candidate_diseases)

    payload = {
        "entities": [
            {
                "id": "e1",
                "text": "chest pain",
                "category": "symptom",
                "status": "present",
            },
            {
                "id": "e2",
                "text": "shortness of breath",
                "category": "symptom",
                "status": "present",
            },
        ],
        "relations": [
            {
                "source_id": "e1",
                "target_id": "e2",
                "relation_type": "ASSOCIATED_WITH",
            }
        ],
    }

    response = client.post("/api/reasoning/process?patient_id=patient_001", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["patient_id"] == "patient_001"
    assert len(data["graph_nodes"]) == 2
    assert len(data["candidate_diseases"]) == 1
    assert data["candidate_diseases"][0]["disease_name"] == "heart disease"
    assert data["risk_analysis"]["risk_level"] == "HIGH"