import unittest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.api import reasoning

client = TestClient(app)

class TestReasoningAPI(unittest.TestCase):
    @patch.object(reasoning.neo4j_client, "sync_patient_nlp", return_value=None)
    @patch.object(reasoning.wiki_client, "get_qid_for_entity", new_callable=AsyncMock)
    @patch.object(reasoning.wiki_client, "find_candidate_diseases", new_callable=AsyncMock)
    def test_reasoning_process_endpoint(self, mock_find_diseases, mock_get_qid, mock_sync):
        mock_get_qid.return_value = "Q12140"
        mock_find_diseases.return_value = [
            {
                "wikidata_id": "Q190805",
                "disease_name": "heart disease",
                "match_count": 2,
            }
        ]

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

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["patient_id"], "patient_001")
        self.assertEqual(len(data["graph_nodes"]), 2)
        self.assertEqual(len(data["candidate_diseases"]), 1)
        self.assertEqual(data["candidate_diseases"][0]["disease_name"], "heart disease")
        self.assertEqual(data["risk_analysis"]["risk_level"], "HIGH")

if __name__ == "__main__":
    unittest.main()