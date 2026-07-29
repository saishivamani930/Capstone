import asyncio
from app.reasoning.missing_info import MissingInfoEngine


def test_missing_info_deduplicated():
    engine = MissingInfoEngine()
    present_symptoms = ["chest pain", "shortness of breath"]
    candidate_diseases = [{"disease_name": "heart disease", "wikidata_id": "Q190805"}]

    # Run the async function using asyncio.run
    result = asyncio.run(engine.analyze(present_symptoms, candidate_diseases))

    assert len(result) == 1
    assert result[0]["disease_name"] == "Further Clinical Evaluation"
    assert "radiation of pain" in result[0]["missing_symptoms"]
    assert "orthopnea" in result[0]["missing_symptoms"]
    assert len(result[0]["suggested_questions"]) == 3