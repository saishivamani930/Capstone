from typing import List
from app.reasoning.wikidata_client import WikidataClient
from app.schemas.reasoning import CandidateDisease, MissingInformation

class MissingInfoEngine:
    def __init__(self, wiki_client=None):
        self.wiki_client = wiki_client

    async def analyze(self, present_symptoms: list, candidate_diseases: list) -> list:
        present_set = {s.lower() for s in present_symptoms}
        
        symptoms_to_ask = []
        questions = []

        # Deduplicated global clinical rules based on present symptoms
        if "chest pain" in present_set:
            symptoms_to_ask.extend(["radiation of pain", "diaphoresis"])
            questions.extend([
                "Does the pain radiate to your left arm, neck, or jaw?",
                "Are you experiencing cold sweats or nausea?"
            ])

        if "shortness of breath" in present_set:
            symptoms_to_ask.append("orthopnea")
            questions.append("Does the shortness of breath worsen when lying flat?")

        missing_info = []

        # Return a single aggregated missing information object across all candidate diseases
        if candidate_diseases:
            # Map the deduplicated questions under a primary or general missing info entry
            missing_info.append({
                "disease_name": "Further Clinical Evaluation",
                "wikidata_id": "",
                "missing_symptoms": symptoms_to_ask,
                "suggested_questions": questions
            })

        return missing_info