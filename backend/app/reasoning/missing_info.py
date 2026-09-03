from typing import List
from app.reasoning.wikidata_client import WikidataClient
from app.schemas.reasoning import CandidateDisease, MissingInformation

class MissingInfoEngine:
    def __init__(self, wiki_client=None):
        self.wiki_client = wiki_client

    def analyze_sync(self, present_symptoms: list, candidate_diseases: list = None, all_entities: list = None) -> list:
        candidate_diseases = candidate_diseases or []
        present_set = {s.lower() for s in present_symptoms}
        
        all_text = ""
        if all_entities:
            all_text = " ".join([(e.get("text") if isinstance(e, dict) else getattr(e, "text", "")).lower() for e in all_entities])

        symptoms_to_ask = []
        questions = []

        # Standard ACC/AHA Clinical Guidelines Rules (Filtered by spoken text)
        if any(term in present_set for term in ["chest pain", "angina", "chest pressure", "chest tightness"]):
            # Only ask about radiation if arm/shoulder/jaw was NOT already mentioned
            if not any(term in all_text for term in ["arm", "shoulder", "jaw", "radiat"]):
                symptoms_to_ask.append("radiation of pain")
                questions.append("Does the pain radiate to your left arm, neck, shoulder, or jaw?")

            # Only ask about cold sweats/nausea if sweating/nausea was NOT already mentioned
            if not any(term in all_text for term in ["sweat", "diaphoresis", "nausea", "fever"]):
                symptoms_to_ask.extend(["diaphoresis", "nausea"])
                questions.append("Are you experiencing cold sweats (diaphoresis) or nausea?")

        if any(term in present_set for term in ["shortness of breath", "dyspnea", "breathlessness"]):
            if not any(term in all_text for term in ["orthopnea", "lying flat"]):
                symptoms_to_ask.append("orthopnea")
                questions.append("Does the shortness of breath worsen when lying flat (orthopnea)?")

        if any(term in present_set for term in ["palpitations", "racing heart", "irregular heartbeat"]):
            if not any(term in all_text for term in ["syncope", "faint", "dizziness"]):
                symptoms_to_ask.append("syncope")
                questions.append("Do the palpitations occur with lightheadedness, dizziness, or fainting?")

        if any(term in present_set for term in ["swelling", "edema", "leg swelling"]):
            symptoms_to_ask.extend(["weight gain", "pitting edema"])
            questions.extend([
                "Have you noticed sudden weight gain over the last few days?",
                "Is the swelling in both legs or just one side?"
            ])

        if any(term in present_set for term in ["dizziness", "syncope", "fainting"]):
            symptoms_to_ask.extend(["chest discomfort", "confusion"])
            questions.extend([
                "Did you experience any chest discomfort or palpitations before fainting?",
                "Was there any loss of consciousness?"
            ])

        missing_info = []

        # Return clinical guideline recommendations whenever symptoms are present or candidates exist
        if symptoms_to_ask or questions or candidate_diseases:
            disease_label = (
                candidate_diseases[0].disease_name 
                if candidate_diseases and hasattr(candidate_diseases[0], "disease_name") 
                else ("Further Clinical Evaluation" if not candidate_diseases else candidate_diseases[0].get("disease_name", "Clinical Evaluation"))
            )
            disease_qid = (
                candidate_diseases[0].wikidata_id 
                if candidate_diseases and hasattr(candidate_diseases[0], "wikidata_id") 
                else (candidate_diseases[0].get("wikidata_id", "") if candidate_diseases else "")
            )

            missing_info.append({
                "disease_name": disease_label,
                "wikidata_id": disease_qid,
                "missing_symptoms": symptoms_to_ask,
                "suggested_questions": questions
            })

        return missing_info

    async def analyze(self, present_symptoms: list, candidate_diseases: list = None) -> list:
        return self.analyze_sync(present_symptoms, candidate_diseases)