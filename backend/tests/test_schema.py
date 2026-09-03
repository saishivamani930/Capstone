import unittest
from pydantic import ValidationError
from app.schemas.clinical import ConsultationResult

def create_valid_result() -> dict:
    return {
        "transcript": "Doctor asked about pain. Patient has chest pain.",
        "language": "en",
        "patient_transcript": "I have chest pain.",
        "conversation": [
            {
                "speaker": "DOCTOR",
                "raw_speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 2.0,
                "text": "What problem are you having?",
            },
            {
                "speaker": "PATIENT",
                "raw_speaker": "SPEAKER_01",
                "start": 2.1,
                "end": 4.0,
                "text": "I have chest pain.",
            },
        ],
        "medical_nlp": {
            "input_text": "I have chest pain.",
            "raw_entities": [],
            "entities": [
                {
                    "text": "chest pain",
                    "label": "Sign_symptom",
                    "start": 7,
                    "end": 17,
                    "source": "normalized",
                    "negated": False,
                    "assertion": "present",
                }
            ],
            "structured_entities": {
                "symptoms": [
                    {
                        "text": "chest pain",
                        "label": "Sign_symptom",
                        "start": 7,
                        "end": 17,
                        "source": "normalized",
                        "negated": False,
                        "assertion": "present",
                    }
                ]
            },
            "clinical_facts": {
                "symptom_records": [
                    {
                        "symptom": {
                            "text": "chest pain",
                            "label": "Sign_symptom",
                            "start": 7,
                            "end": 17,
                            "source": "normalized",
                            "negated": False,
                            "assertion": "present",
                        },
                        "severities": [],
                        "durations": [],
                        "aggravating_factors": [],
                    }
                ]
            },
        },
    }

class TestSchema(unittest.TestCase):
    def test_valid_consultation_result(self):
        result = ConsultationResult.model_validate(
            create_valid_result()
        )

        self.assertEqual(result.language, "en")
        self.assertEqual(result.conversation[0].speaker, "DOCTOR")
        self.assertEqual(result.conversation[1].speaker, "PATIENT")

        symptom = result.medical_nlp.structured_entities.symptoms[0]

        self.assertEqual(symptom.text, "chest pain")
        self.assertFalse(symptom.negated)
        self.assertEqual(symptom.assertion, "present")

    def test_structured_entity_defaults_are_created(self):
        data = create_valid_result()

        data["medical_nlp"]["structured_entities"] = {}
        data["medical_nlp"]["clinical_facts"] = {}

        result = ConsultationResult.model_validate(data)

        structured = result.medical_nlp.structured_entities

        self.assertEqual(structured.symptoms, [])
        self.assertEqual(structured.diseases, [])
        self.assertEqual(structured.medications, [])
        self.assertEqual(structured.negated_entities, [])

        self.assertEqual(
            result.medical_nlp.clinical_facts.symptom_records,
            []
        )

    def test_missing_required_conversation_field_fails(self):
        data = create_valid_result()

        del data["conversation"][0]["speaker"]

        with self.assertRaises(ValidationError):
            ConsultationResult.model_validate(data)

    def test_invalid_entity_position_fails(self):
        data = create_valid_result()

        entity = data["medical_nlp"]["entities"][0]
        entity["start"] = "invalid-position"

        with self.assertRaises(ValidationError):
            ConsultationResult.model_validate(data)

if __name__ == "__main__":
    unittest.main()