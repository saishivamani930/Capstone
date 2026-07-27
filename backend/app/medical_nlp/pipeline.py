from app.medical_nlp.entity_extractor import extract_medical_entities
from app.medical_nlp.normalizer import normalize_entities

from app.medical_nlp.context_rules import extract_context_entities

from app.medical_nlp.negation import apply_negation

from app.medical_nlp.relation_builder import build_symptom_records

CATEGORY_MAP = {
    "Sign_symptom": "symptoms",
    "Disease_disorder": "diseases",
    "Medication": "medications",
    "Allergy": "allergies",
    "Duration": "durations",
    "Severity": "severities",
    "Diagnostic_procedure": "diagnostic_tests",
    "Therapeutic_procedure": "procedures",
    "Family_history": "family_history",
    "Aggravating_factor": "aggravating_factors",
}


def run_medical_nlp(text: str) -> dict:
    raw_entities = extract_medical_entities(text)
    entities = normalize_entities(text, raw_entities)

    entities.extend(extract_context_entities(text))
    entities = sorted(entities, key=lambda item: item["start"])

    SUPPORTED_LABELS = set(CATEGORY_MAP.keys())

    entities = [
        entity
        for entity in entities
        if entity["label"] in SUPPORTED_LABELS
        and not (
            entity["label"] == "Sign_symptom"
            and entity["text"].strip().lower() == "allergic"
        )
    ]

    entities = apply_negation(text, entities)
    symptom_records = build_symptom_records(entities)
    structured = {
        "symptoms": [],
        "diseases": [],
        "medications": [],
        "allergies": [],
        "durations": [],
        "severities": [],
        "diagnostic_tests": [],
        "procedures": [],
        "family_history": [],
        "other_entities": [],
        "aggravating_factors": [],
        "negated_entities": [],
    }

    for entity in entities:
        if entity.get("negated", False):
            structured["negated_entities"].append(entity)
            continue

        category = CATEGORY_MAP.get(entity["label"])

        if category:
            structured[category].append(entity)
        else:
            structured["other_entities"].append(entity)

    return {
    "input_text": text,
    "raw_entities": raw_entities,
    "entities": entities,
    "structured_entities": structured,
    "clinical_facts": {
        "symptom_records": symptom_records,
    },
}