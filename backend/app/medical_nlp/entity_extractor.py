from functools import lru_cache

from transformers import pipeline


MODEL_NAME = "d4data/biomedical-ner-all"


@lru_cache(maxsize=1)
def get_ner_pipeline():
    return pipeline(
        "token-classification",
        model=MODEL_NAME,
        aggregation_strategy="max"
    )


def extract_medical_entities(text: str) -> list[dict]:
    if not text.strip():
        return []

    ner = get_ner_pipeline()
    results = ner(text)

    entities = []

    for item in results:
        entities.append({
            "text": item["word"],
            "label": item["entity_group"],
            "confidence": round(float(item["score"]), 4),
            "start": item["start"],
            "end": item["end"]
        })

    return entities