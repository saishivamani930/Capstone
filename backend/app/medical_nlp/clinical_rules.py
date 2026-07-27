import re


DISEASE_TERMS = {
    "hypertension",
    "diabetes",
    "angina",
    "heart failure",
    "coronary artery disease",
}

MEDICATION_TERMS = {
    "metformin",
    "aspirin",
    "atorvastatin",
    "amlodipine",
    "nitroglycerin",
    "penicillin",
}


def spans_overlap(first: dict, second: dict) -> bool:
    return (
        first["start"] < second["end"]
        and second["start"] < first["end"]
    )


def extract_clinical_rules(text: str) -> list[dict]:
    entities = []
    allergy_spans = []

    medication_pattern = "|".join(
        sorted(
            (re.escape(term) for term in MEDICATION_TERMS),
            key=len,
            reverse=True,
        )
    )

    allergy_pattern = re.compile(
        rf"\b(?:allergic|allergy)\s+(?:to\s+)?"
        rf"(?P<allergen>{medication_pattern})\b",
        re.IGNORECASE,
    )

    for match in allergy_pattern.finditer(text):
        start, end = match.span("allergen")
        allergy_spans.append((start, end))

        entities.append({
            "text": text[start:end],
            "label": "Allergy",
            "start": start,
            "end": end,
            "source": "rule",
            "evidence": match.group(),
        })

    for disease in DISEASE_TERMS:
        pattern = re.compile(
            rf"\b{re.escape(disease)}\b",
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            entities.append({
                "text": match.group(),
                "label": "Disease_disorder",
                "start": match.start(),
                "end": match.end(),
                "source": "lexicon",
            })

    for medication in MEDICATION_TERMS:
        pattern = re.compile(
            rf"\b{re.escape(medication)}\b",
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            is_allergy = any(
                match.start() == start and match.end() == end
                for start, end in allergy_spans
            )

            if is_allergy:
                continue

            entities.append({
                "text": match.group(),
                "label": "Medication",
                "start": match.start(),
                "end": match.end(),
                "source": "lexicon",
            })

    return entities