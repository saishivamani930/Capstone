import re

from app.medical_nlp.clinical_rules import (
    extract_clinical_rules,
    spans_overlap,
)


DURATION_PATTERN = re.compile(
    r"\b(?:for\s+)?(?:the\s+)?(?:last\s+)?"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s+"
    r"(?:second|seconds|minute|minutes|hour|hours|day|days|"
    r"week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)


def normalize_entities(text: str, entities: list[dict]) -> list[dict]:
    entities = sorted(entities, key=lambda item: item["start"])
    normalized = []

    i = 0

    while i < len(entities):
        current = entities[i]

        if i + 1 < len(entities):
            next_entity = entities[i + 1]
            between = text[current["end"]:next_entity["start"]]

            if (
                current["label"] == "Biological_structure"
                and next_entity["label"] == "Sign_symptom"
                and between.strip() == ""
            ):
                normalized.append({
                    "text": text[current["start"]:next_entity["end"]],
                    "label": "Sign_symptom",
                    "start": current["start"],
                    "end": next_entity["end"],
                    "source": "normalized",
                })

                i += 2
                continue

        normalized.append({
            **current,
            "source": "model",
        })

        i += 1

    # Replace incomplete model duration with complete duration phrase.
    normalized = [
        entity
        for entity in normalized
        if entity["label"] != "Duration"
    ]

    for match in DURATION_PATTERN.finditer(text):
        normalized.append({
            "text": match.group(),
            "label": "Duration",
            "start": match.start(),
            "end": match.end(),
            "source": "rule",
        })

    # Add entities found using controlled clinical rules.
    rule_entities = extract_clinical_rules(text)

    for candidate in rule_entities:
        if candidate["label"] == "Allergy":
            normalized = [
                entity
                for entity in normalized
                if not (
                    entity["label"] == "Medication"
                    and spans_overlap(entity, candidate)
                )
            ]

        duplicate = any(
            entity["label"] == candidate["label"]
            and spans_overlap(entity, candidate)
            for entity in normalized
        )

        if not duplicate:
            normalized.append(candidate)

    return sorted(normalized, key=lambda item: item["start"])