def span_distance(first: dict, second: dict) -> int:
    if first["end"] < second["start"]:
        return second["start"] - first["end"]

    if second["end"] < first["start"]:
        return first["start"] - second["end"]

    return 0


def build_symptom_records(entities: list[dict]) -> list[dict]:
    symptoms = [
        entity
        for entity in entities
        if (
            entity["label"] == "Sign_symptom"
            and not entity.get("negated", False)
        )
    ]

    symptom_records = [
        {
            "symptom": symptom,
            "severities": [],
            "durations": [],
            "aggravating_factors": [],
        }
        for symptom in symptoms
    ]

    if not symptom_records:
        return []

    category_map = {
        "Severity": "severities",
        "Duration": "durations",
        "Aggravating_factor": "aggravating_factors",
    }

    modifiers = [
        entity
        for entity in entities
        if (
            entity["label"] in category_map
            and not entity.get("negated", False)
        )
    ]

    for modifier in modifiers:
        nearest_index = min(
            range(len(symptoms)),
            key=lambda index: span_distance(
                symptoms[index],
                modifier,
            ),
        )

        category = category_map[modifier["label"]]

        symptom_records[nearest_index][category].append(
            modifier
        )

    return symptom_records