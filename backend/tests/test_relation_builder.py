from app.medical_nlp.relation_builder import build_symptom_records


def test_builds_complete_symptom_record():
    entities = [
        {
            "text": "severe",
            "label": "Severity",
            "start": 7,
            "end": 13,
            "negated": False,
        },
        {
            "text": "chest pain",
            "label": "Sign_symptom",
            "start": 14,
            "end": 24,
            "negated": False,
        },
        {
            "text": "for three days",
            "label": "Duration",
            "start": 25,
            "end": 39,
            "negated": False,
        },
        {
            "text": "climb stairs",
            "label": "Aggravating_factor",
            "start": 63,
            "end": 77,
            "negated": False,
        },
    ]

    records = build_symptom_records(entities)

    assert len(records) == 1
    assert records[0]["symptom"]["text"] == "chest pain"

    assert len(records[0]["severities"]) == 1
    assert records[0]["severities"][0]["text"] == "severe"

    assert len(records[0]["durations"]) == 1
    assert records[0]["durations"][0]["text"] == "for three days"

    assert len(records[0]["aggravating_factors"]) == 1
    assert (
        records[0]["aggravating_factors"][0]["text"]
        == "climb stairs"
    )


def test_returns_empty_when_no_symptom_exists():
    entities = [
        {
            "text": "for three days",
            "label": "Duration",
            "start": 5,
            "end": 19,
            "negated": False,
        }
    ]

    records = build_symptom_records(entities)

    assert records == []


def test_ignores_negated_symptom():
    entities = [
        {
            "text": "chest pain",
            "label": "Sign_symptom",
            "start": 10,
            "end": 20,
            "negated": True,
        },
        {
            "text": "for three days",
            "label": "Duration",
            "start": 21,
            "end": 35,
            "negated": False,
        },
    ]

    records = build_symptom_records(entities)

    assert records == []


def test_ignores_negated_modifier():
    entities = [
        {
            "text": "chest pain",
            "label": "Sign_symptom",
            "start": 10,
            "end": 20,
            "negated": False,
        },
        {
            "text": "severe",
            "label": "Severity",
            "start": 2,
            "end": 8,
            "negated": True,
        },
    ]

    records = build_symptom_records(entities)

    assert len(records) == 1
    assert records[0]["severities"] == []


def test_modifier_attaches_to_nearest_symptom():
    entities = [
        {
            "text": "chest pain",
            "label": "Sign_symptom",
            "start": 7,
            "end": 17,
            "negated": False,
        },
        {
            "text": "headache",
            "label": "Sign_symptom",
            "start": 40,
            "end": 48,
            "negated": False,
        },
        {
            "text": "since yesterday",
            "label": "Duration",
            "start": 49,
            "end": 64,
            "negated": False,
        },
    ]

    records = build_symptom_records(entities)

    assert len(records) == 2
    assert records[0]["durations"] == []

    assert len(records[1]["durations"]) == 1
    assert records[1]["durations"][0]["text"] == "since yesterday"