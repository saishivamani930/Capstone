from app.medical_nlp import pipeline


def test_pipeline_builds_structured_clinical_information(monkeypatch):
    text = (
        "I have severe chest pain for three days. "
        "It becomes worse when I climb stairs."
    )

    mock_raw_entities = [
        {
            "text": "severe",
            "label": "Severity",
            "confidence": 0.99,
            "start": 7,
            "end": 13,
        },
        {
            "text": "chest",
            "label": "Biological_structure",
            "confidence": 0.99,
            "start": 14,
            "end": 19,
        },
        {
            "text": "pain",
            "label": "Sign_symptom",
            "confidence": 0.99,
            "start": 20,
            "end": 24,
        },
        {
            "text": "three days",
            "label": "Duration",
            "confidence": 0.99,
            "start": 29,
            "end": 39,
        },
    ]

    monkeypatch.setattr(
        pipeline,
        "extract_medical_entities",
        lambda _: mock_raw_entities,
    )

    result = pipeline.run_medical_nlp(text)
    structured = result["structured_entities"]

    assert structured["symptoms"][0]["text"] == "chest pain"
    assert structured["severities"][0]["text"] == "severe"
    assert structured["durations"][0]["text"] == "for three days"

    assert (
        structured["aggravating_factors"][0]["text"]
        == "climb stairs"
    )

    record = result["clinical_facts"]["symptom_records"][0]

    assert record["symptom"]["text"] == "chest pain"
    assert record["severities"][0]["text"] == "severe"
    assert record["durations"][0]["text"] == "for three days"

    assert (
        record["aggravating_factors"][0]["text"]
        == "climb stairs"
    )


def test_pipeline_separates_present_and_negated_entities(
    monkeypatch,
):
    text = (
        "I do not have diabetes. "
        "I have chest pain."
    )

    mock_raw_entities = [
        {
            "text": "chest",
            "label": "Biological_structure",
            "confidence": 0.99,
            "start": 31,
            "end": 36,
        },
        {
            "text": "pain",
            "label": "Sign_symptom",
            "confidence": 0.99,
            "start": 37,
            "end": 41,
        },
    ]

    monkeypatch.setattr(
        pipeline,
        "extract_medical_entities",
        lambda _: mock_raw_entities,
    )

    result = pipeline.run_medical_nlp(text)
    structured = result["structured_entities"]

    assert structured["diseases"] == []

    assert len(structured["negated_entities"]) == 1
    assert structured["negated_entities"][0]["text"] == "diabetes"
    assert structured["negated_entities"][0]["negated"] is True

    assert len(structured["symptoms"]) == 1
    assert structured["symptoms"][0]["text"] == "chest pain"
    assert structured["symptoms"][0]["negated"] is False


def test_pipeline_filters_unsupported_model_labels(monkeypatch):
    text = "Yes, I climb stairs."

    mock_raw_entities = [
        {
            "text": "Yes",
            "label": "Lab_value",
            "confidence": 0.90,
            "start": 0,
            "end": 3,
        },
        {
            "text": "climb",
            "label": "Activity",
            "confidence": 0.80,
            "start": 7,
            "end": 12,
        },
    ]

    monkeypatch.setattr(
        pipeline,
        "extract_medical_entities",
        lambda _: mock_raw_entities,
    )

    result = pipeline.run_medical_nlp(text)

    assert result["entities"] == []
    assert result["structured_entities"]["other_entities"] == []


def test_pipeline_removes_false_allergic_symptom(monkeypatch):
    text = "I am not allergic to penicillin."

    mock_raw_entities = [
        {
            "text": "allergic",
            "label": "Sign_symptom",
            "confidence": 0.99,
            "start": 9,
            "end": 17,
        },
        {
            "text": "penicillin",
            "label": "Medication",
            "confidence": 0.99,
            "start": 21,
            "end": 31,
        },
    ]

    monkeypatch.setattr(
        pipeline,
        "extract_medical_entities",
        lambda _: mock_raw_entities,
    )

    result = pipeline.run_medical_nlp(text)
    structured = result["structured_entities"]

    symptom_texts = [
        entity["text"]
        for entity in structured["symptoms"]
    ]

    assert "allergic" not in symptom_texts

    negated = structured["negated_entities"]

    assert len(negated) == 1
    assert negated[0]["text"] == "penicillin"
    assert negated[0]["label"] == "Allergy"
    assert negated[0]["negated"] is True