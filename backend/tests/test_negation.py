from app.medical_nlp.negation import apply_negation


def test_disease_is_negated():
    text = "I do not have diabetes."

    entities = [
        {
            "text": "diabetes",
            "label": "Disease_disorder",
            "start": 14,
            "end": 22,
            "source": "lexicon",
        }
    ]

    result = apply_negation(text, entities)

    assert len(result) == 1
    assert result[0]["text"] == "diabetes"
    assert result[0]["negated"] is True
    assert result[0]["assertion"] == "absent"
    assert result[0]["negation_evidence"] == "do not have"


def test_allergy_is_negated():
    text = "I am not allergic to penicillin."

    entities = [
        {
            "text": "penicillin",
            "label": "Allergy",
            "start": 21,
            "end": 31,
            "source": "rule",
        }
    ]

    result = apply_negation(text, entities)

    assert result[0]["negated"] is True
    assert result[0]["assertion"] == "absent"


def test_present_symptom_is_not_negated():
    text = "I have chest pain."

    entities = [
        {
            "text": "chest pain",
            "label": "Sign_symptom",
            "start": 7,
            "end": 17,
            "source": "normalized",
        }
    ]

    result = apply_negation(text, entities)

    assert result[0]["negated"] is False
    assert result[0]["assertion"] == "present"
    assert "negation_evidence" not in result[0]


def test_negation_does_not_cross_sentence_boundary():
    text = "I do not have diabetes. I have chest pain."

    entities = [
        {
            "text": "diabetes",
            "label": "Disease_disorder",
            "start": 14,
            "end": 22,
            "source": "lexicon",
        },
        {
            "text": "chest pain",
            "label": "Sign_symptom",
            "start": 31,
            "end": 41,
            "source": "normalized",
        },
    ]

    result = apply_negation(text, entities)

    assert result[0]["negated"] is True
    assert result[1]["negated"] is False


def test_negation_does_not_cross_but_boundary():
    text = "I do not have diabetes, but I have chest pain."

    entities = [
        {
            "text": "diabetes",
            "label": "Disease_disorder",
            "start": 14,
            "end": 22,
            "source": "lexicon",
        },
        {
            "text": "chest pain",
            "label": "Sign_symptom",
            "start": 35,
            "end": 45,
            "source": "normalized",
        },
    ]

    result = apply_negation(text, entities)

    assert result[0]["negated"] is True
    assert result[1]["negated"] is False


def test_contracted_allergy_negation():
    text = "I'm not allergic to penicillin."

    entities = [
        {
            "text": "penicillin",
            "label": "Allergy",
            "start": 20,
            "end": 30,
            "source": "rule",
        }
    ]

    result = apply_negation(text, entities)

    assert result[0]["negated"] is True
    assert result[0]["assertion"] == "absent"


def test_negation_does_not_cross_new_assertion():
    text = (
        "I'm not allergic to penicillin "
        "I take aspirin every morning."
    )

    entities = [
        {
            "text": "penicillin",
            "label": "Allergy",
            "start": 20,
            "end": 30,
            "source": "rule",
        },
        {
            "text": "aspirin",
            "label": "Medication",
            "start": 38,
            "end": 45,
            "source": "lexicon",
        },
    ]

    result = apply_negation(text, entities)

    assert result[0]["negated"] is True
    assert result[1]["negated"] is False