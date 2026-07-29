from app.reasoning.risk_analyzer import RiskAnalyzer


def test_high_risk_chest_pain_and_sob():
    analyzer = RiskAnalyzer()
    present_symptoms = ["chest pain", "shortness of breath"]
    candidate_diseases = [{"disease_name": "Pneumothorax"}]

    result = analyzer.analyze_risk(present_symptoms, candidate_diseases)

    assert result["risk_level"] == "HIGH"
    assert result["risk_score"] == 0.85
    assert len(result["flagged_factors"]) >= 1


def test_medium_risk_chest_pain_only():
    analyzer = RiskAnalyzer()
    present_symptoms = ["chest pain"]
    candidate_diseases = []

    result = analyzer.analyze_risk(present_symptoms, candidate_diseases)

    assert result["risk_level"] == "MEDIUM"
    assert result["risk_score"] == 0.60