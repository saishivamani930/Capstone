import unittest
from app.medical_nlp.pipeline import run_medical_nlp
from app.reasoning.missing_info import MissingInfoEngine
from app.reasoning.risk_analyzer import RiskAnalyzer

class TestLiveMissingInfo(unittest.TestCase):
    def test_missing_info_chest_pain_rules(self):
        engine = MissingInfoEngine()
        symptoms = ["chest pain"]
        missing_info = engine.analyze_sync(symptoms, candidate_diseases=[])
        
        self.assertTrue(len(missing_info) > 0)
        item = missing_info[0]
        self.assertIn("radiation of pain", item["missing_symptoms"])
        self.assertTrue(any("left arm" in q for q in item["suggested_questions"]))

    def test_missing_info_dyspnea_rules(self):
        engine = MissingInfoEngine()
        symptoms = ["shortness of breath"]
        missing_info = engine.analyze_sync(symptoms, candidate_diseases=[])
        
        self.assertTrue(len(missing_info) > 0)
        item = missing_info[0]
        self.assertIn("orthopnea", item["missing_symptoms"])
        self.assertTrue(any("lying flat" in q for q in item["suggested_questions"]))

    def test_risk_analyzer_acute_red_flags(self):
        analyzer = RiskAnalyzer()
        res = analyzer.analyze_risk(["chest pain", "shortness of breath"], candidate_diseases=[])
        
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertGreaterEqual(res["risk_score"], 0.85)
        self.assertTrue(len(res["flagged_factors"]) > 0)

    def test_live_text_nlp_and_missing_info_pipeline(self):
        sample_partial = "Patient reports severe chest pain and shortness of breath for 2 days. No fever."
        nlp_res = run_medical_nlp(sample_partial)
        
        structured_entities = nlp_res.get("structured_entities", {})
        symptoms = structured_entities.get("symptoms", [])
        present_symptoms = [s["text"] for s in symptoms if not s.get("negated", False)]
        
        engine = MissingInfoEngine()
        missing_info = engine.analyze_sync(present_symptoms)
        
        self.assertTrue(len(missing_info) > 0)
        self.assertTrue("chest pain" in present_symptoms or any("chest" in s for s in present_symptoms))

if __name__ == "__main__":
    unittest.main()
