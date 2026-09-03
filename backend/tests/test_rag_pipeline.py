import unittest
from app.rag.rag_engine import run_rag_pipeline


class TestRAGPipeline(unittest.TestCase):
    def test_hypertension_treatment_query(self):
        res = run_rag_pipeline("When should pharmacological treatment start for hypertension?")
        self.assertIn("llm_explanation", res)
        self.assertIn("validation_scores", res)
        self.assertGreaterEqual(res["validation_scores"]["faithfulness"], 0.0)
        self.assertTrue(len(res["retrieved_chunks"]) > 0)


if __name__ == "__main__":
    unittest.main()
