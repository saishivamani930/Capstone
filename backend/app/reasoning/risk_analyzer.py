class RiskAnalyzer:
    def analyze_risk(self, present_symptoms: list, candidate_diseases: list) -> dict:
        symptoms_set = {s.lower() for s in present_symptoms}
        
        # High Risk Red Flags
        has_chest_pain = "chest pain" in symptoms_set
        has_sob = "shortness of breath" in symptoms_set
        
        risk_level = "LOW"
        risk_score = 0.2
        flagged_factors = []
        
        if has_chest_pain and has_sob:
            risk_level = "HIGH"
            risk_score = 0.85
            flagged_factors.append("Concurrent acute chest pain and dyspnea (Potential ACS / Pulmonary Embolism)")
        elif has_chest_pain:
            risk_level = "MEDIUM"
            risk_score = 0.60
            flagged_factors.append("Acute chest pain present")
        elif has_sob:
            risk_level = "MEDIUM"
            risk_score = 0.50
            flagged_factors.append("Dyspnea present")

        # Check candidate diseases for high-risk cardiac conditions
        high_risk_diseases = {"myocardial infarction", "angina pectoris", "pneumothorax"}
        for disease in candidate_diseases:
            d_name = disease.disease_name if hasattr(disease, "disease_name") else disease.get("disease_name", "")
            if d_name.lower() in high_risk_diseases:
                flagged_factors.append(f"High-risk candidate condition matched: {d_name.title()}")
                if risk_level != "HIGH":
                    risk_level = "HIGH"
                    risk_score = max(risk_score, 0.80)

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "flagged_factors": flagged_factors
        }