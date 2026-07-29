from pydantic import BaseModel
from typing import List, Optional

class CandidateDisease(BaseModel):
    wikidata_id: str
    disease_name: str
    matched_symptoms: List[str]
    match_count: int

class MissingInformation(BaseModel):
    disease_name: str
    wikidata_id: str
    missing_symptoms: List[str]
    suggested_questions: List[str]

class GraphNode(BaseModel):
    id: str
    label: str
    category: str
    status: Optional[str] = "present"

class GraphRelationship(BaseModel):
    source: str
    target: str
    type: str

class RiskAnalysisResult(BaseModel):
    risk_level: str  # "HIGH", "MEDIUM", "LOW"
    risk_score: float
    flagged_factors: List[str]

class ReasoningResponse(BaseModel):
    patient_id: str
    graph_nodes: List[GraphNode]
    graph_relationships: List[GraphRelationship]
    candidate_diseases: List[CandidateDisease]
    missing_information: List[MissingInformation]
    risk_analysis: Optional[RiskAnalysisResult] = None