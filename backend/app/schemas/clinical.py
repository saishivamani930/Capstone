from pydantic import BaseModel, Field
from typing import Optional, List


class ClinicalEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int

    source: str | None = None
    confidence: float | None = None

    negated: bool = False
    assertion: str = "present"

    evidence: str | None = None
    negation_evidence: str | None = None


class ConversationTurn(BaseModel):
    speaker: str
    raw_speaker: str
    start: float
    end: float
    text: str


class StructuredEntities(BaseModel):
    symptoms: list[ClinicalEntity] = Field(default_factory=list)
    diseases: list[ClinicalEntity] = Field(default_factory=list)
    medications: list[ClinicalEntity] = Field(default_factory=list)
    allergies: list[ClinicalEntity] = Field(default_factory=list)
    durations: list[ClinicalEntity] = Field(default_factory=list)
    severities: list[ClinicalEntity] = Field(default_factory=list)
    diagnostic_tests: list[ClinicalEntity] = Field(default_factory=list)
    procedures: list[ClinicalEntity] = Field(default_factory=list)
    family_history: list[ClinicalEntity] = Field(default_factory=list)
    aggravating_factors: list[ClinicalEntity] = Field(default_factory=list)
    negated_entities: list[ClinicalEntity] = Field(default_factory=list)
    other_entities: list[ClinicalEntity] = Field(default_factory=list)


class SymptomRecord(BaseModel):
    symptom: ClinicalEntity
    severities: list[ClinicalEntity] = Field(default_factory=list)
    durations: list[ClinicalEntity] = Field(default_factory=list)
    aggravating_factors: list[ClinicalEntity] = Field(default_factory=list)


class ClinicalFacts(BaseModel):
    symptom_records: list[SymptomRecord] = Field(default_factory=list)


class MedicalNLPResult(BaseModel):
    input_text: str
    raw_entities: list[ClinicalEntity]
    entities: list[ClinicalEntity]
    structured_entities: StructuredEntities
    clinical_facts: ClinicalFacts


class ConsultationResult(BaseModel):
    transcript: str
    language: str
    patient_transcript: str
    conversation: list[ConversationTurn]
    medical_nlp: MedicalNLPResult
    rag_result: Optional[dict] = None


class ExtractedEntity(BaseModel):
    id: str
    text: str
    category: str
    status: Optional[str] = "present"
    severity: Optional[str] = None
    duration: Optional[str] = None
    wikidata_id: Optional[str] = None


class ExtractedRelation(BaseModel):
    source_id: str
    target_id: str
    relation_type: str


class ClinicalEntitiesResponse(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


# ==========================================
# Member 2 - Knowledge Graph & Reasoning Models
# ==========================================

class CandidateDisease(BaseModel):
    wikidata_id: str
    disease_name: str
    matched_symptoms: List[str] = Field(default_factory=list)
    match_count: int


class MissingInfoAlert(BaseModel):
    disease_name: str
    wikidata_id: str
    missing_symptoms: List[str] = Field(default_factory=list)
    suggested_questions: List[str] = Field(default_factory=list)


class RiskAnalysisResult(BaseModel):
    risk_level: str  # "HIGH", "MEDIUM", "LOW"
    risk_score: float
    flagged_factors: List[str] = Field(default_factory=list)


class ReasoningResponse(BaseModel):
    patient_id: str
    graph_nodes: List[dict] = Field(default_factory=list)
    graph_relationships: List[dict] = Field(default_factory=list)
    candidate_diseases: List[CandidateDisease] = Field(default_factory=list)
    missing_information: List[MissingInfoAlert] = Field(default_factory=list)
    risk_analysis: Optional[RiskAnalysisResult] = None