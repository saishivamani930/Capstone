from pydantic import BaseModel, Field


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
    diagnostic_tests: list[ClinicalEntity] = Field(
        default_factory=list
    )
    procedures: list[ClinicalEntity] = Field(default_factory=list)
    family_history: list[ClinicalEntity] = Field(
        default_factory=list
    )
    aggravating_factors: list[ClinicalEntity] = Field(
        default_factory=list
    )
    negated_entities: list[ClinicalEntity] = Field(
        default_factory=list
    )
    other_entities: list[ClinicalEntity] = Field(
        default_factory=list
    )


class SymptomRecord(BaseModel):
    symptom: ClinicalEntity
    severities: list[ClinicalEntity] = Field(default_factory=list)
    durations: list[ClinicalEntity] = Field(default_factory=list)
    aggravating_factors: list[ClinicalEntity] = Field(
        default_factory=list
    )


class ClinicalFacts(BaseModel):
    symptom_records: list[SymptomRecord] = Field(
        default_factory=list
    )


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