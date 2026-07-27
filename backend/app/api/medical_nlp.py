from fastapi import APIRouter
from pydantic import BaseModel

from app.medical_nlp.pipeline import run_medical_nlp


router = APIRouter(
    prefix="/medical-nlp",
    tags=["Medical NLP"],
)


class ClinicalTextRequest(BaseModel):
    text: str


@router.post("/extract")
def extract_entities(request: ClinicalTextRequest):
    return run_medical_nlp(request.text)