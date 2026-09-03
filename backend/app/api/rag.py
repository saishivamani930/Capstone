from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app.rag.rag_engine import run_rag_pipeline, generate_dynamic_patient_prompts

router = APIRouter(
    prefix="/api/rag",
    tags=["RAG Guideline Evidence & Claim Validation"]
)


class RAGQueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    entities: Optional[List[Dict[str, Any]]] = None
    risk_analysis: Optional[Dict[str, Any]] = None


class DynamicPromptsRequest(BaseModel):
    entities: Optional[List[Dict[str, Any]]] = None
    risk_analysis: Optional[Dict[str, Any]] = None


@router.post("/query")
def query_rag_guidelines(request: RAGQueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question prompt cannot be empty.")
        return run_rag_pipeline(
            request.question, 
            top_k=request.top_k or 5,
            entities=request.entities,
            risk_analysis=request.risk_analysis
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG process error: {str(e)}")


@router.post("/prompts")
def fetch_dynamic_prompts(request: DynamicPromptsRequest):
    try:
        prompts = generate_dynamic_patient_prompts(request.entities, request.risk_analysis)
        return {"prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompts generation error: {str(e)}")

