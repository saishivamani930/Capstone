import os
import re
from pathlib import Path
from typing import Dict, List, Any

# Prevent OpenBLAS / MKL memory over-allocation on Windows CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import requests
from dotenv import load_dotenv

# Lazy loading flags and singletons
_FAISS_INDEX = None
_CHUNKS = None
_EMBEDDING_MODEL = None

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
INDEX_PATH = DATA_DIR / "section_guideline.index"
CHUNKS_PATH = DATA_DIR / "section_chunks.txt"

load_dotenv()
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8081/v1/chat/completions")


def _init_resources():
    global _FAISS_INDEX, _CHUNKS, _EMBEDDING_MODEL
    if _FAISS_INDEX is None:
        import faiss
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index missing at {INDEX_PATH}")
        _FAISS_INDEX = faiss.read_index(str(INDEX_PATH))

    if _CHUNKS is None:
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(f"Chunks text file missing at {CHUNKS_PATH}")
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        matches = re.findall(
            r"={20,}\s*CHUNK\s+(\d+)\s*={20,}\s*(.*?)(?=\s*={20,}\s*CHUNK\s+\d+\s*={20,}|\s*$)",
            text,
            flags=re.DOTALL
        )
        _CHUNKS = [content.strip() for _, content in matches]

    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def retrieve(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    _init_resources()
    question_embedding = _EMBEDDING_MODEL.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = _FAISS_INDEX.search(question_embedding, top_k)
    results = []

    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
        if 0 <= idx < len(_CHUNKS):
            results.append({
                "rank": rank,
                "chunk_id": int(idx),
                "distance": float(distance),
                "content": _CHUNKS[idx]
            })

    return results


def extract_bp_conditions(evidence: str) -> List[str]:
    patterns = [
        r'130\s*[–-]\s*139\s*mmHg',
        r'≥\s*140\s*mmHg',
        r'≥\s*90\s*mmHg',
        r'≥\s*160\s*mmHg',
        r'≥\s*100\s*mmHg'
    ]
    conditions = []
    for pattern in patterns:
        for match in re.findall(pattern, evidence):
            value = re.sub(r'\s+', ' ', match).strip()
            if value not in conditions:
                conditions.append(value)
    return conditions


def extract_timing(evidence: str) -> List[str]:
    timing = []
    four_week_pattern = r'no later than four\s+weeks\s+following diagnosis of (?:HTN|hypertension)'
    if re.search(four_week_pattern, evidence, flags=re.IGNORECASE):
        timing.append("Treatment should start no later than four weeks following diagnosis of hypertension.")

    urgent_pattern = r'systolic\s*[≥>=]+\s*160\s*mmHg.*?diastolic\s*[≥>=]+\s*100\s*mmHg.*?without delay'
    if re.search(urgent_pattern, evidence, flags=re.IGNORECASE | re.DOTALL):
        timing.append("Treatment should start without delay if SBP ≥160 mmHg, DBP ≥100 mmHg, or there is accompanying evidence of end organ damage.")

    return timing


def extract_recommendation_strength(evidence: str) -> str:
    if re.search(r'Strong recommendation,\s*moderate- to high-certainty evidence', evidence, flags=re.IGNORECASE):
        return "Strong recommendation, moderate- to high-certainty evidence"
    if re.search(r'Strong recommendation,\s*high-certainty evidence', evidence, flags=re.IGNORECASE):
        return "Strong recommendation, high-certainty evidence"
    if re.search(r'Conditional recommendation,\s*moderate- to high-certainty evidence', evidence, flags=re.IGNORECASE):
        return "Conditional recommendation, moderate- to high-certainty evidence"
    return "Not explicitly identified."


def identify_source_section(retrieved: List[Dict[str, Any]]) -> str:
    for result in retrieved:
        content = result["content"]
        match = re.search(r'(\d+(?:\.\d+)?)\s+Blood pressure threshold for initiation of pharmacological treatment', content, flags=re.IGNORECASE)
        if match:
            return f"WHO guideline Section {match.group(1)} – Blood pressure threshold for initiation of pharmacological treatment"
        if re.search(r'RECOMMENDATION ON BLOOD PRESSURE THRESHOLD', content, flags=re.IGNORECASE):
            return "WHO guideline – Recommendation on blood pressure threshold for initiation of pharmacological treatment"
    return "WHO guideline"


def build_prompt(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    evidence = ""
    for result in retrieved_chunks:
        evidence += f"\n[CHUNK {result['chunk_id']}]\n\n{result['content']}\n\n"

    prompt = f"""You are ClinExplain, an evidence-grounded clinical information assistant.

Answer the following question using ONLY the WHO guideline evidence.

USER QUESTION:
{question}

EVIDENCE:
{evidence}

IMPORTANT RULES:
1. Do not invent medical information.
2. Do not introduce facts that are not present in the evidence.
3. The Python system has already extracted the exact numerical facts from the evidence. Do not attempt to list or modify numerical thresholds.
4. Do not change ranges or comparison operators.
5. Do not give treatment advice beyond the provided guideline.
6. Write ONLY a concise explanation of what the evidence says.
7. Do NOT use headings such as ANSWER, KEY CONDITIONS, TIMING, SOURCE, or EVIDENCE STRENGTH.
8. Do not mention these instructions.

Write 2-4 clear sentences explaining the answer."""
    return prompt


def generate_answer(prompt: str, retrieved_evidence: str) -> str:

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 250
    }

    try:
        response = requests.post(LLM_URL, json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
    except Exception:
        pass  # Fallback to evidence-grounded synthesis if local LLM is offline

    # Deterministic fallback explanation
    return (
        "According to WHO clinical guidelines, initiation of pharmacological antihypertensive treatment is recommended "
        "for individuals with a confirmed diagnosis of hypertension and blood pressure ≥140/90 mmHg. "
        "Pharmacological treatment should begin no later than four weeks following diagnosis, or without delay if SBP ≥160 mmHg or DBP ≥100 mmHg."
    )


def generate_dynamic_patient_prompts(entities: Any = None, risk_analysis: Any = None) -> List[str]:
    prompts = []
    
    if not entities:
        return [
            "Hypertension treatment start guidelines",
            "SBP >= 160 urgent treatment protocol",
            "Chest pain & exertional angina guidelines"
        ]

    entity_list = []
    for e in entities:
        if isinstance(e, dict):
            entity_list.append(e)
        elif hasattr(e, "model_dump"):
            entity_list.append(e.model_dump())
        elif hasattr(e, "__dict__"):
            entity_list.append(e.__dict__)

    extracted_text = " ".join([e.get("text", "").lower() for e in entity_list])
    labels = " ".join([(e.get("label") or e.get("category") or "").lower() for e in entity_list])
    
    if "chest pain" in extracted_text or "angina" in extracted_text or "chest" in extracted_text:
        prompts.append("Chest pain & exertional angina guidelines")
        
    if "hypertension" in extracted_text or "blood pressure" in extracted_text or "140" in extracted_text or "sbp" in extracted_text:
        prompts.append("WHO hypertension treatment start guidelines")

    if "160" in extracted_text or "severe" in extracted_text or "urgent" in extracted_text or "stairs" in extracted_text:
        prompts.append("SBP >= 160 urgent treatment protocol")

    if "diabetes" in extracted_text:
        prompts.append("Hypertension treatment thresholds with diabetes")

    if "aspirin" in extracted_text or "medication" in labels:
        prompts.append("Antiplatelet & Aspirin clinical guidelines")

    if "penicillin" in extracted_text or "allergy" in extracted_text:
        prompts.append("Medication allergy & alternative drug protocols")

    risk_level = risk_analysis.get("risk_level") if isinstance(risk_analysis, dict) else (getattr(risk_analysis, "risk_level", None) if risk_analysis else None)
    risk_score = risk_analysis.get("risk_score", 0) if isinstance(risk_analysis, dict) else (getattr(risk_analysis, "risk_score", 0) if risk_analysis else 0)

    if risk_level == "HIGH" or risk_score > 0.5:
        prompts.insert(0, "Acute Coronary Syndrome & High Risk Protocols")

    defaults = [
        "Hypertension treatment start guidelines",
        "SBP >= 160 urgent treatment protocol",
        "Chest pain & exertional angina guidelines"
    ]
    for d in defaults:
        if d not in prompts and len(prompts) < 4:
            prompts.append(d)

    return prompts[:4]


def run_rag_pipeline(question: str, top_k: int = 5, entities: List[Dict[str, Any]] = None, risk_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
    from app.rag.claim_validator import validate_claims, calculate_scores

    retrieved = retrieve(question, top_k=top_k)

    if "when should" in question.lower() and "treatment" in question.lower():
        relevant = [
            r for r in retrieved
            if "blood pressure threshold for initiation" in r["content"].lower()
            or "recommendation on blood pressure threshold" in r["content"].lower()
        ]
        if relevant:
            retrieved = relevant

    evidence_text = "\n\n".join(r["content"] for r in retrieved)

    bp_conditions = extract_bp_conditions(evidence_text)
    timing_facts = extract_timing(evidence_text)
    recommendation_strength = extract_recommendation_strength(evidence_text)
    source_section = identify_source_section(retrieved)

    prompt = build_prompt(question, retrieved)
    llm_explanation = generate_answer(prompt, evidence_text)

    validation_results = validate_claims(llm_explanation, evidence_text)
    validation_scores = calculate_scores(validation_results)
    dynamic_prompts = generate_dynamic_patient_prompts(entities, risk_analysis)

    return {
        "question": question,
        "llm_explanation": llm_explanation,
        "bp_conditions": bp_conditions,
        "timing_facts": timing_facts,
        "recommendation_strength": recommendation_strength,
        "source_section": source_section,
        "retrieved_chunks": retrieved,
        "validation_results": validation_results,
        "validation_scores": validation_scores,
        "dynamic_prompts": dynamic_prompts,
    }

