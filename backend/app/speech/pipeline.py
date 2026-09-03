from app.medical_nlp.clinical_text import build_patient_text
from app.speech.diarizer import diarize_audio
from app.speech.role_mapper import map_speaker_roles
from app.speech.transcriber import transcribe_audio

from app.medical_nlp.pipeline import run_medical_nlp

from app.schemas.clinical import ConsultationResult

def find_speaker(
    word_start: float,
    word_end: float,
    speaker_segments: list[dict]
) -> str:

    best_speaker = None
    best_overlap = 0.0

    for segment in speaker_segments:
        overlap_start = max(word_start, segment["start"])
        overlap_end = min(word_end, segment["end"])
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment["speaker"]

    if best_speaker is not None:
        return best_speaker

    nearest_speaker = "UNKNOWN"
    nearest_distance = float("inf")

    for segment in speaker_segments:
        if word_end <= segment["start"]:
            distance = segment["start"] - word_end
        elif word_start >= segment["end"]:
            distance = word_start - segment["end"]
        else:
            distance = 0.0

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_speaker = segment["speaker"]

    if nearest_distance <= 0.75:
        return nearest_speaker

    return "UNKNOWN"


import re

DOCTOR_QUESTION_PATTERNS = [
    r"\bdo you have\b",
    r"\bare you allergic\b",
    r"\bwhat problem\b",
    r"\bhow long\b",
    r"\bany allergy\b",
    r"\bdo you take\b",
    r"\bany other symptoms\b",
]


def refine_conversation_turns(conversation: list[dict]) -> list[dict]:
    refined = []

    for turn in conversation:
        text = turn["text"]
        speaker = turn["speaker"]

        if speaker == "PATIENT":
            doc_pattern = "|".join(DOCTOR_QUESTION_PATTERNS)
            match = re.search(doc_pattern, text, re.IGNORECASE)

            if match:
                split_idx = match.start()
                pre_patient_text = text[:split_idx].strip()
                post_text = text[split_idx:].strip()

                q_end = post_text.find("?")
                if q_end != -1:
                    doc_question = post_text[:q_end + 1].strip()
                    post_patient_text = post_text[q_end + 1:].strip()
                else:
                    # Look for end of sentence if no question mark
                    sentences = re.split(r"(?<=[.!?])\s+", post_text, maxsplit=1)
                    doc_question = sentences[0].strip()
                    post_patient_text = sentences[1].strip() if len(sentences) > 1 else ""

                if pre_patient_text:
                    refined.append({**turn, "speaker": "PATIENT", "text": pre_patient_text})
                if doc_question:
                    refined.append({**turn, "speaker": "DOCTOR", "text": doc_question})
                if post_patient_text:
                    refined.append({**turn, "speaker": "PATIENT", "text": post_patient_text})
                continue

        refined.append(turn)

    return refined


def process_conversation(audio_path: str) -> dict:
    transcription = transcribe_audio(audio_path)
    speaker_segments = diarize_audio(audio_path)

    labelled_words = []

    for word in transcription["words"]:
        speaker = find_speaker(
            word["start"],
            word["end"],
            speaker_segments
        )

        labelled_words.append({
            "speaker": speaker,
            "word": word["word"],
            "start": word["start"],
            "end": word["end"]
        })

    conversation = []

    for item in labelled_words:
        if (
            not conversation
            or conversation[-1]["speaker"] != item["speaker"]
        ):
            conversation.append({
                "speaker": item["speaker"],
                "start": item["start"],
                "end": item["end"],
                "text": item["word"]
            })
        else:
            conversation[-1]["text"] += " " + item["word"]
            conversation[-1]["end"] = item["end"]

    conversation_with_roles = map_speaker_roles(conversation)
    refined_conversation = refine_conversation_turns(conversation_with_roles)

    patient_text = build_patient_text(refined_conversation)

    medical_nlp_result = run_medical_nlp(patient_text)

    # Automatic RAG Guideline Evidence & Claim Verification
    from app.rag.rag_engine import run_rag_pipeline
    symptoms = [
        s["text"] for s in medical_nlp_result.get("structured_entities", {}).get("symptoms", [])
        if not s.get("negated", False)
    ]
    query_text = f"{' '.join(symptoms)} guidelines" if symptoms else "Chest pain & exertional angina guidelines"
    rag_result = run_rag_pipeline(
        query_text, 
        top_k=5, 
        entities=medical_nlp_result.get("entities", [])
    )

    result = {
        "transcript": transcription["text"],
        "language": transcription["language"],
        "patient_transcript": patient_text,
        "conversation": refined_conversation,
        "medical_nlp": medical_nlp_result,
        "rag_result": rag_result,
    }

    validated_result = ConsultationResult.model_validate(result)

    return validated_result.model_dump()