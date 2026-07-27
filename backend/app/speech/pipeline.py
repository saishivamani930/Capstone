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

    patient_text = build_patient_text(conversation_with_roles)

    medical_nlp_result = run_medical_nlp(patient_text)

    result = {
    "transcript": transcription["text"],
    "language": transcription["language"],
    "patient_transcript": patient_text,
    "conversation": conversation_with_roles,
    "medical_nlp": medical_nlp_result,
    }

    validated_result = ConsultationResult.model_validate(result)

    return validated_result.model_dump()