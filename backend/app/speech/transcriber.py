from functools import lru_cache

from faster_whisper import WhisperModel


MEDICAL_HOTWORDS = (
    "metformin diabetes hypertension angina aspirin "
    "atorvastatin amlodipine nitroglycerin chest pain "
    "palpitations dyspnea tachycardia bradycardia"
)


@lru_cache(maxsize=1)
def get_whisper_model():
    """
    Accurate model used for the final transcript.
    """
    return WhisperModel(
        "small.en",
        device="cpu",
        compute_type="int8",
    )


@lru_cache(maxsize=1)
def get_partial_whisper_model():
    """
    Faster model used only for temporary live updates.
    """
    return WhisperModel(
        "tiny.en",
        device="cpu",
        compute_type="int8",
    )


def transcribe_audio(audio_path: str) -> dict:
    model = get_whisper_model()

    segments, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        hotwords=MEDICAL_HOTWORDS,
    )

    segments = list(segments)

    transcript = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    words = []

    for segment in segments:
        if not segment.words:
            continue

        for word in segment.words:
            words.append({
                "word": word.word.strip(),
                "start": round(word.start, 2),
                "end": round(word.end, 2),
            })

    return {
        "text": transcript,
        "language": info.language,
        "language_probability": round(
            info.language_probability,
            4,
        ),
        "words": words,
    }


def transcribe_partial_audio(audio_path: str) -> dict:
    """
    Fast temporary transcription while recording.

    This does not generate word timestamps because the
    partial transcript is not used for diarisation.
    """
    model = get_partial_whisper_model()

    segments, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=1,
        best_of=1,
        vad_filter=True,
        word_timestamps=False,
        condition_on_previous_text=False,
        hotwords=MEDICAL_HOTWORDS,
    )

    transcript = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    return {
        "text": transcript,
        "language": info.language,
        "language_probability": round(
            info.language_probability,
            4,
        ),
    }