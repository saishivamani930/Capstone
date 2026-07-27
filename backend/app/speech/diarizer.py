import os
from functools import lru_cache
from pathlib import Path

import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def get_diarization_pipeline():
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing from the .env file")

    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )


def diarize_audio(audio_path: str) -> list[dict]:
    pipeline = get_diarization_pipeline()

    # Load WAV ourselves so pyannote does not depend on TorchCodec.
    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )

    # soundfile gives: (time, channels)
    # pyannote expects: (channels, time)
    waveform = torch.from_numpy(audio.T)

    output = pipeline(
    {
        "waveform": waveform,
        "sample_rate": sample_rate,
    },
     num_speakers=2,
)

    segments = []

    for turn, speaker in output.exclusive_speaker_diarization:
        segments.append(
            {
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
            }
        )

    return segments