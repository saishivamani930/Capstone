from pathlib import Path

import av
import numpy as np
import soundfile as sf


def convert_webm_to_wav(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000,
) -> None:
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError("Input audio file was not found")

    container = av.open(str(input_file))

    audio_stream = next(
        (
            stream
            for stream in container.streams
            if stream.type == "audio"
        ),
        None,
    )

    if audio_stream is None:
        raise ValueError("No audio stream found in recording")

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=sample_rate,
    )

    audio_parts = []

    for frame in container.decode(audio_stream):
        converted_frames = resampler.resample(frame)

        for converted_frame in converted_frames:
            audio_array = converted_frame.to_ndarray()
            audio_parts.append(audio_array.reshape(-1))

    container.close()

    if not audio_parts:
        raise ValueError("No audio data could be decoded")

    combined_audio = np.concatenate(audio_parts).astype(np.int16)

    sf.write(
        output_path,
        combined_audio,
        sample_rate,
        subtype="PCM_16",
    )