def map_speaker_roles(conversation: list[dict]) -> list[dict]:
    speaker_order = []

    for item in conversation:
        speaker = item["speaker"]

        if speaker == "UNKNOWN":
            continue

        if speaker not in speaker_order:
            speaker_order.append(speaker)

    if len(speaker_order) < 2:
        raise ValueError(
            "Two different speakers are required: Doctor and Patient."
        )

    role_map = {
        speaker_order[0]: "DOCTOR",
        speaker_order[1]: "PATIENT",
    }

    result = []

    for item in conversation:
        result.append({
            "speaker": role_map.get(item["speaker"], "UNKNOWN"),
            "raw_speaker": item["speaker"],
            "start": item["start"],
            "end": item["end"],
            "text": item["text"],
        })

    return result