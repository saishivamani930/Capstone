def build_patient_text(conversation: list[dict]) -> str:
    patient_turns = []

    for item in conversation:
        if item.get("speaker") != "PATIENT":
            continue

        turn_text = item.get("text", "").strip()

        if not turn_text:
            continue

        # Every patient turn must be treated as a separate sentence.
        if turn_text[-1] not in ".?!":
            turn_text += "."

        patient_turns.append(turn_text)

    return " ".join(patient_turns)