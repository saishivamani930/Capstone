# ClinExplain — AI Speech & Medical NLP Integration Guide

## 1. Module purpose

This module converts a live Doctor–Patient consultation into structured clinical information.

```text
Live microphone audio
→ WebSocket audio chunks
→ Partial live transcription
→ Final Whisper transcription
→ Speaker diarisation
→ Doctor/Patient role mapping
→ Patient-only transcript
→ Medical entity extraction
→ Negation detection
→ Clinical relation building
→ Validated JSON output
```

The output is designed for the Patient Knowledge Graph and symbolic reasoning modules.

## 2. Current implementation status

| Capability | Status |
|---|---|
| Live microphone recording | Completed |
| WebSocket audio streaming | Completed |
| Partial live transcript | Completed |
| Final Whisper transcription | Completed |
| Speaker diarisation | Completed |
| Doctor/Patient role mapping | Completed |
| Patient-only transcript | Completed |
| Medical entity extraction | Completed |
| Negation handling | Completed |
| Symptom relation building | Completed |
| Pydantic JSON validation | Completed |
| Automated tests | 20 passed |
| Real two-person validation | Passed |

## 3. Prototype assumptions

1. The consultation contains exactly two speakers.
2. The Doctor speaks first.
3. The first detected speaker is mapped to `DOCTOR`.
4. The second detected speaker is mapped to `PATIENT`.
5. Overlapping speech should be avoided.
6. A short pause between speakers improves diarisation.
7. Partial transcripts are temporary.
8. The final transcript is authoritative.

## 4. Models and processing modes

### Partial live transcription

```text
Model: tiny.en
Device: CPU
Compute type: int8
Purpose: fast temporary transcript updates
```

### Final transcription

```text
Model: small.en
Device: CPU
Compute type: int8
Purpose: accurate final transcript with word timestamps
```

### Speaker diarisation

```text
Model: pyannote/speaker-diarization-community-1
Expected speakers: 2
```

### Medical NER

```text
Model: d4data/biomedical-ner-all
```

Rule-based post-processing is used for symptom normalisation, durations, diseases, medications, allergies, aggravating factors, negation, and clinical relation building.

## 5. Backend API

Base URL:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

### `POST /speech/upload`
Confirms that an audio file was received.

### `POST /speech/transcribe`
Runs Whisper transcription only.

Supported formats:

```text
.wav
.mp3
.m4a
.flac
.ogg
```

### `POST /speech/diarize`
Separates the uploaded recording into speaker segments.

### `POST /speech/process`
Runs the complete consultation pipeline.

### `POST /medical-nlp/extract`
Extracts structured clinical information from supplied text.

Example request:

```json
{
  "text": "I have severe chest pain for three days. I do not have diabetes."
}
```

## 6. Live WebSocket API

WebSocket URL:

```text
ws://127.0.0.1:8000/speech/live
```

### Client-to-server

- Binary WebM/Opus audio chunks
- Text control message: `stop`

### Server-to-client events

#### Connected

```json
{
  "event": "connected",
  "session_id": "unique-session-id"
}
```

#### Chunk received

```json
{
  "event": "chunk_received",
  "chunk_number": 6,
  "chunk_size_bytes": 16438,
  "total_bytes": 95000
}
```

#### Partial transcript

```json
{
  "event": "partial_transcript",
  "session_id": "unique-session-id",
  "through_chunk": 12,
  "transcript": "temporary live transcript",
  "language": "en",
  "is_final": false
}
```

#### Processing

```json
{
  "event": "processing",
  "message": "Processing consultation audio."
}
```

#### Completed

```json
{
  "event": "completed",
  "session_id": "unique-session-id",
  "chunk_count": 26,
  "total_bytes": 410000,
  "is_final": true,
  "result": {}
}
```

#### Error

```json
{
  "event": "error",
  "message": "Two different speakers are required: Doctor and Patient.",
  "error_type": "ValueError"
}
```

## 7. Final consultation JSON

```json
{
  "transcript": "complete consultation transcript",
  "language": "en",
  "patient_transcript": "patient-only transcript",
  "conversation": [],
  "medical_nlp": {
    "input_text": "patient-only transcript",
    "raw_entities": [],
    "entities": [],
    "structured_entities": {},
    "clinical_facts": {}
  }
}
```

## 8. Field guide

### `transcript`
Complete Doctor–Patient transcript.

### `patient_transcript`
Only Patient turns, separated into sentences. This is the main Medical NLP input.

### `conversation`
Speaker-separated consultation turns.

```json
{
  "speaker": "PATIENT",
  "raw_speaker": "SPEAKER_00",
  "start": 5.2,
  "end": 9.94,
  "text": "I have severe chest pain for the last three days"
}
```

### `raw_entities`
Direct biomedical NER output. Use mainly for debugging, explainability, and model evaluation. Do not use it directly as final clinical truth.

### `entities`
Cleaned and normalised entities after rule processing, filtering, context extraction, and negation.

### `structured_entities`

```json
{
  "symptoms": [],
  "diseases": [],
  "medications": [],
  "allergies": [],
  "durations": [],
  "severities": [],
  "diagnostic_tests": [],
  "procedures": [],
  "family_history": [],
  "aggravating_factors": [],
  "negated_entities": [],
  "other_entities": []
}
```

### `clinical_facts`
Connected clinical observations.

```json
{
  "symptom_records": [
    {
      "symptom": {"text": "chest pain", "label": "Sign_symptom"},
      "severities": [{"text": "severe"}],
      "durations": [{"text": "for the last three days"}],
      "aggravating_factors": [{"text": "climb stairs"}]
    }
  ]
}
```

## 9. Knowledge Graph integration

The Knowledge Graph module should primarily consume:

```text
result.medical_nlp.structured_entities
result.medical_nlp.clinical_facts.symptom_records
```

### Recommended mapping

| JSON category | Suggested graph representation |
|---|---|
| `symptoms` | Symptom node |
| `diseases` | Disease node |
| `medications` | Medication node |
| `allergies` | Allergy node |
| `durations` | Duration property |
| `severities` | Severity property |
| `aggravating_factors` | `WORSENED_BY` relation |
| `diagnostic_tests` | DiagnosticTest node |
| `procedures` | Procedure node |
| `family_history` | FamilyHistory relation |
| `negated_entities` | Negative or absent fact |

### Example graph facts

```text
Patient ──HAS_SYMPTOM──> Chest Pain
Chest Pain ──HAS_SEVERITY──> Severe
Chest Pain ──HAS_DURATION──> Three Days
Chest Pain ──WORSENED_BY──> Climbing Stairs
Patient ──TAKES_MEDICATION──> Aspirin
Patient ──DOES_NOT_HAVE──> Diabetes
Patient ──NOT_ALLERGIC_TO──> Penicillin
```

### Negation rule

Do not create a positive graph fact from `negated_entities`.

```json
{
  "text": "diabetes",
  "label": "Disease_disorder",
  "negated": true,
  "assertion": "absent"
}
```

Correct:

```text
Patient ──DOES_NOT_HAVE──> Diabetes
```

Incorrect:

```text
Patient ──HAS_DISEASE──> Diabetes
```

## 10. Recommended consumer logic

```python
result = consultation_result["medical_nlp"]
structured = result["structured_entities"]
facts = result["clinical_facts"]

for symptom_record in facts["symptom_records"]:
    create_symptom_node(symptom_record["symptom"])

    for severity in symptom_record["severities"]:
        link_severity(symptom_record["symptom"], severity)

    for duration in symptom_record["durations"]:
        link_duration(symptom_record["symptom"], duration)

    for factor in symptom_record["aggravating_factors"]:
        link_aggravating_factor(symptom_record["symptom"], factor)

for medication in structured["medications"]:
    create_medication_fact(medication)

for entity in structured["negated_entities"]:
    create_negative_fact(entity)
```

## 11. Validated example

```text
Doctor: Good morning. What problem are you having today?
Patient: I have severe chest pain for the last three days, and it becomes worse when I climb stairs.
Doctor: Do you have diabetes or any allergy to medicines?
Patient: I do not have diabetes and I am not allergic to penicillin. I take aspirin every morning.
```

Expected facts:

```text
Chest pain          → present
Severe              → present
Last three days     → duration
Climb stairs        → aggravating factor
Diabetes            → absent
Penicillin allergy  → absent
Aspirin             → present medication
```

## 12. Known limitations

1. The Doctor must speak first.
2. The prototype expects exactly two speakers.
3. Overlapping speech may reduce diarisation accuracy.
4. Similar voices may occasionally be merged.
5. Speaker-boundary words may sometimes be assigned to the wrong turn.
6. Partial transcription is less accurate than final transcription.
7. Relation building mainly connects modifiers to the nearest symptom.
8. The current prototype focuses on English consultations.
9. The current clinical focus is cardiology-oriented.

## 13. Error handling

### One speaker detected

```text
Two different speakers are required: Doctor and Patient.
```

### Empty recording

```text
No audio was received.
```

### Audio conversion failure

```text
Audio conversion failed: WAV file was not created.
```

or:

```text
Audio conversion failed: WAV file is empty.
```

### Unsupported format

```text
Unsupported audio format
```

## 14. Local run commands

### Backend

```powershell
cd D:\capstone\ClinExplain\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload
```

### Frontend

```powershell
cd D:\capstone\ClinExplain\frontend
py -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500/live-test.html
```

## 15. Automated tests

```powershell
cd D:\capstone\ClinExplain\backend
python -m pytest tests -v
```

Validated result:

```text
20 passed
```

Coverage includes negation, sentence boundaries, contracted negation, relation building, unsupported-label filtering, false-positive filtering, structured Medical NLP output, and Pydantic schema validation.

## 16. Handover checklist

- [x] Backend starts successfully
- [x] Frontend live-test page opens
- [x] Live transcript updates during recording
- [x] Final transcript is generated
- [x] Two speakers are separated
- [x] Doctor and Patient roles are assigned
- [x] Patient transcript is generated
- [x] Medical entities are extracted
- [x] Negated entities are separated
- [x] Symptom relations are built
- [x] Final JSON validates with Pydantic
- [x] Automated tests pass
- [x] Real two-person test passes

## 17. Stable integration contract

```text
ConsultationResult
├── transcript
├── language
├── patient_transcript
├── conversation[]
└── medical_nlp
    ├── input_text
    ├── raw_entities[]
    ├── entities[]
    ├── structured_entities
    │   ├── symptoms[]
    │   ├── diseases[]
    │   ├── medications[]
    │   ├── allergies[]
    │   ├── durations[]
    │   ├── severities[]
    │   ├── diagnostic_tests[]
    │   ├── procedures[]
    │   ├── family_history[]
    │   ├── aggravating_factors[]
    │   ├── negated_entities[]
    │   └── other_entities[]
    └── clinical_facts
        └── symptom_records[]
```

The downstream module should use `structured_entities` and `clinical_facts` as the main integration inputs.
