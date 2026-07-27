# ClinExplain

ClinExplain is an AI-based clinical consultation processing system.

## Current module

The AI Speech and Medical NLP module supports:

- Live microphone audio streaming
- Partial live transcription
- Final Whisper transcription
- Doctor–Patient speaker diarisation
- Patient-only transcript generation
- Medical entity extraction
- Negation handling
- Clinical relation building
- Structured JSON output

## Run the backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload