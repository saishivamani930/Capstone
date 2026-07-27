from fastapi import FastAPI

from app.api.speech import router as speech_router
from app.api.medical_nlp import router as medical_nlp_router


app = FastAPI(
    title="ClinExplain API",
    version="0.1.0"
)

app.include_router(speech_router)
app.include_router(medical_nlp_router)


@app.get("/")
def root():
    return {"message": "ClinExplain backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}