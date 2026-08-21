from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.speech import router as speech_router
from app.api.medical_nlp import router as medical_nlp_router
from app.api.reasoning import router as reasoning_router


app = FastAPI(
    title="ClinExplain API",
    version="0.1.0"
)

# Enable CORS middleware to handle browser preflight OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from Live Server or any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, OPTIONS, GET, etc.
    allow_headers=["*"],  # Allows Content-Type and other headers
)

# Include API routers
app.include_router(speech_router)
app.include_router(medical_nlp_router)
app.include_router(reasoning_router)


@app.get("/")
def root():
    return {"message": "ClinExplain backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}