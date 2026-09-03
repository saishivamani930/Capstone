from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.speech import router as speech_router
from app.api.medical_nlp import router as medical_nlp_router
from app.api.reasoning import router as reasoning_router
from app.api.rag import router as rag_router


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

from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

# Include API routers
app.include_router(speech_router)
app.include_router(medical_nlp_router)
app.include_router(reasoning_router)
app.include_router(rag_router)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {"status": "ok"}


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "ClinExplain backend is running"}
else:
    @app.get("/")
    def root():
        return {"message": "ClinExplain backend is running"}