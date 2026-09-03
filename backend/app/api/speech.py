import asyncio
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder

from app.speech.audio_processor import convert_webm_to_wav
from app.speech.diarizer import diarize_audio
from app.speech.pipeline import process_conversation
from app.speech.transcriber import (
    transcribe_audio,
    transcribe_partial_audio,
)
from app.medical_nlp.pipeline import run_medical_nlp
from app.reasoning.missing_info import MissingInfoEngine
from app.reasoning.risk_analyzer import RiskAnalyzer
from app.reasoning.wikidata_client import WikidataClient

missing_info_engine = MissingInfoEngine()
risk_analyzer_engine = RiskAnalyzer()
wiki_client_engine = WikidataClient()

router = APIRouter(
    prefix="/speech",
    tags=["Speech"]
)

ALLOWED_AUDIO_TYPES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg"
}

PARTIAL_TRANSCRIPT_EVERY_CHUNKS = 6


def transcribe_webm_snapshot(audio_bytes: bytes) -> dict:
    """
    Convert the currently received WebM audio into WAV and
    run Whisper without diarisation or Medical NLP.
    """

    webm_path = None
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".webm",
        ) as webm_file:
            webm_file.write(audio_bytes)
            webm_path = webm_file.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as wav_file:
            wav_path = wav_file.name

        convert_webm_to_wav(
            webm_path,
            wav_path,
        )

        if not os.path.exists(wav_path):
            raise RuntimeError(
                "Partial audio conversion failed."
            )

        if os.path.getsize(wav_path) == 0:
            raise RuntimeError(
                "Partial WAV file is empty."
            )

        return transcribe_partial_audio(wav_path)

    finally:
        for temporary_path in (webm_path, wav_path):
            if (
                temporary_path
                and os.path.exists(temporary_path)
            ):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass


@router.post("/reset")
def reset_session():
    """
    Explicitly clear active session data and temporary audio cache.
    """
    return {
        "status": "success",
        "message": "Session reset: Temporary audio buffer and background tasks cleared."
    }


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    content = await file.read()

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "message": "Audio file received successfully"
    }


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format"
        )

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty"
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        result = transcribe_audio(temp_path)

        return {
            "filename": file.filename,
            "transcript": result["text"],
            "language": result["language"],
            "language_probability": result["language_probability"]
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/diarize")
async def diarize(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format"
        )

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty"
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        segments = diarize_audio(temp_path)

        return {
            "filename": file.filename,
            "speaker_count": len(set(
                segment["speaker"]
                for segment in segments
            )),
            "segments": segments
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/process")
async def process_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format"
        )

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty"
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        result = process_conversation(temp_path)

        return {
            "filename": file.filename,
            **result
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@router.websocket("/live")
async def live_audio(websocket: WebSocket):
    await websocket.accept()

    session_id = uuid4().hex
    chunk_count = 0
    total_bytes = 0
    audio_buffer = bytearray()

    webm_path = None
    wav_path = None

    last_partial_text = ""

    latest_partial_snapshot: bytes | None = None
    latest_partial_chunk = 0

    partial_event = asyncio.Event()
    partial_worker_should_stop = False
    partial_worker_task: asyncio.Task | None = None

    send_lock = asyncio.Lock()

    async def send_event(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(
                jsonable_encoder(payload)
            )

    async def partial_transcription_worker() -> None:
        nonlocal last_partial_text

        while True:
            await partial_event.wait()
            partial_event.clear()

            if partial_worker_should_stop:
                return

            snapshot = latest_partial_snapshot
            through_chunk = latest_partial_chunk

            if not snapshot:
                continue

            try:
                partial_result = await run_in_threadpool(
                    transcribe_webm_snapshot,
                    snapshot,
                )

                # Do not send an outdated partial result after
                # the user has clicked Stop Recording.
                if partial_worker_should_stop:
                    return

                partial_text = (
                    partial_result.get("text", "").strip()
                )

                if not partial_text:
                    continue

                if partial_text == last_partial_text:
                    continue

                last_partial_text = partial_text

                # Run live Medical NLP, missing information detection, and risk analysis
                nlp_res = run_medical_nlp(partial_text)
                structured_entities = nlp_res.get("structured_entities", {})

                # Extract present symptoms (non-negated)
                symptoms = structured_entities.get("symptoms", [])
                present_symptom_names = [
                    s["text"] for s in symptoms if not s.get("negated", False)
                ]

                # Resolve Wikidata Q-IDs for present symptoms
                symptom_qids = []
                for s_name in present_symptom_names:
                    try:
                        qid = await wiki_client_engine.get_qid_for_entity(s_name)
                        if qid:
                            symptom_qids.append(qid)
                    except Exception:
                        pass

                # Find candidate diseases via SPARQL
                candidate_diseases = []
                if symptom_qids:
                    try:
                        candidate_diseases = await wiki_client_engine.find_candidate_diseases(symptom_qids)
                    except Exception:
                        candidate_diseases = []

                all_entities = nlp_res.get("entities", [])

                # Analyze missing information & risk
                missing_info = missing_info_engine.analyze_sync(
                    present_symptom_names, candidate_diseases, all_entities
                )
                risk_res = risk_analyzer_engine.analyze_risk(
                    present_symptom_names, candidate_diseases
                )

                # Construct live knowledge graph node & relationship previews
                all_entities = nlp_res.get("entities", [])
                graph_nodes = [
                    {
                        "id": f"e_{idx}",
                        "label": e["text"],
                        "category": e["label"],
                        "status": "absent" if e.get("negated", False) else "present"
                    }
                    for idx, e in enumerate(all_entities)
                ]
                graph_relationships = []
                for idx, record in enumerate(nlp_res.get("clinical_facts", {}).get("symptom_records", [])):
                    s_id = f"symp_{idx}"
                    for m in record.get("severities", []):
                        graph_relationships.append({
                            "source": s_id,
                            "target": m["text"],
                            "type": "HAS_SEVERITY"
                        })
                    for m in record.get("durations", []):
                        graph_relationships.append({
                            "source": s_id,
                            "target": m["text"],
                            "type": "HAS_DURATION"
                        })

                await send_event({
                    "event": "partial_transcript",
                    "session_id": session_id,
                    "through_chunk": through_chunk,
                    "transcript": partial_text,
                    "language": partial_result.get(
                        "language"
                    ),
                    "is_final": False,
                    "medical_nlp": nlp_res,
                    "present_symptoms": present_symptom_names,
                    "missing_information": missing_info,
                    "risk_analysis": risk_res,
                    "candidate_diseases": candidate_diseases,
                    "graph_nodes": graph_nodes,
                    "graph_relationships": graph_relationships,
                })

            except Exception as error:
                if partial_worker_should_stop:
                    return

                try:
                    await send_event({
                        "event": "partial_warning",
                        "message": str(error),
                        "through_chunk": through_chunk,
                    })
                except Exception:
                    return

    await send_event({
        "event": "connected",
        "session_id": session_id,
    })

    partial_worker_task = asyncio.create_task(
        partial_transcription_worker()
    )

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            audio_chunk = message.get("bytes")
            control_message = message.get("text")

            if audio_chunk:
                audio_buffer.extend(audio_chunk)

                chunk_count += 1
                total_bytes += len(audio_chunk)

                await send_event({
                    "event": "chunk_received",
                    "chunk_number": chunk_count,
                    "chunk_size_bytes": len(audio_chunk),
                    "total_bytes": total_bytes,
                })

                if (
                    chunk_count
                    % PARTIAL_TRANSCRIPT_EVERY_CHUNKS
                    == 0
                ):
                    # Always store the newest available audio.
                    # If Whisper is busy, older pending snapshots
                    # are replaced with this latest snapshot.
                    latest_partial_snapshot = bytes(
                        audio_buffer
                    )
                    latest_partial_chunk = chunk_count

                    partial_event.set()

                continue

            if control_message != "stop":
                continue

            if not audio_buffer:
                await send_event({
                    "event": "error",
                    "message": "No audio was received.",
                })
                break

            # Stop temporary updates before final processing.
            partial_worker_should_stop = True
            partial_event.set()

            await send_event({
                "event": "processing",
                "message": "Processing consultation audio.",
            })

            if (
                partial_worker_task is not None
                and not partial_worker_task.done()
            ):
                partial_worker_task.cancel()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".webm",
            ) as webm_file:
                webm_file.write(audio_buffer)
                webm_path = webm_file.name

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav",
            ) as wav_file:
                wav_path = wav_file.name

            await run_in_threadpool(
                convert_webm_to_wav,
                webm_path,
                wav_path,
            )

            if not os.path.exists(wav_path):
                raise RuntimeError(
                    "Audio conversion failed: "
                    "WAV file was not created."
                )

            if os.path.getsize(wav_path) == 0:
                raise RuntimeError(
                    "Audio conversion failed: "
                    "WAV file is empty."
                )

            result = await run_in_threadpool(
                process_conversation,
                wav_path,
            )

            if result is None:
                raise RuntimeError(
                    "The speech-processing pipeline "
                    "returned no result."
                )

            await send_event({
                "event": "completed",
                "session_id": session_id,
                "chunk_count": chunk_count,
                "total_bytes": total_bytes,
                "is_final": True,
                "result": result,
            })

            await asyncio.sleep(0.2)
            break

    except WebSocketDisconnect:
        pass

    except Exception as error:
        try:
            await send_event({
                "event": "error",
                "message": str(error),
                "error_type": type(error).__name__,
            })

            await asyncio.sleep(0.1)

        except Exception:
            pass

    finally:
        partial_worker_should_stop = True
        partial_event.set()

        if (
            partial_worker_task is not None
            and not partial_worker_task.done()
        ):
            partial_worker_task.cancel()

            try:
                await partial_worker_task
            except asyncio.CancelledError:
                pass

        for temporary_path in (webm_path, wav_path):
            if (
                temporary_path
                and os.path.exists(temporary_path)
            ):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

        try:
            await websocket.close()
        except Exception:
            pass