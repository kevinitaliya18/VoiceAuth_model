"""VoiceAuth API (FastAPI).  Start:  uvicorn api:app --host 127.0.0.1 --port 8000
Docs page (try it in the browser):  http://127.0.0.1:8000/docs

POST /analyze  multipart form: audio (file, optional), metadata (JSON text), voiceprint (name), transcript (text), language
POST /engine   JSON body with ready model outputs (spoof.p_spoof, speaker.similarity/threshold,
               intent.scam_probability/flags, metadata) - no audio or models needed
GET  /health   which models are loaded
POST /harness/analyze   endpoint for the tool-deepfake-testing harness (see harness_bridge.py, bridge_config.json)
Environment: VISHING_API_KEY (optional API key), VOICEAUTH_DEVICE (cpu/cuda), VOICEAUTH_USE_WHISPER (0 = skip speech-to-text)
"""
import json
import os
import tempfile
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import harness_bridge
from voiceauth_pipeline import VoiceAuthPipeline, safe_voiceprint_name

app = FastAPI(title="VoiceAuth API", version="0.1")
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = VoiceAuthPipeline(device=os.environ.get("VOICEAUTH_DEVICE", "cpu"),
                                      use_whisper=os.environ.get("VOICEAUTH_USE_WHISPER", "1") != "0")
    return _pipeline


harness_bridge.PIPELINE_PROVIDER = get_pipeline          # lets /harness/analyze reuse the same loaded models
app.include_router(harness_bridge.router)


class SpoofOutput(BaseModel):
    p_spoof: float = Field(ge=0.0, le=1.0)


class SpeakerOutput(BaseModel):
    similarity: float
    threshold: float


class IntentOutput(BaseModel):
    scam_probability: float = Field(ge=0.0, le=1.0)
    flags: dict[str, float] = Field(default_factory=dict)


class EngineRequest(BaseModel):
    spoof: Optional[SpoofOutput] = None
    speaker: Optional[SpeakerOutput] = None
    intent: Optional[IntentOutput] = None
    metadata: Optional[dict] = None


@app.get("/health")
def health(p: VoiceAuthPipeline = Depends(get_pipeline)):
    return {"status": "ok", "models": p.status}


@app.post("/engine")
def engine_only(req: EngineRequest, p: VoiceAuthPipeline = Depends(get_pipeline)):
    return p.engine.assess(
        spoof=req.spoof.model_dump() if req.spoof is not None else None,
        speaker=req.speaker.model_dump() if req.speaker is not None else None,
        intent=req.intent.model_dump() if req.intent is not None else None,
        metadata=req.metadata,
    )


@app.post("/analyze")
async def analyze(audio: Optional[UploadFile] = File(None), metadata: str = Form("{}"),
                  voiceprint: Optional[str] = Form(None), transcript: Optional[str] = Form(None),
                  language: Optional[str] = Form(None), p: VoiceAuthPipeline = Depends(get_pipeline)):
    try:
        meta = json.loads(metadata or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON text")
    if voiceprint and safe_voiceprint_name(voiceprint) is None:
        raise HTTPException(status_code=400, detail="voiceprint must be a plain name (letters, digits, _ - .)")
    tmp_path = None
    try:
        if audio is not None:
            suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(await audio.read())
                tmp_path = f.name
        if tmp_path is None and not transcript:
            raise HTTPException(status_code=400, detail="send an audio file and/or a transcript")
        return p.analyze(audio_path=tmp_path, metadata=meta, voiceprint=voiceprint,
                         transcript=transcript, language=language)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
