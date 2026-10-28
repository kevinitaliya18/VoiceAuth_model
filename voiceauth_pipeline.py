"""VoiceAuth pipeline - runs one call through Model 1 (spoof), Model 2 (speaker), Model 3 (intent) and the risk engine.

Expected folder layout (everything is optional - a missing model just means less evidence):
    VoiceAuth_Local/
      models/spoof_detector/         spoof_model.py, spoof_detector_final.pt
      models/speaker_verification/   speaker_model.py, speaker_config.json, (wavlm_sv/)
      models/intent_classifier/      intent_model.py, head.npz, intent_config.json, (encoder/)
      voiceprints/                   <name>.pt   (made with the Model 2 'enroll' command)
      risk_engine.py, risk_rules.json, voiceauth_pipeline.py, api.py, test_pipeline.py
"""
import importlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from risk_engine import RiskEngine  # noqa: E402

import re  # noqa: E402

MODELS_DIR = os.path.join(HERE, "models")
VOICEPRINT_DIR = os.path.join(HERE, "voiceprints")


_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def safe_voiceprint_name(name):
    """For API use: only plain names (no paths) so a request cannot make the server open arbitrary files."""
    return name if name and _NAME_RE.match(name) and ".." not in name else None


def _import_from(folder, module):
    path = os.path.join(MODELS_DIR, folder)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(module)


class VoiceAuthPipeline:
    def __init__(self, device="cpu", rules_path=None, use_spoof=True, use_speaker=True,
                 use_intent=True, use_whisper=True):
        self.device = device
        self.engine = RiskEngine(rules_path or os.path.join(HERE, "risk_rules.json"))
        self.status = {}
        self.use_whisper = use_whisper
        self._spoof = self._speaker = self._intent = self._whisper = None
        self._speaker_cfg = {}
        if use_spoof:
            self._load("spoof", self._load_spoof)
        if use_speaker:
            self._load("speaker", self._load_speaker)
        if use_intent:
            self._load("intent", self._load_intent)

    # ------------------------------------------------------------------ loading
    def _load(self, name, fn):
        try:
            fn()
            self.status[name] = "loaded"
        except Exception as e:                                   # missing file, missing library, ...
            self.status[name] = f"not available ({type(e).__name__}: {str(e)[:120]})"

    def _load_spoof(self):
        mod = _import_from("spoof_detector", "spoof_model")
        ckpt = os.path.join(MODELS_DIR, "spoof_detector", "spoof_detector_final.pt")
        model, cfg = mod.load_model(ckpt, device=self.device)
        self._spoof = (mod, model, cfg)

    def _load_speaker(self):
        mod = _import_from("speaker_verification", "speaker_model")
        folder = os.path.join(MODELS_DIR, "speaker_verification")
        cfg = mod.load_config(folder)
        local = os.path.join(folder, "wavlm_sv")
        embedder = mod.SpeakerEmbedder(device=self.device, source=local if os.path.isdir(local) else mod.MODEL_ID)
        self._speaker, self._speaker_cfg = (mod, embedder), cfg

    def _load_intent(self):
        mod = _import_from("intent_classifier", "intent_model")
        self._intent = (mod, mod.IntentClassifier(os.path.join(MODELS_DIR, "intent_classifier"), device=self.device))

    def _get_whisper(self):
        if self._whisper is None:
            src = self._intent[1].cfg.get("whisper_id") or self._intent[0].WHISPER_ID   # same Whisper size as in training notebook
            self._whisper = self._intent[0].Transcriber(device=self.device, source=src)
        return self._whisper

    # ------------------------------------------------------------------ voiceprints
    @staticmethod
    def voiceprint_path(name_or_path):
        if os.path.exists(name_or_path):
            return name_or_path
        cand = os.path.join(VOICEPRINT_DIR, name_or_path if name_or_path.endswith(".pt") else name_or_path + ".pt")
        return cand if os.path.exists(cand) else None

    # ------------------------------------------------------------------ main entry
    def analyze(self, audio_path=None, metadata=None, voiceprint=None, transcript=None, language=None):
        """audio_path : call recording (wav/flac/...). Needed for Model 1, Model 2 and (if no transcript) Model 3.
        voiceprint : name or path of an enrolled voiceprint of the CLAIMED person (enables Model 2).
        transcript : text of the call if you already have it (skips Whisper)."""
        timings, notes = {}, []
        spoof_res = speaker_res = intent_res = None

        if audio_path and self._spoof:
            t = time.time(); mod, model, cfg = self._spoof
            spoof_res = mod.predict_file(model, cfg, audio_path, device=self.device)
            timings["spoof_s"] = round(time.time() - t, 2)
        elif audio_path:
            notes.append("Model 1 not available: " + self.status.get("spoof", "disabled"))

        if audio_path and voiceprint and self._speaker:
            vp_path = self.voiceprint_path(voiceprint)
            if vp_path is None:
                notes.append(f"Voiceprint '{voiceprint}' not found in {VOICEPRINT_DIR}")
            else:
                t = time.time(); mod, embedder = self._speaker
                vp, _ = mod.load_voiceprint(vp_path)
                speaker_res = mod.verify_file(embedder, vp, audio_path, self._speaker_cfg["threshold"],
                                              self._speaker_cfg.get("seg_seconds", 6.0))
                timings["speaker_s"] = round(time.time() - t, 2)
        elif audio_path and voiceprint:
            notes.append("Model 2 not available: " + self.status.get("speaker", "disabled"))

        if self._intent:
            if not transcript and audio_path and self.use_whisper:
                t = time.time()
                try:
                    transcript = self._get_whisper().transcribe(audio_path, language=language)
                    timings["whisper_s"] = round(time.time() - t, 2)
                except Exception as e:
                    notes.append(f"Speech-to-text failed: {type(e).__name__}: {str(e)[:100]}")
            if transcript:
                t = time.time()
                intent_res = self._intent[1].analyze_text(transcript)
                timings["intent_s"] = round(time.time() - t, 2)
        elif transcript or audio_path:
            notes.append("Model 3 not available: " + self.status.get("intent", "disabled"))

        assessment = self.engine.assess(spoof=spoof_res, speaker=speaker_res, intent=intent_res, metadata=metadata)
        return {
            "assessment": assessment,
            "models": {"spoof": spoof_res, "speaker": speaker_res,
                       "intent": intent_res, "transcript": transcript},
            "model_status": self.status, "notes": notes, "timings": timings,
        }
