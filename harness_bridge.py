"""Bridge between the 'tool-deepfake-testing' harness and the VoiceAuth pipeline.

Adds   POST /harness/analyze   (+ GET /harness/contract) to the VoiceAuth API.

It accepts the request styles an HTTP test harness normally uses:
  * JSON with base64 audio   {"audio_base64": "...", "context": {...}, "speaker_id": "alice"}
  * multipart/form-data      file field + context (JSON text) + speaker_id
  * raw audio body           Content-Type: audio/wav (or application/octet-stream)
and answers with the flat fields the harness README lists:
  p_spoof, similarity, manipulation_score, flags, action   (+ extras such as risk_level, reasons)

EVERYTHING that depends on the harness contract lives in bridge_config.json (field names, action names,
context key mapping). Run  `check-connection --adapter http`  in the harness, read which fields it reports as
missing/unexpected, then edit bridge_config.json - no code change needed.
"""
import base64
import binascii
import copy
import hmac
import json
import os
import re
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "bridge_config.json")

DEFAULT_CFG = {
    "api_key_env": "VISHING_API_KEY",
    "include_extras": True,
    "include_debug": False,
    "flags_format": "list",
    "request_fields": {
        "audio_json": ["audio_base64", "audio_b64", "audio_data", "audio", "data", "file"],
        "audio_multipart": ["audio", "file", "audio_file", "upload", "data"],
        "context": ["context", "metadata", "meta"],
        "voiceprint": ["speaker_id", "claimed_speaker", "claimed_identity", "enrolled_id",
                       "enrolled_speaker", "voiceprint", "identity", "user_id"],
        "transcript": ["transcript", "text"],
        "language": ["language", "lang"],
        "format": ["format", "audio_format", "ext", "extension"],
    },
    "default_voiceprint": None,
    "output_names": {
        "p_spoof": "p_spoof", "similarity": "similarity", "manipulation_score": "manipulation_score",
        "flags": "flags", "action": "action",
        "is_spoof": "is_spoof", "is_match": "is_match", "risk_level": "risk_level", "risk_score": "risk_score",
        "reasons": "reasons", "recommendations": "recommendations", "transcript": "transcript",
        "escalate_to_supervisor": "escalate_to_supervisor",
    },
    "core_fields": ["p_spoof", "similarity", "manipulation_score", "flags", "action"],
    "action_map": {"allow": "allow", "warn_and_verify": "warn", "force_verification": "force_mfa_callback"},
    "flag_name_map": {},
    "missing_values": {"p_spoof": None, "similarity": None, "manipulation_score": None},
    "context_map": {
        "transaction_amount": "transaction_amount", "amount": "transaction_amount", "amount_inr": "transaction_amount",
        "txn_amount": "transaction_amount", "value": "transaction_amount",
        "new_beneficiary": "new_beneficiary", "new_payee": "new_beneficiary", "is_new_beneficiary": "new_beneficiary",
        "beneficiary_new": "new_beneficiary",
        "call_origin": "call_origin", "origin": "call_origin", "caller_origin": "call_origin",
        "request_type": "request_type", "request": "request_type", "requested_action": "request_type",
        "action_requested": "request_type",
        "flagged_number": "flagged_number", "number_flagged": "flagged_number", "known_fraud_number": "flagged_number",
        "reported_number": "flagged_number",
        "after_hours": "after_hours", "off_hours": "after_hours",
        "caller_trusted": "caller_trusted", "trusted_caller": "caller_trusted", "known_contact": "caller_trusted",
        "claimed_role": "claimed_role", "role": "claimed_role",
    },
    "value_map": {
        "request_type": {"transfer": "fund_transfer", "payment": "fund_transfer", "wire": "fund_transfer",
                         "fund_transfer": "fund_transfer", "otp": "otp_disclosure", "password": "otp_disclosure",
                         "pin": "otp_disclosure", "otp_disclosure": "otp_disclosure",
                         "privileged_access": "privileged_access", "admin_access": "privileged_access",
                         "access": "privileged_access", "info": "info_request", "info_request": "info_request",
                         "none": "none"},
        "call_origin": {"known": "known_contact", "known_contact": "known_contact", "trusted": "known_contact",
                        "contact": "known_contact", "unknown": "unknown", "withheld": "withheld", "private": "withheld",
                        "hidden": "withheld", "international": "international", "foreign": "international",
                        "voip": "voip", "internet": "voip"},
    },
}

PIPELINE_PROVIDER = None            # api.py sets this to its get_pipeline function
router = APIRouter()


def bridge_pipeline():
    if PIPELINE_PROVIDER is None:
        raise HTTPException(status_code=500, detail="pipeline not configured")
    return PIPELINE_PROVIDER()


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_cfg(path=None):
    cfg = copy.deepcopy(DEFAULT_CFG)
    p = path or CONFIG_PATH
    if os.path.exists(p):
        _merge(cfg, json.load(open(p, encoding="utf-8")))
    return cfg


def check_api_key(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    key = os.environ.get(load_cfg()["api_key_env"])
    if not key:
        return                                                    # no key configured -> open (local use only!)
    provided = x_api_key
    if not provided and authorization:
        provided = authorization.split(None, 1)[1] if authorization.lower().startswith("bearer ") else authorization
    if not provided or not hmac.compare_digest(provided.encode(), key.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


# ---------------------------------------------------------------- request parsing
def _sniff_suffix(data):
    if data[:4] == b"RIFF": return ".wav"
    if data[:4] == b"fLaC": return ".flac"
    if data[:4] == b"OggS": return ".ogg"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"): return ".mp3"
    return ".wav"


def _first(fields, names):
    for n in names:
        v = fields.get(n)
        if v not in (None, "", {}):
            return v
    return None


def _decode_b64(s):
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="audio field is not valid base64")


def _parse_text_mapping(v):
    """context may arrive as dict, JSON text or YAML text."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        try:
            out = json.loads(v)
            return out if isinstance(out, dict) else {}
        except json.JSONDecodeError:
            try:
                import yaml
                out = yaml.safe_load(v)
                return out if isinstance(out, dict) else {}
            except Exception:
                return {}
    return {}


async def _parse_request(request: Request, cfg):
    rf, ctype = cfg["request_fields"], (request.headers.get("content-type") or "").lower()
    audio, fields = None, {}
    if "multipart/form-data" in ctype or "application/x-www-form-urlencoded" in ctype:
        form = await request.form()
        for name in rf["audio_multipart"]:
            v = form.get(name)
            if v is not None and hasattr(v, "read"):
                audio = await v.read(); break
        fields = {k: v for k, v in form.items() if isinstance(v, str)}
    elif "application/json" in ctype:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="body is not valid JSON")
        fields = body if isinstance(body, dict) else {}
        for name in rf["audio_json"]:
            v = fields.get(name)
            if isinstance(v, str) and len(v) > 20:
                audio = _decode_b64(v); break
    elif ctype.startswith("audio/") or "octet-stream" in ctype:
        audio = await request.body()
    return audio, fields


# ---------------------------------------------------------------- context -> VoiceAuth metadata
def _to_bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def _to_amount(v):
    s = re.sub(r"[^0-9.]", "", str(v))
    return float(s) if s.count(".") <= 1 and s not in ("", ".") else None


def _flatten(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten(v)
        else:
            yield str(k).lower(), v


def map_context(ctx, cfg):
    meta, unmapped = {}, []
    cmap = {k.lower(): v for k, v in cfg["context_map"].items()}
    for key, val in _flatten(ctx or {}):
        target = cmap.get(key)
        if target is None:
            unmapped.append(key); continue
        if target in ("new_beneficiary", "flagged_number", "after_hours", "caller_trusted"):
            meta[target] = _to_bool(val)
        elif target == "transaction_amount":
            amt = _to_amount(val)
            if amt is not None: meta[target] = amt
        else:
            s = str(val).strip().lower().replace(" ", "_")
            meta[target] = cfg["value_map"].get(target, {}).get(s, s)
    return meta, unmapped


# ---------------------------------------------------------------- VoiceAuth result -> harness contract
def to_contract(result, cfg, debug=None):
    a, m = result["assessment"], result["models"]
    sp, sk, it = m["spoof"], m["speaker"], m["intent"]
    fmap = cfg["flag_name_map"]
    if it is None:
        flags = {} if cfg["flags_format"] == "dict" else []
    elif cfg["flags_format"] == "dict":
        flags = {fmap.get(k, k): round(v, 4) for k, v in it["flags"].items()}
    else:
        flags = [fmap.get(k, k) for k in it["triggered_flags"]]
    flat = {
        "p_spoof": None if sp is None else sp["p_spoof"], "is_spoof": None if sp is None else sp["is_spoof"],
        "similarity": None if sk is None else sk["similarity"], "is_match": None if sk is None else sk["is_match"],
        "manipulation_score": None if it is None else it["scam_probability"], "flags": flags,
        "action": cfg["action_map"].get(a["action"], a["action"]),
        "risk_level": a["risk_level"], "risk_score": a["risk_score"], "reasons": a["reasons"],
        "recommendations": a["recommendations"], "transcript": m["transcript"],
        "escalate_to_supervisor": a["escalate_to_supervisor"],
    }
    out = {}
    for ours, contract_name in cfg["output_names"].items():
        if contract_name is None:
            continue
        if not cfg["include_extras"] and ours not in cfg["core_fields"]:
            continue
        val = flat.get(ours)
        if val is None and ours in cfg["missing_values"]:
            val = cfg["missing_values"][ours]
        out[contract_name] = val
    if debug is not None and cfg["include_debug"]:
        out["_bridge"] = debug
    return out


# ---------------------------------------------------------------- endpoints
@router.post("/harness/analyze")
async def harness_analyze(request: Request, pipeline=Depends(bridge_pipeline), _auth=Depends(check_api_key)):
    from voiceauth_pipeline import safe_voiceprint_name
    cfg = load_cfg()
    rf = cfg["request_fields"]
    audio, fields = await _parse_request(request, cfg)
    ctx = _parse_text_mapping(_first(fields, rf["context"]))
    meta, unmapped = map_context(ctx, cfg)
    transcript = _first(fields, rf["transcript"]) or _first(ctx, rf["transcript"])
    language = _first(fields, rf["language"])
    raw_vp = _first(fields, rf["voiceprint"]) or _first(ctx, rf["voiceprint"]) or cfg["default_voiceprint"]
    voiceprint = safe_voiceprint_name(str(raw_vp)) if raw_vp else None
    if audio is None and not transcript:
        raise HTTPException(status_code=400, detail="send an audio file/base64 and/or a transcript")

    tmp = None
    try:
        if audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=_sniff_suffix(audio)) as f:
                f.write(audio); tmp = f.name
        try:
            result = pipeline.analyze(audio_path=tmp, metadata=meta, voiceprint=voiceprint,
                                      transcript=transcript, language=language)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"could not analyse the audio: {type(e).__name__}: {str(e)[:150]}")
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    debug = {"metadata_used": meta, "unmapped_context_keys": unmapped, "voiceprint_used": voiceprint,
             "notes": result["notes"], "model_status": result["model_status"], "timings": result["timings"]}
    return to_contract(result, cfg, debug)


@router.get("/harness/contract")
def harness_contract(_auth=Depends(check_api_key)):
    """Shows the active mapping so you can compare it with the harness check-connection report."""
    cfg = load_cfg()
    return {"output_fields": [v for v in cfg["output_names"].values() if v],
            "core_fields": cfg["core_fields"], "include_extras": cfg["include_extras"],
            "action_map": cfg["action_map"], "flags_format": cfg["flags_format"],
            "accepted_audio_fields": cfg["request_fields"], "context_keys_understood": sorted(cfg["context_map"])}
