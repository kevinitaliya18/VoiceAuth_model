"""Risk engine - turns the outputs of the 3 models + call metadata into LOW / MEDIUM / HIGH and an action.

Rule-based and fully transparent (no training). All numbers live in risk_rules.json, so they can be tuned
without touching code. The default weights are expert-set starting points, NOT learned values: tune them on
your own test calls (see test_pipeline.py batch).

Inputs (all optional - a missing model simply means less evidence):
  spoof    : Model 1 result  {"p_spoof": 0..1, ...}
  speaker  : Model 2 result  {"similarity": x, "threshold": t, ...}
  intent   : Model 3 result  {"scam_probability": 0..1, "flags": {name: prob}, ...}
  metadata : {"request_type": "fund_transfer|privileged_access|otp_disclosure|info_request|none",
              "transaction_amount": 400000, "new_beneficiary": true,
              "call_origin": "known_contact|unknown|withheld|international|voip",
              "flagged_number": false, "after_hours": false, "caller_trusted": false,
              "claimed_role": "ceo|bank_employee|customer|government|family|..."}
"""
import copy
import json
import math
import os

DEFAULT_RULES = {
    "evidence_weights": {"voice_fake": 0.90, "identity_mismatch": 0.70, "manipulation": 0.70,
                         "sensitive_ask": 0.50, "pressure": 0.30},
    "evidence_floor": 0.20,
    "speaker_sharpness": 25,
    "sensitive_ask_flags": ["asks_otp_password", "asks_money_transfer",
                            "asks_personal_documents", "asks_remote_access_or_link"],
    "pressure_flags": ["urgency_pressure", "threat_fear", "secrecy_request"],
    "context": {
        "new_beneficiary": 0.25,
        "amount_tiers": [[1000000, 0.30], [100000, 0.20], [25000, 0.10]],
        "origin_weights": {"unknown": 0.10, "withheld": 0.15, "international": 0.15, "voip": 0.10, "known_contact": 0.0},
        "flagged_number": 0.30,
        "request_weights": {"fund_transfer": 0.20, "privileged_access": 0.25, "otp_disclosure": 0.30,
                            "info_request": 0.10, "none": 0.0},
        "after_hours": 0.05,
    },
    "context_direct_weight": 0.25,
    "request_stakes": {"fund_transfer": 0.8, "privileged_access": 0.9, "otp_disclosure": 0.9,
                       "info_request": 0.4, "none": 0.1, "unknown": 0.3},
    "amount_stakes": [[1000000, 1.0], [100000, 0.7], [25000, 0.4]],
    "thresholds": {"medium": 0.40, "high": 0.70, "stakes_shift": 0.20, "min_medium": 0.18, "min_high": 0.45},
    "trusted_discount": 0.70,
    "corroboration_cap": {"enabled": True, "strong_min": 0.60, "stakes_below": 0.50},
    "hard_rules": {
        "fake_voice_sensitive_request": {"enabled": True, "voice_fake_min": 0.85},
        "credential_phishing": {"enabled": True, "otp_flag_min": 0.80, "manipulation_min": 0.60},
        "identity_mismatch_sensitive": {"enabled": True, "mismatch_min": 0.80},
        "unverifiable_high_value": {"enabled": True, "stakes_min": 0.70},
    },
    "sensitive_request_types": ["fund_transfer", "privileged_access", "otp_disclosure"],
}

FLAG_TEXT = {
    "authority_claim": "claims to be an official or authority", "urgency_pressure": "creates urgency",
    "threat_fear": "uses threats", "asks_otp_password": "asks for OTP/PIN/password",
    "asks_money_transfer": "asks for a money transfer", "asks_personal_documents": "asks for documents or personal data",
    "asks_remote_access_or_link": "asks to install an app or open a link", "secrecy_request": "asks for secrecy",
    "emotional_emergency": "uses a family emergency story",
}
LEVELS = ["LOW", "MEDIUM", "HIGH"]
ACTIONS = {"LOW": "allow", "MEDIUM": "warn_and_verify", "HIGH": "force_verification"}


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


class RiskEngine:
    def __init__(self, rules=None):
        self.r = copy.deepcopy(DEFAULT_RULES)
        if isinstance(rules, str) and os.path.exists(rules):
            rules = json.load(open(rules, encoding="utf-8"))
        if isinstance(rules, dict):
            _merge(self.r, rules)

    # ------------------------------------------------------------ evidence from the models
    def _flag_max(self, intent, names):
        if not intent:
            return None
        flags = intent.get("flags", {})
        vals = [float(flags.get(n, 0.0)) for n in names]
        return max(vals) if vals else 0.0

    def _evidence(self, spoof, speaker, intent):
        ev = {"voice_fake": None, "identity_mismatch": None, "manipulation": None,
              "sensitive_ask": None, "pressure": None}
        if spoof is not None:
            ev["voice_fake"] = float(spoof["p_spoof"])
        if speaker is not None:
            ev["identity_mismatch"] = _sigmoid((float(speaker["threshold"]) - float(speaker["similarity"])) * self.r["speaker_sharpness"])
        if intent is not None:
            ev["manipulation"] = float(intent["scam_probability"])
            ev["sensitive_ask"] = self._flag_max(intent, self.r["sensitive_ask_flags"])
            ev["pressure"] = self._flag_max(intent, self.r["pressure_flags"])
        return ev

    # ------------------------------------------------------------ context
    def _context(self, meta):
        c, terms = self.r["context"], {}
        if meta.get("new_beneficiary"):
            terms["new_beneficiary"] = c["new_beneficiary"]
        amt = meta.get("transaction_amount")
        if amt:
            for limit, w in sorted(c["amount_tiers"], key=lambda t: -t[0]):
                if float(amt) >= limit:
                    terms["high_value_amount"] = w; break
        origin = meta.get("call_origin")
        if origin in c["origin_weights"] and c["origin_weights"][origin] > 0:
            terms["call_origin_" + origin] = c["origin_weights"][origin]
        if meta.get("flagged_number"):
            terms["flagged_number"] = c["flagged_number"]
        rt = meta.get("request_type", "none")
        if c["request_weights"].get(rt, 0) > 0:
            terms["request_" + rt] = c["request_weights"][rt]
        if meta.get("after_hours"):
            terms["after_hours"] = c["after_hours"]
        risk = 1.0
        for w in terms.values():
            risk *= (1.0 - w)
        return 1.0 - risk, terms

    def _stakes(self, meta):
        rs = self.r["request_stakes"]
        s = rs.get(meta.get("request_type", "none"), rs["unknown"])
        amt = meta.get("transaction_amount")
        if amt:
            for limit, v in sorted(self.r["amount_stakes"], key=lambda t: -t[0]):
                if float(amt) >= limit:
                    s = max(s, v); break
        return s

    # ------------------------------------------------------------ main entry
    def assess(self, spoof=None, speaker=None, intent=None, metadata=None):
        meta = dict(metadata or {})
        meta.setdefault("request_type", "none")
        r = self.r
        ev = self._evidence(spoof, speaker, intent)

        floor = r["evidence_floor"]
        contrib, keep = {}, 1.0
        for name, s in ev.items():
            if s is None:
                continue
            eff = max(0.0, (s - floor) / (1.0 - floor))
            c = r["evidence_weights"][name] * eff
            contrib[name] = round(c, 3)
            keep *= (1.0 - c)
        evidence_risk = 1.0 - keep

        context_risk, ctx_terms = self._context(meta)
        score = 1.0 - (1.0 - evidence_risk) * (1.0 - r["context_direct_weight"] * context_risk)

        trusted = False
        if meta.get("caller_trusted") and (ev["voice_fake"] is not None and ev["voice_fake"] < 0.3) \
                and (ev["identity_mismatch"] is None or ev["identity_mismatch"] < 0.3):
            score *= r["trusted_discount"]; trusted = True

        stakes = self._stakes(meta)
        t = r["thresholds"]
        med_thr = max(t["min_medium"], t["medium"] - t["stakes_shift"] * stakes)
        high_thr = max(t["min_high"], t["high"] - t["stakes_shift"] * stakes)
        level_idx = 2 if score >= high_thr else 1 if score >= med_thr else 0

        # ---- hard rules (can only raise the level)
        fired, hr = [], r["hard_rules"]
        sensitive_req = meta["request_type"] in r["sensitive_request_types"]
        if hr["fake_voice_sensitive_request"]["enabled"] and sensitive_req and ev["voice_fake"] is not None \
                and ev["voice_fake"] >= hr["fake_voice_sensitive_request"]["voice_fake_min"]:
            fired.append(("fake_voice_sensitive_request", "AI-generated voice combined with a sensitive request", 2))
        otp = self._flag_max(intent, ["asks_otp_password"])
        if hr["credential_phishing"]["enabled"] and otp is not None and ev["manipulation"] is not None \
                and otp >= hr["credential_phishing"]["otp_flag_min"] and ev["manipulation"] >= hr["credential_phishing"]["manipulation_min"]:
            fired.append(("credential_phishing", "Caller asks for OTP/PIN/password inside a scam-like conversation", 2))
        if hr["identity_mismatch_sensitive"]["enabled"] and sensitive_req and ev["identity_mismatch"] is not None \
                and ev["identity_mismatch"] >= hr["identity_mismatch_sensitive"]["mismatch_min"]:
            fired.append(("identity_mismatch_sensitive", "Voice does not match the claimed person and a sensitive action is requested", 2))
        if hr["unverifiable_high_value"]["enabled"] and ev["voice_fake"] is None and stakes >= hr["unverifiable_high_value"]["stakes_min"]:
            fired.append(("unverifiable_high_value", "High-stakes request but the voice could not be checked", 1))
        for _, _, lvl in fired:
            level_idx = max(level_idx, lvl)

        # ---- corroboration: HIGH needs a hard rule, a high-stakes request, or two strong model signals
        capped = False
        cc = r["corroboration_cap"]
        if cc["enabled"] and level_idx == 2 and not any(f[2] == 2 for f in fired) and stakes < cc["stakes_below"]:
            strong = [k for k in ("voice_fake", "identity_mismatch", "manipulation")
                      if ev[k] is not None and ev[k] >= cc["strong_min"]]
            if len(strong) < 2:
                level_idx, capped = 1, True
        level = LEVELS[level_idx]

        reasons = self._reasons(ev, intent, ctx_terms, meta, fired, trusted)
        if capped:
            reasons.append("Only one model signal is strong, so the level is limited to MEDIUM until a second signal confirms it")
        available = [k for k, v in (("spoof", spoof), ("speaker", speaker), ("intent", intent)) if v is not None]
        missing = [k for k in ("spoof", "speaker", "intent") if k not in available]
        return {
            "risk_level": level, "risk_score": round(score, 3), "action": ACTIONS[level],
            "escalate_to_supervisor": level == "HIGH" and (stakes >= 0.7 or (ev["voice_fake"] or 0) >= 0.85),
            "stakes": round(stakes, 2), "thresholds_used": {"medium": round(med_thr, 3), "high": round(high_thr, 3)},
            "evidence": {k: (None if v is None else round(v, 3)) for k, v in ev.items()},
            "contributions": contrib, "context_risk": round(context_risk, 3), "context_terms": ctx_terms,
            "rules_fired": [f[0] for f in fired], "reasons": reasons,
            "recommendations": self._recommend(level, ev, intent, sensitive_req),
            "missing_inputs": missing, "coverage": round(len(available) / 3.0, 2),
        }

    # ------------------------------------------------------------ explanations
    def _reasons(self, ev, intent, ctx, meta, fired, trusted):
        out = []
        if ev["voice_fake"] is not None and ev["voice_fake"] >= 0.5:
            out.append(f"Voice looks AI-generated or manipulated (spoof likelihood {ev['voice_fake']:.2f})")
        if ev["identity_mismatch"] is not None and ev["identity_mismatch"] >= 0.5:
            out.append(f"Voice does not match the claimed person (mismatch {ev['identity_mismatch']:.2f})")
        if ev["manipulation"] is not None and ev["manipulation"] >= 0.5:
            flags = [FLAG_TEXT.get(k, k) for k, v in intent.get("flags", {}).items() if v >= 0.5]
            out.append(f"Conversation matches scam patterns ({ev['manipulation']:.2f})" + (": caller " + ", ".join(flags) if flags else ""))
        if "high_value_amount" in ctx:
            out.append(f"High-value request (Rs {float(meta.get('transaction_amount', 0)):,.0f})")
        if "new_beneficiary" in ctx:
            out.append("Payment to a new beneficiary")
        if any(k.startswith("call_origin_") for k in ctx):
            out.append("Call comes from an unknown, withheld, international or VoIP source")
        if "flagged_number" in ctx:
            out.append("Number has been reported for fraud before")
        out += [f[1] for f in fired]
        if trusted:
            out.append("Trusted caller with a genuine-sounding voice (risk reduced)")
        return out or ["No strong risk signals found"]

    def _recommend(self, level, ev, intent, sensitive_req):
        if level == "LOW":
            return ["Proceed normally."]
        rec = ["Warn the agent/user that this caller may be manipulating them.",
               "Verify the caller through a second channel (call back on the number on file, or MFA) before any sensitive action."]
        if level == "HIGH":
            rec = ["Do NOT carry out the requested action yet.", "Force MFA or a call-back on the registered number.",
                   "Escalate to a supervisor / fraud team."]
        if ev.get("sensitive_ask") and ev["sensitive_ask"] >= 0.5:
            rec.append("Remind the user: never share OTP, PIN or passwords on a call.")
        if ev.get("voice_fake") is not None and ev["voice_fake"] >= 0.5:
            rec.append("Treat the voice as possibly cloned; do not rely on voice familiarity.")
        return rec
