"""Testing tool for the whole VoiceAuth system.

  python test_pipeline.py scenarios
        Runs built-in what-if scenarios through the RISK ENGINE only (no audio, no models needed).
        Use it to check that the rules behave sensibly after you edit risk_rules.json.

  python test_pipeline.py call --audio test_audio/call1.wav --voiceprint alice \
        --meta "{\"request_type\":\"fund_transfer\",\"transaction_amount\":400000,\"new_beneficiary\":true}"
        Runs ONE recording through Model 1 + Model 2 + Model 3 + the engine.
        (--meta accepts JSON text or a path to a .json file; --transcript skips Whisper; --voiceprint is optional)

  python test_pipeline.py batch --manifest tests.csv --out batch_results.csv
        Runs many calls listed in a CSV and compares the result with the level you expected.
        CSV columns (all optional except audio or transcript):
          audio, voiceprint, transcript, language, expected_level (LOW/MEDIUM/HIGH),
          request_type, transaction_amount, new_beneficiary, call_origin, flagged_number,
          after_hours, caller_trusted, claimed_role
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from risk_engine import LEVELS, RiskEngine  # noqa: E402


def sp(p): return {"p_spoof": p}
def sk(sim, thr=0.86): return {"similarity": sim, "threshold": thr}
def it(p, **flags): return {"scam_probability": p, "flags": flags}


SCENARIOS = [
    ("normal colleague call", "LOW", sp(0.04), sk(0.93), it(0.05), {"call_origin": "known_contact", "caller_trusted": True}),
    ("legit high-value transfer by the real customer", "LOW", sp(0.05), sk(0.92), it(0.08), {"request_type": "fund_transfer", "transaction_amount": 1000000, "new_beneficiary": True, "call_origin": "known_contact"}),
    ("CEO voice-clone fraud", "HIGH", sp(0.96), sk(0.55), it(0.75, urgency_pressure=0.8, secrecy_request=0.7, asks_money_transfer=0.8), {"request_type": "fund_transfer", "transaction_amount": 500000, "new_beneficiary": True, "call_origin": "voip", "claimed_role": "ceo"}),
    ("family emergency scam with cloned voice (no voiceprint)", "HIGH", sp(0.55), None, it(0.92, emotional_emergency=0.9, asks_money_transfer=0.85, secrecy_request=0.7), {"request_type": "fund_transfer", "transaction_amount": 50000, "new_beneficiary": True, "call_origin": "unknown"}),
    ("human vishing call asking for OTP", "HIGH", sp(0.10), None, it(0.95, asks_otp_password=0.92, authority_claim=0.9, urgency_pressure=0.8), {"request_type": "otp_disclosure", "call_origin": "unknown"}),
    ("bank safety notice that Model 3 over-scores", "LOW", sp(0.05), None, it(0.69, asks_otp_password=0.3), {"call_origin": "known_contact", "caller_trusted": True}),
    ("AI-generated voice but harmless content", "MEDIUM", sp(0.90), None, it(0.10), {"call_origin": "known_contact"}),
    ("pressured but plausible caller", "MEDIUM", sp(0.08), sk(0.90), it(0.60, urgency_pressure=0.7), {"request_type": "info_request", "call_origin": "unknown"}),
    ("no evidence at all", "LOW", None, None, None, {}),
    ("voice does not match, big transfer requested", "HIGH", sp(0.10), sk(0.60), it(0.10), {"request_type": "fund_transfer", "transaction_amount": 200000}),
    ("text-only scam, no audio, payment requested", "HIGH", None, None, it(0.90, asks_money_transfer=0.8, urgency_pressure=0.8), {"request_type": "fund_transfer", "transaction_amount": 25000}),
    ("calm unknown caller, big transfer, voice not checked", "MEDIUM", None, None, None, {"request_type": "fund_transfer", "transaction_amount": 500000, "call_origin": "unknown"}),
    ("everyday call from an unknown number", "LOW", sp(0.05), None, it(0.10), {"call_origin": "unknown"}),
    ("scam-like text, low stakes, no OTP ask", "MEDIUM", sp(0.05), None, it(0.95, urgency_pressure=0.8, threat_fear=0.7), {"call_origin": "unknown"}),
    ("flagged number, casual talk", "LOW", sp(0.06), None, it(0.15), {"flagged_number": True, "call_origin": "unknown"}),
]


def to_bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def load_meta(arg):
    if not arg:
        return {}
    if os.path.exists(arg):
        return json.load(open(arg, encoding="utf-8"))
    return json.loads(arg)


def show(result):
    a = result["assessment"]
    print(f"\nRISK LEVEL : {a['risk_level']}   (score {a['risk_score']:.2f}, stakes {a['stakes']:.2f})")
    print(f"ACTION     : {a['action']}" + ("   + escalate to supervisor" if a["escalate_to_supervisor"] else ""))
    print("WHY        :"); [print("  -", r) for r in a["reasons"]]
    print("WHAT TO DO :"); [print("  -", r) for r in a["recommendations"]]
    m = result["models"]
    print("MODEL OUTPUTS:")
    if m["spoof"]: print(f"  Model 1 spoof   : p_spoof={m['spoof']['p_spoof']:.2f} ({'FAKE' if m['spoof']['is_spoof'] else 'real'})")
    if m["speaker"]: print(f"  Model 2 speaker : similarity={m['speaker']['similarity']:.2f} vs threshold {m['speaker']['threshold']:.2f} ({'match' if m['speaker']['is_match'] else 'MISMATCH'})")
    if m["intent"]: print(f"  Model 3 intent  : scam={m['intent']['scam_probability']:.2f}, flags={m['intent']['triggered_flags']}")
    if m["transcript"]: print(f"  transcript      : {m['transcript'][:140]}")
    if a["missing_inputs"]: print(f"  missing models  : {a['missing_inputs']} (coverage {a['coverage']})")
    for n in result["notes"]: print("  note:", n)
    if result["timings"]: print("  timings:", result["timings"])


def cmd_scenarios(args):
    eng = RiskEngine(os.path.join(HERE, "risk_rules.json"))
    ok = 0
    print(f"{'scenario':58s} {'expected':9s} {'got':7s} score  rules fired")
    for name, exp, spoof, speaker, intent, meta in SCENARIOS:
        r = eng.assess(spoof, speaker, intent, meta)
        good = r["risk_level"] == exp; ok += good
        print(f"{'OK ' if good else 'BAD'} {name:54s} {exp:9s} {r['risk_level']:7s} {r['risk_score']:.2f}   {', '.join(r['rules_fired'])}")
    print(f"\n{ok}/{len(SCENARIOS)} scenarios gave the expected level")


def cmd_call(args):
    from voiceauth_pipeline import VoiceAuthPipeline
    p = VoiceAuthPipeline(device=args.device, use_whisper=not args.no_whisper)
    print("Model status:", json.dumps(p.status, indent=1))
    show(p.analyze(audio_path=args.audio, metadata=load_meta(args.meta), voiceprint=args.voiceprint,
                   transcript=args.transcript, language=args.language))


def cmd_batch(args):
    from voiceauth_pipeline import VoiceAuthPipeline
    p = VoiceAuthPipeline(device=args.device, use_whisper=not args.no_whisper)
    print("Model status:", json.dumps(p.status, indent=1))
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    out_rows, confusion, n_exp = [], {(e, g): 0 for e in LEVELS for g in LEVELS}, 0
    base_dir = os.path.dirname(os.path.abspath(args.manifest))
    for i, row in enumerate(rows, 1):
        audio = row.get("audio") or None
        if audio and not os.path.isabs(audio):
            audio = os.path.join(base_dir, audio)
        meta = {}
        for k in ("request_type", "call_origin", "claimed_role"):
            if row.get(k): meta[k] = row[k]
        if row.get("transaction_amount"): meta["transaction_amount"] = float(row["transaction_amount"])
        for k in ("new_beneficiary", "flagged_number", "after_hours", "caller_trusted"):
            if row.get(k): meta[k] = to_bool(row[k])
        res = p.analyze(audio_path=audio, metadata=meta, voiceprint=row.get("voiceprint") or None,
                        transcript=row.get("transcript") or None, language=row.get("language") or None)
        a, m = res["assessment"], res["models"]
        exp = (row.get("expected_level") or "").upper()
        if exp in LEVELS:
            n_exp += 1; confusion[(exp, a["risk_level"])] += 1
        out_rows.append({"row": i, "audio": row.get("audio", ""), "expected": exp, "level": a["risk_level"],
                         "score": a["risk_score"], "action": a["action"],
                         "p_spoof": None if not m["spoof"] else round(m["spoof"]["p_spoof"], 3),
                         "similarity": None if not m["speaker"] else round(m["speaker"]["similarity"], 3),
                         "scam_prob": None if not m["intent"] else round(m["intent"]["scam_probability"], 3),
                         "reasons": " | ".join(a["reasons"])})
        print(f"[{i}/{len(rows)}] {os.path.basename(audio or 'transcript only'):30s} expected={exp or '-':6s} got={a['risk_level']:6s} score={a['risk_score']:.2f}")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    print(f"\nResults saved to {args.out}")
    if n_exp:
        correct = sum(confusion[(l, l)] for l in LEVELS)
        print(f"\nExact level matches: {correct}/{n_exp} ({100 * correct / n_exp:.0f}%)")
        print("Rows = expected, columns = got:")
        print("          " + "".join(f"{g:>8s}" for g in LEVELS))
        for e in LEVELS:
            print(f"{e:9s} " + "".join(f"{confusion[(e, g)]:8d}" for g in LEVELS))
        print(f"Dangerous misses (expected HIGH, got LOW): {confusion[('HIGH', 'LOW')]}")
        print(f"False alarms (expected LOW, got HIGH)    : {confusion[('LOW', 'HIGH')]}")


def main():
    ap = argparse.ArgumentParser(description="VoiceAuth testing tool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scenarios")
    c = sub.add_parser("call")
    c.add_argument("--audio"); c.add_argument("--voiceprint"); c.add_argument("--meta")
    c.add_argument("--transcript"); c.add_argument("--language")
    b = sub.add_parser("batch")
    b.add_argument("--manifest", required=True); b.add_argument("--out", default="batch_results.csv")
    for s in (c, b):
        s.add_argument("--device", default="cpu"); s.add_argument("--no-whisper", action="store_true")
    args = ap.parse_args()
    {"scenarios": cmd_scenarios, "call": cmd_call, "batch": cmd_batch}[args.cmd](args)


if __name__ == "__main__":
    main()
