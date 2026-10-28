"""Local test for Model 3 (call behavior / intent).

Folder must contain:
    intent_model.py         (downloaded from Drive: VoiceAuth_Project/models/intent_classifier/)
    head.npz                (same folder - the trained classifier)
    intent_config.json      (same folder)
    encoder/                (optional offline copy of the sentence encoder; otherwise it is
                             downloaded from Hugging Face on the first run - internet needed once)
    test_intent_local.py    (this file)

Install:  pip install torch transformers soundfile numpy torchaudio

Usage:
  1) Check a transcript you type (works in English, Hindi, Gujarati, Hinglish):
       python test_intent_local.py text "Sir tell me the OTP you just received or your account will be blocked"
  2) Check a transcript saved in a text file:
       python test_intent_local.py file transcript.txt
  3) Check a call recording (Whisper speech-to-text runs first; the first run downloads ~1 GB):
       python test_intent_local.py audio call.wav
       python test_intent_local.py audio call.wav --language hindi      (optional language hint)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for needed in ("intent_model.py", "head.npz", "intent_config.json"):
    if not os.path.exists(os.path.join(HERE, needed)):
        sys.exit(f"{needed} not found next to this script. Download it from Drive "
                 "(VoiceAuth_Project/models/intent_classifier/).")

from intent_model import IntentClassifier, Transcriber, analyze_audio  # noqa: E402


def level(p):
    return "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.3 else "LOW"


def show(res):
    if "transcript" in res:
        print(f"transcript : {res['transcript']}")
    print(f"scam probability : {res['scam_probability']:.2f}   -> manipulation risk: {level(res['scam_probability'])}")
    print(f"verdict          : {'SUSPICIOUS CALL' if res['is_scam'] else 'looks normal'}")
    trig = res["triggered_flags"]
    print(f"flags triggered  : {', '.join(trig) if trig else 'none'}")
    top = sorted(res["flags"].items(), key=lambda kv: -kv[1])[:4]
    print("top flag scores  : " + ", ".join(f"{k}={v:.2f}" for k, v in top))
    print(f"(analysed in {res['n_windows']} text window(s))\n")


def main():
    ap = argparse.ArgumentParser(description="Model 3 - call behavior / intent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("text"); p1.add_argument("text", nargs="+")
    p2 = sub.add_parser("file"); p2.add_argument("path")
    p3 = sub.add_parser("audio"); p3.add_argument("files", nargs="+"); p3.add_argument("--language", default=None)
    args = ap.parse_args()

    print("Loading the intent classifier ...")
    clf = IntentClassifier(HERE, device="cpu")

    if args.cmd == "text":
        show(clf.analyze_text(" ".join(args.text)))
    elif args.cmd == "file":
        show(clf.analyze_text(open(args.path, encoding="utf-8", errors="ignore").read()))
    else:
        print("Loading Whisper (first run downloads the model) ...")
        tr = Transcriber(device="cpu", source=clf.cfg.get("whisper_id", "openai/whisper-small"))
        for f in args.files:
            print(f"=== {os.path.basename(f)} ===")
            try:
                show(analyze_audio(tr, clf, f, language=args.language))
            except Exception as e:
                print(f"could not process file ({e})\n")


if __name__ == "__main__":
    main()
