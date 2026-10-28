"""Local test for Model 2 (speaker verification).

Folder must contain:
    speaker_model.py        (downloaded from Drive: VoiceAuth_Project/models/speaker_verification/)
    speaker_config.json     (same Drive folder - holds the calibrated threshold)
    wavlm_sv/               (optional offline copy of the pretrained model; without it the model
                             is downloaded from Hugging Face on the first run, so internet is needed once)
    test_speaker_local.py   (this file)

Install:  pip install torch transformers soundfile numpy torchaudio

Usage:
  1) Enroll a person (3 to 5 clean recordings of the same person work best):
       python test_speaker_local.py enroll --name alice --out alice.pt voice1.wav voice2.wav voice3.wav
  2) Check a call recording against that voiceprint:
       python test_speaker_local.py verify --voiceprint alice.pt call.wav
  3) Quick comparison of two recordings:
       python test_speaker_local.py compare a.wav b.wav
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(HERE, "speaker_model.py")):
    sys.exit("speaker_model.py not found next to this script. Download it from Drive "
             "(VoiceAuth_Project/models/speaker_verification/).")

from speaker_model import (MODEL_ID, SpeakerEmbedder, load_config, load_voiceprint,  # noqa: E402
                           make_voiceprint, save_voiceprint, similarity, verify_file)


def get_embedder():
    local = os.path.join(HERE, "wavlm_sv")
    source = local if os.path.isdir(local) else MODEL_ID
    print(f"Loading speaker model from: {'local folder' if source == local else 'Hugging Face (first run downloads ~380 MB)'}")
    return SpeakerEmbedder(device="cpu", source=source)


def main():
    cfg = load_config(HERE)
    threshold, seg = float(cfg["threshold"]), float(cfg.get("seg_seconds", 6.0))
    if not cfg.get("calibrated", False):
        print("WARNING: speaker_config.json not found, using the default threshold "
              "(not calibrated). Copy speaker_config.json from Drive for proper results.\n")

    ap = argparse.ArgumentParser(description="Model 2 - speaker verification")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("enroll"); p1.add_argument("files", nargs="+")
    p1.add_argument("--out", required=True); p1.add_argument("--name", default="")
    p2 = sub.add_parser("verify"); p2.add_argument("files", nargs="+"); p2.add_argument("--voiceprint", required=True)
    p3 = sub.add_parser("compare"); p3.add_argument("file_a"); p3.add_argument("file_b")
    args = ap.parse_args()

    emb = get_embedder()

    if args.cmd == "enroll":
        vp = make_voiceprint(emb, args.files, seg)
        save_voiceprint(args.out, vp, name=args.name, n_files=len(args.files))
        print(f"Voiceprint saved to {args.out} (built from {len(args.files)} recordings)")

    elif args.cmd == "verify":
        vp, meta = load_voiceprint(args.voiceprint)
        print(f"Claimed speaker: {meta.get('name') or args.voiceprint} | threshold {threshold:.3f}\n")
        for f in args.files:
            try:
                r = verify_file(emb, vp, f, threshold, seg)
            except Exception as e:
                print(f"{os.path.basename(f)}: could not read file ({e})"); continue
            verdict = "SAME PERSON (match)" if r["is_match"] else "DIFFERENT PERSON (no match)"
            print(f"{os.path.basename(f)}\n   verdict    : {verdict}\n   similarity : {r['similarity']:.3f}   (threshold {threshold:.3f})\n")

    elif args.cmd == "compare":
        a, b = emb.embed_file(args.file_a, seg), emb.embed_file(args.file_b, seg)
        sim = similarity(a, b)
        print(f"similarity: {sim:.3f}  (threshold {threshold:.3f}) -> "
              f"{'SAME PERSON' if sim > threshold else 'DIFFERENT PERSON'}")


if __name__ == "__main__":
    main()
