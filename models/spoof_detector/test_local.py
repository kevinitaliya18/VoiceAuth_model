"""Local test for Model 1 (spoof detector).

Folder must contain:
    spoof_model.py              (downloaded from Drive)
    spoof_detector_final.pt     (downloaded from Drive)
    test_local.py               (this file)

Usage:
    python test_local.py my_call.wav
    python test_local.py folder_with_audio_files
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "spoof_detector_final.pt")
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

if not os.path.exists(os.path.join(HERE, "spoof_model.py")):
    sys.exit("spoof_model.py not found next to test_local.py. Download it from Drive "
             "(VoiceAuth_Project/models/spoof_detector/).")
if not os.path.exists(MODEL_PATH):
    sys.exit("spoof_detector_final.pt not found next to test_local.py. Download it from Drive "
             "(VoiceAuth_Project/models/spoof_detector/).")

from spoof_model import load_model, predict_file  # noqa: E402


def collect_files(target):
    if os.path.isdir(target):
        return sorted(os.path.join(target, f) for f in os.listdir(target)
                      if f.lower().endswith(AUDIO_EXTS))
    return [target]


def main():
    if len(sys.argv) < 2:
        sys.exit("Give an audio file or folder, e.g.:  python test_local.py my_call.wav")

    files = collect_files(sys.argv[1])
    if not files:
        sys.exit("No audio files found.")

    print("Loading model ...")
    model, cfg = load_model(MODEL_PATH, device="cpu")
    print(f"Model loaded. Decision threshold: {cfg.get('threshold', 0.0):.3f}\n")

    for path in files:
        try:
            r = predict_file(model, cfg, path, device="cpu")
        except Exception as e:
            print(f"{os.path.basename(path)}: could not read file ({e})")
            continue
        verdict = "FAKE / AI-GENERATED" if r["is_spoof"] else "REAL HUMAN VOICE"
        print(f"{os.path.basename(path)}")
        print(f"   verdict : {verdict}")
        print(f"   p_spoof : {r['p_spoof']:.3f}   (0 = surely real, 1 = surely fake)")
        print(f"   score   : {r['score']:.3f}   (threshold {r['threshold']:.3f})\n")


if __name__ == "__main__":
    main()
