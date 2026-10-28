r"""Turn the harness's enrolled voices into VoiceAuth voiceprints.

Usage (from the VoiceAuth_Local folder):
    python bridge_enroll.py "C:\path\to\tool-deepfake-testing\data\enrolled"

Understands both layouts:
    enrolled/<speaker_name>/*.wav|flac|mp3   -> one voiceprint per sub-folder (all its files are averaged)
    enrolled/<speaker_name>.wav              -> one voiceprint per file
Voiceprints are saved to voiceprints/<speaker_name>.pt. The request's speaker_id must equal <speaker_name>.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from voiceauth_pipeline import VOICEPRINT_DIR, VoiceAuthPipeline, safe_voiceprint_name  # noqa: E402

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


def collect(root):
    groups = {}
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(AUDIO_EXT)]
            if files:
                groups[entry] = files
        elif entry.lower().endswith(AUDIO_EXT):
            groups[os.path.splitext(entry)[0]] = [path]
    return groups


def main():
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        sys.exit(__doc__)
    groups = collect(sys.argv[1])
    if not groups:
        sys.exit("No audio files found in that folder.")
    pipe = VoiceAuthPipeline(use_spoof=False, use_intent=False)
    if pipe._speaker is None:
        sys.exit("Model 2 is not available: " + pipe.status.get("speaker", "?"))
    mod, embedder = pipe._speaker
    seg = pipe._speaker_cfg.get("seg_seconds", 6.0)
    os.makedirs(VOICEPRINT_DIR, exist_ok=True)
    for name, files in groups.items():
        safe = safe_voiceprint_name(name)
        if safe is None:
            print(f"skipped '{name}': use letters, digits, _ - . only in the name"); continue
        vp = mod.make_voiceprint(embedder, files, seg)
        out = os.path.join(VOICEPRINT_DIR, safe + ".pt")
        mod.save_voiceprint(out, vp, name=safe, n_files=len(files))
        print(f"{safe}: voiceprint from {len(files)} file(s) -> {out}" + ("   (only 1 recording: weaker voiceprint)" if len(files) == 1 else ""))


if __name__ == "__main__":
    main()
