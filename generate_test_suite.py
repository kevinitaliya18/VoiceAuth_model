import asyncio
import csv
import os
import tempfile
import edge_tts
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

# Output directories matching VoiceAuth_Local layout
TEST_AUDIO_DIR = "test_audio"
ENROLL_DIR = os.path.join(TEST_AUDIO_DIR, "enrolled", "alice")
MANIFEST_PATH = "tests_generated.csv"
SAMPLE_RATE = 16000  # Standard 16kHz for WavLM & ASVspoof

os.makedirs(TEST_AUDIO_DIR, exist_ok=True)
os.makedirs(ENROLL_DIR, exist_ok=True)

# 1. Voice Enrollment Samples for "alice" (needs >6.0s for seg_seconds in speaker_config)
ENROLL_SAMPLES = [
    (
        os.path.join(ENROLL_DIR, "enroll_01.wav"),
        "en-US-AriaNeural",
        "Hello, my name is Alice and I am recording my voiceprint for the company security authentication system. "
        "This sample contains clear speech across several seconds so the WavLM speaker embedder can extract a reliable vector."
    ),
    (
        os.path.join(ENROLL_DIR, "enroll_02.wav"),
        "en-US-AriaNeural",
        "I work in the finance department and regularly review quarterly budgets, vendor invoices, and internal audit reports. "
        "Please verify my voice identity before authorizing any sensitive account changes."
    ),
]

# 2. Test Scenarios (Audio + Metadata for test_pipeline.py batch)
TEST_CASES = [
    {
        "filename": "01_normal_talk.wav",
        "voice": "en-US-AriaNeural",  # Matches enrolled "alice"
        "degrade": None,
        "text": (
            "Hey team, it is Alice. I am just calling to confirm that our weekly project sync is still happening "
            "at three o'clock this afternoon. Let me know if you need me to bring the updated spreadsheet. Talk to you soon."
        ),
        "voiceprint": "alice",
        "expected_level": "LOW",
        "request_type": "none",
        "transaction_amount": "",
        "new_beneficiary": "false",
        "call_origin": "known_contact",
        "flagged_number": "false",
        "after_hours": "false",
        "caller_trusted": "true",
        "claimed_role": "colleague",
    },
    {
        "filename": "02_ceo_fraud_clone.wav",
        "voice": "en-US-AriaNeural",  # Cloned voice of "alice"
        "degrade": "codec_artifact",
        "text": (
            "Listen carefully, I am about to walk into an executive board meeting and I need an urgent wire transfer "
            "of five hundred thousand rupees completed right now to a new vendor account. Keep this strictly confidential "
            "and do not discuss it with anyone else until the acquisition is announced."
        ),
        "voiceprint": "alice",
        "expected_level": "HIGH",
        "request_type": "fund_transfer",
        "transaction_amount": "500000",
        "new_beneficiary": "true",
        "call_origin": "voip",
        "flagged_number": "false",
        "after_hours": "true",
        "caller_trusted": "false",
        "claimed_role": "ceo",
    },
    {
        "filename": "03_vishing_otp_scam.wav",
        "voice": "en-IN-PrabhatNeural",  # Indian English male authority voice
        "degrade": "phone_bandpass",
        "text": (
            "This is the Cyber Fraud Department from your bank headquarters. We detected an unauthorized international transaction "
            "on your savings account. Your account will be permanently frozen within ten minutes unless you verify your identity. "
            "Please read out the six digit OTP verification code sent to your mobile phone immediately, and click the security link we texted you."
        ),
        "voiceprint": "",
        "expected_level": "HIGH",
        "request_type": "otp_disclosure",
        "transaction_amount": "",
        "new_beneficiary": "false",
        "call_origin": "unknown",
        "flagged_number": "true",
        "after_hours": "false",
        "caller_trusted": "false",
        "claimed_role": "bank_employee",
    },
    {
        "filename": "04_complex_family_emergency.wav",
        "voice": "en-US-JennyNeural",
        "degrade": "noisy_phone",
        "text": (
            "Please help me, I was just in a terrible car accident and my phone broke so I am calling from a stranger's number. "
            "The hospital will not start the emergency surgery until I pay a deposit of fifty thousand rupees right now. "
            "Please transfer the money immediately to the doctor's UPI account and do not tell mom yet, she will panic."
        ),
        "voiceprint": "",
        "expected_level": "HIGH",
        "request_type": "fund_transfer",
        "transaction_amount": "50000",
        "new_beneficiary": "true",
        "call_origin": "unknown",
        "flagged_number": "false",
        "after_hours": "true",
        "caller_trusted": "false",
        "claimed_role": "family",
    },
    {
        "filename": "05_complex_harmless_ai_notice.wav",
        "voice": "en-IN-NeerjaNeural",
        "degrade": None,
        "text": (
            "Hello, this is an automated security reminder from your bank. Please remember that bank officials will never ask "
            "for your OTP, PIN, or internet banking password over a phone call. Have a safe and wonderful day."
        ),
        "voiceprint": "",
        "expected_level": "MEDIUM",  # Tests corroboration_cap: AI voice (high spoof) + low stakes -> capped at MEDIUM
        "request_type": "none",
        "transaction_amount": "",
        "new_beneficiary": "false",
        "call_origin": "known_contact",
        "flagged_number": "false",
        "after_hours": "false",
        "caller_trusted": "false",
        "claimed_role": "bank_employee",
    },
    {
        "filename": "06_complex_noisy_mismatch.wav",
        "voice": "en-US-GuyNeural",  # Wrong speaker claiming to be "alice"
        "degrade": "noisy_phone",
        "text": (
            "Hi, this is Alice calling from a remote line. Can you please go ahead and approve the two hundred thousand rupee "
            "payment to our new supplier account? Everything looks fine on my end."
        ),
        "voiceprint": "alice",
        "expected_level": "HIGH",  # Triggers identity_mismatch_sensitive hard rule
        "request_type": "fund_transfer",
        "transaction_amount": "200000",
        "new_beneficiary": "true",
        "call_origin": "unknown",
        "flagged_number": "false",
        "after_hours": "false",
        "caller_trusted": "false",
        "claimed_role": "colleague",
    },
]


def apply_audio_effects(y, sr, effect_type):
    """Simulates real-world telephony bandwidth, background noise, or vocoder artifacts."""
    if effect_type == "phone_bandpass":
        # 300Hz - 3400Hz telephone bandpass filter
        nyq = 0.5 * sr
        b, a = butter(4, [300.0 / nyq, 3400.0 / nyq], btype="band")
        y = lfilter(b, a, y)
    elif effect_type == "noisy_phone":
        nyq = 0.5 * sr
        b, a = butter(4, [300.0 / nyq, 3400.0 / nyq], btype="band")
        y = lfilter(b, a, y)
        noise = np.random.normal(0, 0.008, len(y))
        y = y + noise
    elif effect_type == "codec_artifact":
        # Downsample to 8kHz and back to 16kHz + slight quantization to mimic VoIP compression
        y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
        y = librosa.resample(y_8k, orig_sr=8000, target_sr=sr)
        y = np.round(y * 128.0) / 128.0

    # Normalize amplitude to prevent clipping
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val * 0.9
    return y.astype(np.float32)


async def synth_to_wav(text, voice, out_path, degrade=None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp_mp3 = tmp.name
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_mp3)
        # Convert to 16kHz Mono PCM_16 WAV
        y, _ = librosa.load(tmp_mp3, sr=SAMPLE_RATE, mono=True)
        if degrade:
            y = apply_audio_effects(y, SAMPLE_RATE, degrade)
        sf.write(out_path, y, SAMPLE_RATE, subtype="PCM_16")
        print(f"[OK] Saved {out_path} ({len(y)/SAMPLE_RATE:.1f}s, 16kHz PCM_16)")
    finally:
        if os.path.exists(tmp_mp3):
            os.remove(tmp_mp3)


async def main():
    print("--- 1. Generating Enrollment Audios for 'alice' ---")
    for path, voice, text in ENROLL_SAMPLES:
        await synth_to_wav(text, voice, path)

    print("\n--- 2. Generating Test Suite Audios ---")
    csv_rows = []
    for case in TEST_CASES:
        wav_path = os.path.join(TEST_AUDIO_DIR, case["filename"])
        await synth_to_wav(case["text"], case["voice"], wav_path, degrade=case["degrade"])
        csv_rows.append({
            "audio": wav_path,
            "voiceprint": case["voiceprint"],
            "transcript": "",  # Leave blank so Whisper transcribes the audio automatically
            "language": "en",
            "expected_level": case["expected_level"],
            "request_type": case["request_type"],
            "transaction_amount": case["transaction_amount"],
            "new_beneficiary": case["new_beneficiary"],
            "call_origin": case["call_origin"],
            "flagged_number": case["flagged_number"],
            "after_hours": case["after_hours"],
            "caller_trusted": case["caller_trusted"],
            "claimed_role": case["claimed_role"],
        })

    print(f"\n--- 3. Writing Batch Manifest to {MANIFEST_PATH} ---")
    fieldnames = list(csv_rows[0].keys())
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[DONE] Generated {len(csv_rows)} test files and {MANIFEST_PATH}")


if __name__ == "__main__":
    asyncio.run(main())