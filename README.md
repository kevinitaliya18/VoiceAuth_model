# VoiceAuth_model - voice cloning / vishing detection (local setup)

One call goes through 3 models and a rule-based risk engine:

| Part | Question | Files |
|---|---|---|
| Model 1 - spoof detector | Is the voice real, cloned or converted? | models/spoof_detector/ |
| Model 2 - speaker verification | Is it the claimed person? | models/speaker_verification/ + voiceprints/ |
| Model 3 - call intent | Is the caller manipulating / asking for OTP, money, documents? | models/intent_classifier/ |
| Risk engine | LOW / MEDIUM / HIGH + action + reasons | risk_engine.py, risk_rules.json |

## 1. Setup (once)
1. Copy the trained files from Google Drive into the three `models/...` folders (see PUT_FILES_HERE.txt in each).
2. `pip install -r requirements.txt`
3. The first run downloads Whisper (~1 GB) and, if you did not copy the offline folders, the pretrained speaker and text encoders (~850 MB).

## 2. Test in this order
```bash
python test_pipeline.py scenarios                  # engine only - works even before the models are copied
python models/spoof_detector/test_local.py test_audio/call.wav
python test_pipeline.py call --audio test_audio/call.wav --voiceprint alice --meta meta_example.json     # edit meta_example.json for your case
python test_pipeline.py batch --manifest tests_example.csv --out batch_results.csv
```
A model whose files are missing is skipped and listed under "missing models"; the engine still works with the rest.

## 3. Run the API
```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000/docs to try `/analyze` (audio + metadata), `/engine` (model outputs only) and `/health`.

`/engine` expects model outputs using the same fields as the local models. For example:
```json
{
  "spoof": {"p_spoof": 0.12},
  "speaker": {"similarity": 0.91, "threshold": 0.86},
  "intent": {"scam_probability": 0.08, "flags": {}},
  "metadata": {"request_type": "none"}
}
```
Each model output may be omitted or set to `null`; if supplied, its required fields must be present. Invalid
payloads receive a `422` response with the missing or invalid field instead of an internal server error.

## 4. Tuning
Edit `risk_rules.json` (weights, thresholds, hard rules), then run `python test_pipeline.py scenarios`.
The defaults are expert-set starting points, not learned values. Build a CSV of your own test calls with the level you expect
(`tests_example.csv` shows the format) and use `batch` to see dangerous misses (expected HIGH, got LOW) and false alarms.

## Metadata fields the engine understands
request_type (fund_transfer / privileged_access / otp_disclosure / info_request / none), transaction_amount, new_beneficiary,
call_origin (known_contact / unknown / withheld / international / voip), flagged_number, after_hours, caller_trusted, claimed_role.

## Known limits
Trained on English datasets (ASVspoof 5); Model 3 trained on generated scripts (Hindi/Gujarati need native-speaker review);
Whisper-small is weak on Gujarati. Not validated on real phone calls. Use as a decision-support tool, not as the only control.

## 5. Connect the testing harness (tool-deepfake-testing)
New files: `harness_bridge.py` (adds POST /harness/analyze to the API), `bridge_config.json` (all names/mappings, editable),
`bridge_enroll.py`, `harness_config_snippet.yaml`.

1. (Optional) Turn the harness's enrolled voices into voiceprints:
   `python bridge_enroll.py <path-to-harness>/data/enrolled`   (the request's speaker_id must equal the voiceprint name)
2. Start the API (optionally set the same VISHING_API_KEY in both terminals; VOICEAUTH_USE_WHISPER=0 skips speech-to-text):
   `uvicorn api:app --host 127.0.0.1 --port 8000`
3. In the harness `config.yaml` use the values from `harness_config_snippet.yaml` (adapter: http, url .../harness/analyze,
   timeout 120, similarity_threshold = the "threshold" in speaker_config.json). Install FFmpeg for the robustness test.
4. In the harness folder run `python cli.py check-connection --adapter http`. It lists missing/unexpected fields.
   Fix them by editing `bridge_config.json` (output_names, flags_format, action_map, context_map ...). The file is re-read on
   every request, so no restart is needed. `GET http://127.0.0.1:8000/harness/contract` shows the active mapping.
5. Then run the harness: `single`, `robustness` (MP3 128k/64k/32k), and the scenario files.
