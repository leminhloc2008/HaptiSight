---
title: YOLOER V2 Realtime Assist
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
python_version: "3.10"
pinned: false
---

# HaptiSight / YOLOER V2 Realtime Assist

Realtime assistive vision app for object reaching:
- YOLOE detection (target-focused prompt classes supported)
- MiDaS depth fusion for distance + 3D coordinates
- Hand tracking + contact/grasp state estimation
- Safety-aware guidance and voice output for realtime support
- Deployable on Hugging Face Spaces (CPU) or Modal (GPU)

## Project Structure

- `app.py`: main Gradio app and realtime inference pipeline
- `smart_agent.py`: Gemini planning/realtime guidance client
- `modal_app.py`: Modal deployment entrypoint
- `distance_estimation_core/`: distance regression assets and legacy core utilities used by runtime
- `requirements.txt`: dependencies

## Run Locally

Best option for speed and accuracy: run locally on your own computer.

```bash
pip install -r requirements.txt
python app.py
```

Then open the Gradio URL shown in terminal.

## If Modal Shows Usage Limit

When Modal usage is exhausted, the Modal web app shows a notice page and points to:

- Hugging Face fallback: `https://huggingface.co/spaces/lml2008/haptisight-realtime-gemini`

Important:

- Hugging Face Space runs on CPU, so realtime FPS can be slower.
- For better accuracy and performance, run this project locally (GPU preferred).

## Local Run (Recommended for Best Quality)

1. Install Python 3.10+.
2. Clone this repository.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set Gemini key (optional but recommended for full guidance):

```bash
export GEMINI_API_KEY=YOUR_KEY
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
```

5. Start app:

```bash
python app.py
```

6. Open the printed local URL in browser and allow camera + voice permissions.

## Hugging Face Space

Recommended:
- SDK: `Gradio`
- Startup file: `app.py`
- Python deps: `requirements.txt`
- Hardware: CPU (Pro tier recommended for better stability)

Push to Space:

```bash
git add .
git commit -m "deploy space"
git push
```

## Modal Deployment (GPU)

```bash
pip install -U modal
modal setup
modal deploy modal_app.py
```

Optional:

```bash
modal run modal_app.py::warmup
modal run modal_app.py::benchmark
```

## Important Environment Variables

- `YOLOE_MODEL_ID`: detector model id (for example `yoloe-11s`, `yoloe-11m`)
- `YOLOE_WEIGHTS`: custom `.pt` checkpoint path
- `FORCE_CPU`: force CPU mode (`1` or `0`)
- `CPU_THREADS`: torch CPU thread count
- `DEPTH_MODEL`: `MiDaS_small` or `DPT_Hybrid`
- `GEMINI_API_KEY`: Gemini API key (optional)
- `HAND_DETECT_MODE`: keep `mediapipe` (recommended realtime hand backend)

## Runtime Notes

- `30fps-stable` profile is tuned for smoother realtime behavior.
- `Balanced` and `Precision` trade speed for higher depth/detail quality.
- If browser voice is silent, click `Enable Camera Permission` or `Test Voice` once to unlock speech synthesis.
