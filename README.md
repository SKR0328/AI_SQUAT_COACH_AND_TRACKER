# LiftLens — local-first squat coach

LiftLens analyzes uploaded squat clips and optional live camera sessions on your computer. MediaPipe tracks body landmarks and estimates reps, depth, and selected movement patterns. An optional local Qwen3-VL model reviews a recorded set and offers practical coaching cues after capture.

LiftLens is a training aid, not an official powerlifting referee or a medical tool. A single 2-D camera view cannot reliably identify the official visible hip-crease and top-of-knee landmarks, diagnose an injury, or verify every form issue. Treat its results as prompts for review, not definitive judgments.

## Features

- Analyze a prerecorded squat video and export a full-clip pose-tracking preview.
- Run an opt-in browser-camera session with live pose, rep, phase, and depth estimates.
- Record a local replay and request post-set coaching from the local Qwen3-VL model.
- Keep video decoding, pose estimation, and language-model inference on the computer.

The live metrics use MediaPipe and rule-based logic; Qwen3-VL does **not** coach on every live frame. For recorded sessions, stop the set and choose **Get local post-set coaching**. For uploaded clips, choose **Review video with local Qwen3-VL coach**.

## Stack

- Python, Streamlit, and Streamlit WebRTC
- OpenCV and MediaPipe Pose for local video and landmark processing
- NumPy and Pillow for analysis and image handling
- Qwen3-VL 2B Instruct GGUF with `llama.cpp` `llama-mtmd-cli` for local video coaching
- FFmpeg for local video conversion and browser-compatible H.264 tracking exports
- Optional CUDA acceleration 

The Windows download script fetches the Qwen model and vision projector, a CUDA-enabled `llama.cpp` runtime, and FFmpeg. It downloads several gigabytes; the model and runtime files are excluded from Git. The pose landmarker file in `models/` is small and distributed under Apache 2.0; see its [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf).

## Windows setup

The project was developed on Windows with Python 3.14. Use a Python version for which the pinned MediaPipe version has a wheel.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_local_video_coach.py
.\start.ps1
```

The download step fetches the local video-coaching model and its runtime. It is optional for pose analysis; without it, video coaching is unavailable. The first model review can take a few minutes, especially on a low-memory . The current app flow uses `llama.cpp`; it does not require a running Ollama service.

To start the pose-analysis app without the launcher, run:

```powershell
streamlit run app.py
```

## How the estimates work

- The pose pipeline samples video frames locally and uses visible landmarks to estimate rep cycles and movement.
- Depth is an approximate 2-D hip/knee joint-center proxy. The result may be marked reached when a sufficiently visible side crosses the app's threshold; unclear cases are sent for visual review. The coach does not receive separate left/right proxy measurements.
- The depth proxy is not the official IPF hip-crease versus top-of-knee comparison, and its review band is an engineering heuristic rather than a validated judging tolerance.
- The local video coach reviews a time-ordered clip and selected raw frames alongside rep timing and movement signals. It can miss or misread a fault; it should not invent a correction when evidence is weak.
- Live camera metrics are separate from the Qwen review. Recording must be enabled before starting; after stopping, use the post-set coaching button. The temporary unannotated recording is deleted after coaching or when a later live session starts.

For best results, use one lifter in a fixed side view with the full body visible. A side view cannot reliably establish knee tracking over the foot, and visible torso lean alone does not prove spinal rounding or butt wink.

## Tests

Run the local rep-quality checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These checks cover synthetic movement cycles and edge cases; they do not replace evaluation on human-labeled squat footage.

## Privacy

Uploaded and recorded videos are processed locally and are not sent to a cloud model. Coaching normalizes video to a temporary audio-free file for local inference, then removes that temporary copy. The app only starts camera capture after browser permission is granted.

**MediaPipe Tasks telemetry:** MediaPipe's [privacy notice](https://github.com/google-ai-edge/mediapipe#privacy-notice) says the Tasks API may send performance and usage metrics to Google. It says input images and video are processed on-device and are not sent to Google. The Python Tasks wrapper used by this project does not expose a documented metrics opt-out, so the runtime is not strictly network-silent.

## Licensing

This repository does not yet include a project-level license. Choose and add one before inviting others to reuse the application. The Qwen model, MediaPipe model, `llama.cpp`, and FFmpeg have their own separate license terms.
