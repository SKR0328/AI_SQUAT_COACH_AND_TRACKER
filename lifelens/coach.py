"""Grounded squat coaching with local pose signals and Qwen3-VL via llama.cpp."""

import json
import base64
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Dict

import cv2
import numpy as np
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:4b"
VISION_MODEL = "qwen3-vl:4b-instruct"
COACH_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "cue": {"type": "string"},
        "why": {"type": "string"},
        "limitation": {"type": "string"},
    },
    "required": ["headline", "cue", "why", "limitation"],
    "additionalProperties": False,
}

VISUAL_COACH_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "primary_cue": {"type": "string"},
        "reasoning": {"type": "string"},
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rep": {"type": ["integer", "null"]},
                    "timestamp_s": {"type": ["number", "null"]},
                    "area": {"type": "string", "enum": ["depth", "back_position", "trunk", "knees", "tempo", "setup", "other"]},
                    "observation": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "cue": {"type": "string"},
                },
                "required": ["rep", "timestamp_s", "area", "observation", "confidence", "cue"],
                "additionalProperties": False,
            },
        },
        "depth_note": {"type": "string"},
        "limitation": {"type": "string"},
        "next_set": {"type": "string"},
    },
    "required": ["headline", "primary_cue", "reasoning", "observations", "depth_note", "limitation", "next_set"],
    "additionalProperties": False,
}

MAX_COACH_IMAGES = 2
MAX_VIDEO_COACH_IMAGES = 6
MAX_COACH_IMAGE_SIDE = 320


def _video_runtime_paths():
    project_root = Path(__file__).resolve().parent.parent
    runtime = project_root / ".runtime"
    executable = runtime / "llama" / "llama-mtmd-cli.exe"
    ffmpeg_dir = runtime / "ffmpeg" / "bin"
    model = project_root / "models" / "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    mmproj = project_root / "models" / "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    missing = [path for path in (executable, ffmpeg_dir / "ffmpeg.exe", ffmpeg_dir / "ffprobe.exe", model, mmproj)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The local full-video coach runtime is incomplete. Run "
            "`.venv\\Scripts\\python.exe scripts\\download_local_video_coach.py` and retry."
        )
    return project_root, executable, ffmpeg_dir, model, mmproj


def _coach_prompt_facts(facts: Dict) -> Dict:
    """Allow only set timing and coarse whole-body movement signals into the prompt."""
    facts = dict(facts or {})
    timeline = facts.get("rep_timeline_seconds")
    if not isinstance(timeline, list):
        bottoms = facts.get("detected_rep_bottom_times_seconds")
        timeline = ([{"rep": i + 1, "bottom_seconds": round(float(value), 2)}
                    for i, value in enumerate(bottoms[:8])
                    if isinstance(value, (int, float))]
                   if isinstance(bottoms, list) else [])
    safe_timeline = []
    for item in timeline[:8]:
        if not isinstance(item, dict):
            continue
        safe_item = {key: item[key] for key in (
            "rep", "descent_start_seconds", "bottom_seconds", "standing_return_seconds"
        ) if isinstance(item.get(key), (int, float))}
        if safe_item:
            safe_timeline.append(safe_item)
    rep_count = facts.get("rep_count_estimate", len(safe_timeline))
    allowed_signals = {"possible_hips_first", "no_clear_hips_first_signal", "unclear"}
    movement_signals = []
    raw_signals = facts.get("movement_signals")
    if isinstance(raw_signals, list):
        for signal in raw_signals[:8]:
            if not isinstance(signal, dict):
                continue
            label = signal.get("ascent_coordination")
            rep = signal.get("rep")
            if label in allowed_signals:
                movement_signals.append({
                    "rep": int(rep) if isinstance(rep, (int, float)) else len(movement_signals) + 1,
                    "ascent_coordination": label,
                })
    return {
        "exercise": "barbell squat",
        "rep_count_estimate": int(rep_count) if isinstance(rep_count, (int, float)) else len(safe_timeline),
        "rep_timeline_seconds": safe_timeline,
        "movement_signals": movement_signals,
    }


def _movement_signal_coach(movement_signals):
    """Create a conservative set-level response if visual model text is unusable."""
    labels = [item.get("ascent_coordination") for item in movement_signals if isinstance(item, dict)]
    possible_reps = [
        int(item["rep"]) for item in movement_signals
        if isinstance(item, dict) and item.get("ascent_coordination") == "possible_hips_first"
        and isinstance(item.get("rep"), (int, float))
    ]
    if possible_reps:
        rep_names = [f"rep {rep}" for rep in possible_reps]
        if len(rep_names) == 1:
            rep_text = rep_names[0]
        elif len(rep_names) == 2:
            rep_text = f"{rep_names[0]} and {rep_names[1]}"
        else:
            rep_text = ", ".join(rep_names[:-1]) + f", and {rep_names[-1]}"
        other_reps = [
            int(item["rep"]) for item in movement_signals
            if isinstance(item, dict) and item.get("ascent_coordination") != "possible_hips_first"
            and isinstance(item.get("rep"), (int, float))
        ]
        if other_reps:
            other_text = ", ".join(f"rep {rep}" for rep in other_reps)
            reasoning = (
                f"On {rep_text}, your hips may start rising before your chest. "
                f"I couldn't confirm the same pattern on {other_text}, so treat it as a point to practice, not a confirmed fault."
            )
            observation = f"Your hips may rise ahead of your chest on {rep_text}; the other rep timing was unclear."
        else:
            reasoning = (
                f"On {rep_text}, your hips may start rising before your chest. "
                "Treat it as a technique point to check, not a confirmed fault."
            )
            observation = f"Your hips may rise ahead of your chest on {rep_text}."
        cue = "Brace before you descend, then try to bring your chest and hips up together out of the bottom."
        drill = (
            "Optional drill: paused squats, 2 sets of 3 at bodyweight or an easy familiar load. "
            "Pause briefly at a comfortable bottom, brace, then stand while keeping your chest and hips rising together."
        )
        return {
            "headline": "One ascent timing point to check",
            "primary_cue": cue,
            "reasoning": reasoning,
            "next_set": cue,
            "drill": drill,
            "limitation": "This is a 2-D pose estimate; the possible timing pattern is a practice cue, not a confirmed fault.",
            "observations": [{
                "rep": possible_reps[0], "timestamp_s": None, "area": "ascent",
                "observation": observation, "confidence": "low", "cue": cue,
            }],
            "depth_note": "",
            "review_status": "grounded",
            "review_source": "pose_signal",
        }
    if not labels or "unclear" in labels:
        return None
    if all(label == "no_clear_hips_first_signal" for label in labels):
        observation = "The pose check found no sustained hips-first timing gap through the early ascent."
        cue = "No hips-first correction is indicated by this clip; keep your usual controlled ascent."
        reasoning = "I don't see a clear hips-first rise in the tracked ascent, so this clip doesn't justify changing that part of your technique."
    elif all(label == "possible_hips_first" for label in labels):
        observation = "The pose check flagged a possible pelvis-first rise during the early ascent."
        cue = "On your next set, think 'chest and hips rise together' through the first half of the ascent."
        reasoning = "The tracked ascent suggests your pelvis may be getting ahead of your chest. Treat that as a point to review in the video, not a definitive fault call."
    else:
        return None
    return {
        "headline": "Ascent timing check",
        "primary_cue": cue,
        "reasoning": reasoning,
        "next_set": cue,
        "drill": "No drill recommended: this clip did not show a clear, repeatable ascent timing issue.",
        "limitation": "This is a conservative 2-D pose estimate. It cannot confirm spinal rounding or replace a full-speed review by a qualified coach.",
        "observations": [{
            "rep": None, "timestamp_s": None, "area": "ascent",
            "observation": observation, "confidence": "medium", "cue": cue,
        }],
        "depth_note": "",
        "review_status": "grounded",
        "review_source": "pose_signal",
    }


def _coach_video_prompt(facts: Dict, keyframe_map=None) -> str:
    compact = _coach_prompt_facts(facts)
    return (
        "You are LiftLens, an AI squat coach that communicates with the calm judgement and practical clarity "
        "of a seasoned coach who has spent over ten years coaching strength athletes. Never claim to be human "
        "or claim personal credentials. Speak directly to the lifter, in natural gym language.\n\n"
        "Review the attached full squat set as a time-ordered video. The sequence is sparsely sampled, so "
        "describe only what the frames support and do not pretend to see every instant. Review the squat as a "
        "whole: set-up, descent control, bottom position, the transition, and the ascent across the set. Compare "
        "Ignore captions, on-screen text, arrows, and edits as evidence; judge the lifter's movement itself. "
        "when the chest/torso and pelvis start rising: if the pelvis moves first while the chest stays tipped "
        "forward, describe that exact sequence; do not call it a fault unless the change is clear and repeatable. "
        "Use the coarse pose movement signals below as a cross-check. If a rep is marked possible_hips_first, "
        "you may describe only that numbered rep as a possible pelvis-first rise; do not generalize it to the whole set "
        "or call it a confirmed fault. If another rep is marked no_clear_hips_first_signal, say the pattern was not "
        "clear on that rep; never claim that rep's pelvis rose ahead of its chest. If marked unclear, do not make a firm claim. "
        "Do not label the back or spine as rounded, neutral, or safe: this camera/model cannot reliably verify "
        "spinal shape. A forward-leaning torso is not proof of rounding. Do not call ordinary hip travel during "
        "the descent or ascent a fault; hips moving forward by itself is not a useful diagnosis. Notice control, balance, foot "
        "stability, and rep-to-rep consistency when visible. Do not give knee-tracking cues in this review; "
        "the current camera-view check cannot reliably determine when that assessment is valid.\n\n"
        "Depth is reported separately by LiftLens. Do not judge depth, explain depth, mention any hip/knee "
        "comparison, joint estimate, proxy, margin, or which side passed. Those measurements are not evidence "
        "of a technique fault. Never connect a depth estimate to knee tracking. Do not give an official referee "
        "call, diagnose injury, infer pain, or tell the lifter to force a position.\n\n"
        "Be holistic but selective: summarize the set in one sentence, then give a distinct observation for each "
        "phase only when the video supports one. Never copy the same observation into OVERALL and a phase or "
        "repeat it within a field. Each phase statement must describe a visible event, not a generic quality judgment. "
        "Use the provided phase labels and timestamps; never assign an event to a phase that the timeline contradicts. "
        "Ground every statement in a visible event or frame. Do not invent stance width, spinal neutrality, "
        "bracing, weight distribution, or bar path when the view does not show it. Avoid empty praise such as "
        "'proper form' or 'stable base' unless you identify what is visibly stable. Do not fill a phase with "
        "generic 'smooth and controlled' language. If a phase or fault is not clear, say so instead of guessing. "
        "Give one highest-priority next-set cue only when it directly addresses a clear visible observation. "
        "If a specific, repeatable technique fault is clearly visible, recommend exactly one simple practice drill "
        "that directly trains that correction. Keep it practical and low risk: name the drill and give a brief way "
        "to perform it for 2-3 sets of 3-5 controlled reps using bodyweight or an easy familiar load. Do not prescribe "
        "heavy loads, max attempts, pain-provoking movements, or medical treatment. A movement observation alone is "
        "not automatically a fault; if no actionable fault is clearly supported, say 'No drill recommended: no clear, "
        "repeatable form fault was visible.' Do not suggest a drill for spinal rounding, knee tracking, or depth, "
        "because this review cannot reliably verify those. The drill must not repeat the next-set cue verbatim. "
        "Use plain, unambiguous instructions. Never say 'push your back into the bar' or give a cue about back "
        "rounding. If no clear fault is visible, say no technique change is justified from this review; do not invent a cue. "
        "Avoid generic cues and do not repeat the same advice.\n\n"
        "APP REP TIMING AND COARSE WHOLE-BODY ASCENT CHECK:\n" +
        json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        + ("\nSUPPLEMENTAL RAW FRAME MAP (frames are selected from the same clip and supplement the video sequence):\n"
           + json.dumps(keyframe_map, ensure_ascii=False, separators=(",", ":")) if keyframe_map else "")
    )


def generate_video_coaching(facts: Dict, video_path, frames=None, timeout_s: int = 600) -> Dict:
    """Run a local Qwen3-VL video review; the video is sampled by llama.cpp, never uploaded."""
    project_root, executable, ffmpeg_dir, model, mmproj = _video_runtime_paths()
    video_path = Path(video_path).resolve()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise ValueError("The local video file is missing or empty. Record or upload the squat clip again.")

    facts = dict(facts or {})
    movement_signals = _coach_prompt_facts(facts)["movement_signals"]
    embedded_frames = facts.pop("coach_frames", None)
    frames = list(frames if frames is not None else embedded_frames or [])
    frames = [item for item in frames if isinstance(item, dict) and isinstance(item.get("image"), bytes)]
    # Add raw bottom/ascent-phase frames across representative reps. The video
    # remains the primary evidence and these frames fill gaps in sparse sampling.
    keyframes = []
    if frames:
        by_rep = {}
        for item in sorted(frames, key=lambda x: x.get("timestamp_s", 0.0)):
            if item.get("rep") is not None:
                by_rep.setdefault(item["rep"], []).append(item)
        if by_rep:
            rep_ids = list(by_rep)
            max_reps = max(1, MAX_VIDEO_COACH_IMAGES // 2)
            indexes = np.linspace(0, len(rep_ids) - 1, min(max_reps, len(rep_ids))).round().astype(int)
            selected_reps = [rep_ids[i] for i in sorted(set(indexes.tolist()))]
            for rep_id in selected_reps:
                rep_frames = by_rep[rep_id]
                bottom = next((item for item in rep_frames if item.get("phase") == "bottom"), rep_frames[0])
                ascent = next((item for item in rep_frames if item.get("phase") == "early_ascent"), None)
                if ascent is None and len(rep_frames) > 1:
                    ascent = rep_frames[-1]
                keyframes.append(bottom)
                if ascent is not None and ascent is not bottom:
                    keyframes.append(ascent)
        else:
            indexes = np.linspace(0, len(frames) - 1, min(MAX_VIDEO_COACH_IMAGES, len(frames))).round().astype(int)
            keyframes = [frames[i] for i in sorted(set(indexes.tolist()))]
    keyframes = sorted(keyframes[:MAX_VIDEO_COACH_IMAGES], key=lambda x: x.get("timestamp_s", 0.0))
    keyframe_map = [
        {"image_number": index + 1, "rep": item.get("rep"), "phase": item.get("phase", "sample"),
         "time_seconds": item.get("timestamp_s")}
        for index, item in enumerate(keyframes)
    ]
    prompt = _coach_video_prompt(facts, keyframe_map)
    prompt += (
        "\n\nOutput exactly eight plain-text lines. Start each with the label shown below. "
        "Replace the labels with actual observations; never repeat instructions or describe this format. "
        "Use one short sentence per phase; if it is not assessable, say 'Not clear from this view.'\n"
        "OVERALL: holistic read of the set and its consistency\n"
        "SETUP: visible set-up observation\n"
        "DESCENT: visible descent observation\n"
        "BOTTOM: visible bottom/transition observation\n"
        "ASCENT: visible ascent observation\n"
        "NEXT SET: one concrete action based on the most important clear observation\n"
        "DRILL: one targeted low-risk practice drill if a clear, repeatable fault is visible; otherwise explain that no drill is recommended\n"
        "LIMIT: only a specific limitation that affected this review"
    )
    environment = os.environ.copy()
    environment["PATH"] = str(ffmpeg_dir) + os.pathsep + str(executable.parent) + os.pathsep + environment.get("PATH", "")

    # Normalize to a small, fast-start MP4. llama.cpp's video probe reads a local
    # file through ffprobe; this also handles AVI/MOV and large phone recordings.
    with tempfile.TemporaryDirectory(prefix="liftlens-video-coach-") as temp_dir:
        normalized_video = Path(temp_dir) / "review.mp4"
        normalize = [
            str(ffmpeg_dir / "ffmpeg.exe"), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path), "-map", "0:v:0", "-an",
            "-vf", "fps=15,scale=w='min(640,iw)':h=-2",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(normalized_video),
        ]
        try:
            normalized = subprocess.run(
                normalize, cwd=project_root, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="replace", timeout=180, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Preparing the local video took too long. Try a shorter clip.") from exc
        if normalized.returncode != 0 or not normalized_video.is_file():
            detail = (normalized.stderr or "").strip()[-800:]
            raise RuntimeError("Could not prepare this video for local review: " + (detail or "FFmpeg failed."))

        # Long clips can exceed the context budget after video and phase frames
        # are encoded. Retry once with half as many video frames and smaller
        # decode batches while keeping the phase evidence.
        attempts = [
            {"fps": 0.5, "context": 12288, "batch": 32, "ubatch": 16, "stills": keyframes},
            {"fps": 0.25, "context": 12288, "batch": 24, "ubatch": 8, "stills": keyframes},
        ]
        process = None
        reduced_sampling = False
        last_error = ""
        for attempt_index, attempt in enumerate(attempts):
            attempt_prompt = prompt
            if attempt_index:
                attempt_map = [
                    {"image_number": index + 1, "rep": item.get("rep"), "time_seconds": item.get("timestamp_s")}
                    for index, item in enumerate(attempt["stills"])
                ]
                attempt_prompt = _coach_video_prompt(facts, attempt_map)
                attempt_prompt += (
                    "\n\nThe video is sampled at a reduced rate to fit local memory. Review the sequence that is present, "
                    "and do not infer brief movements between sampled frames. Follow the seven-line output format "
                    "in the preceding instructions; write observations rather than repeating those instructions."
                )
            command = [
                str(executable), "-m", str(model), "--mmproj", str(mmproj),
                "--video", str(normalized_video), "--video-fps", str(attempt["fps"]),
                "--video-timestamp-interval", str(round(1000 / attempt["fps"])),
                "--video-ffmpeg-dir", str(ffmpeg_dir),
                "--image-min-tokens", "1024", "--image-max-tokens", "1024",
                "-ngl", "99", "-c", str(attempt["context"]), "-b", str(attempt["batch"]),
                "-ub", str(attempt["ubatch"]), "-t", "4",
                "-n", "384", "--temp", "0.1", "--no-warmup",
                "-p", attempt_prompt,
            ]
            image_paths = []
            for index, item in enumerate(attempt["stills"]):
                image_path = Path(temp_dir) / f"bottom-{attempt_index + 1}-{index + 1}.jpg"
                image_path.write_bytes(_prepare_coach_image(item["image"], max_side=512))
                image_paths.append(str(image_path))
            if image_paths:
                command.extend(["--image", ",".join(image_paths)])
            try:
                process = subprocess.run(
                    command, cwd=project_root, env=environment, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    encoding="utf-8", errors="replace", timeout=timeout_s, check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    "The local video coach exceeded its time limit. Try a shorter clip or close other GPU-heavy apps."
                ) from exc
            if process.returncode == 0:
                reduced_sampling = bool(attempt_index)
                break
            last_error = (process.stdout or "").strip()
            lower_error = last_error.lower()
            capacity_failure = any(marker in lower_error for marker in (
                "failed to find a memory slot", "unable to eval text chunk", "failed to decode text",
                "exceeds the available context", "context size", "out of memory", "cuda error",
                "cuda_error_out_of_memory",
            ))
            if attempt_index == 0 and capacity_failure:
                continue
            detail = last_error[-500:]
            if capacity_failure:
                raise RuntimeError(
                    "This clip exceeded the local model's context or GPU memory budget, even after reducing video sampling. "
                    "Try a shorter clip, or close other GPU-heavy apps and retry."
                )
            raise RuntimeError("The local video model could not complete the review: " + (detail or f"exit code {process.returncode}"))

        if process is None or process.returncode != 0:
            raise RuntimeError("The local video model could not complete the review: " + (last_error[-500:] or "unknown error"))

    import re

    output = process.stdout or ""
    # The CLI prints model-load and vision-encoding diagnostics before its answer.
    # Take text after the last image/video encoding event and remove trailing logs.
    segments = re.split(r"(?m)^.*I mtmd batch encoding done.*\r?\n", output)
    answer = segments[-1].strip()
    answer_lines = [
        line.strip() for line in answer.splitlines()
        if line.strip() and not re.match(r"^\d+\.\d+\.\d+\.\d+\s+[IWE]\s", line)
    ]
    answer = "\n".join(answer_lines).strip()
    if not answer:
        raise ValueError("The local video model returned no coaching text. Retry after confirming the local GPU has free memory.")

    labels = {"OVERALL", "SETUP", "DESCENT", "BOTTOM", "ASCENT", "NEXT SET", "DRILL", "LIMIT"}
    values = {}
    active_label = None
    for line in answer.splitlines():
        line = re.sub(r"^\s*[`*#-]+\s*|\s*[`*]+\s*$", "", line.strip())
        match = re.match(r"^(OVERALL|SETUP|DESCENT|BOTTOM|ASCENT|NEXT SET|DRILL|LIMIT|COACH)\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            active_label = "OVERALL" if match.group(1).upper() == "COACH" else match.group(1).upper()
            values.setdefault(active_label, [])
            if match.group(2).strip():
                values[active_label].append(match.group(2).strip())
        elif active_label and line:
            values[active_label].append(line)

    prompt_echo = re.compile(
        r"(?i)^\s*(?:your evidence-based read of the visible set in one or two natural sentences\.?\s*|"
        r"holistic read of the set and its consistency\.?\s*|visible (?:set-up|descent|bottom/transition|ascent) observation\.?\s*|"
        r"one concrete action based on the most important clear observation\.?\s*|"
        r"only a specific limitation that affected this review\.?\s*)+"
    )
    unsafe_coach_content = re.compile(
        r"(?i)\bproxy\b|\bdepth\b|\b(?:left|right)\s+(?:side|hip|knee|leg|foot|ankle|shoulder)\b|"
        r"\bhips?\b.{0,55}\b(?:knees?|lower|below|above|measurement|margin|similar angle)\b|"
        r"\bknees?\b.{0,55}\bhips?\b|\b(?:hips?|knees?)\s+joint\b|"
        r"\bknees?\b.{0,35}\b(?:track(?:s|ing)?|align(?:s|ment)?)\b|"
        r"\b(?:injur(?:y|ies)|pain|risk of injury)\b|\bcould lead to\b|\bpotentially cause\b"
    )
    hips_first_claim = re.compile(
        r"(?i)(?:\b(?:hips?|pelvis)\b.{0,65}\b(?:before|ahead of|faster than|first|while)\b.{0,40}\b(?:chest|torso)\b|"
        r"\b(?:hips?|pelvis)\b.{0,50}\b(?:rises?|shoots?|starts? moving)\b.{0,35}\b(?:as|while)\b.{0,35}\b(?:chest|torso)\b.{0,25}\b(?:stay|remain)s? down\b|"
        r"\b(?:chest|torso)\b.{0,40}\b(?:stays? down|lags?|does not rise)\b.{0,35}\b(?:hips?|pelvis)\b)"
    )
    spinal_shape_claim = re.compile(
        r"(?i)\b(?:back|spine)\b.{0,45}\b(?:round(?:ed|ing)?|curv(?:ed|ing)|flex(?:ed|ion|ing)?|straight|neutral|arch(?:ed|ing)?)\b|"
        r"\b(?:round(?:ed|ing)?|curv(?:ed|ing)|flex(?:ed|ion|ing)?|straight|neutral|arch(?:ed|ing)?)\b.{0,45}\b(?:back|spine)\b"
    )
    vague_hip_forward_claim = re.compile(
        r"(?i)\b(?:hips?|pelvis)\b.{0,55}\b(?:move|moves|moving|moved|shift|shifts|shifting|"
        r"drift|drifts|drifting|travel|travels|traveling)\s+(?:slightly\s+)?forward\b"
    )
    ambiguous_back_bar_cue = re.compile(
        r"(?i)\b(?:push|press|drive)\s+(?:your\s+)?(?:back|spine)\s+"
        r"(?:back\s+)?(?:into|against)\s+(?:the\s+)?bar(?:bell)?\b"
    )
    possible_hips_first_reps = {
        int(item["rep"]) for item in movement_signals
        if item.get("ascent_coordination") == "possible_hips_first"
    }

    def supported_hips_first_claim(sentence):
        """Allow only a qualified claim tied to a rep flagged possible by pose analysis."""
        if not hips_first_claim.search(sentence):
            return True
        rep_match = re.search(r"\b(?:rep|repetition)\s*#?\s*(\d+)\b", sentence, re.IGNORECASE)
        is_qualified = re.search(r"\b(?:may|might|possible|possibly|appears? to|seems? to)\b", sentence, re.IGNORECASE)
        return bool(
            rep_match and int(rep_match.group(1)) in possible_hips_first_reps and is_qualified
        )
    generic_coach_content = re.compile(
        r"(?i)\b(?:proper|good|sound) form\b|\bform appears? to be (?:sound|good|proper)\b|"
        r"\b(?:stable posture|stable base|stable and balanced|stable position)\b|"
        r"\b(?:smooth|steady|controlled) (?:and )?(?:controlled )?(?:descent|movement|manner|transition|position)\b|"
        r"\b(?:slowly and steadily|smoothly and steadily|in a controlled manner)\b|"
        r"\bclear and consistent transition\b|\brelatively wide stance\b|"
        r"\b(?:back|spine) (?:is|remains|stays|maintains?) (?:straight|neutral)\b|"
        r"\bspine (?:is|remains|stays|maintains?) (?:in )?a neutral position\b|"
        r"\bkeep(?:ing)?\b.{0,25}\b(?:back|spine)\b.{0,25}\b(?:straight|neutral)\b|"
        r"\bknees?\b.{0,20}\bslightly bent\b|\bproper angle\b|\bcoordinated manner\b|"
        r"\bneutral\b|"
        r"\bensure proper form\b|\bmaintain proper form\b|\bkeep each rep smooth\b|"
        r"\bbarbell\b.{0,45}\b(?:same level as|level with|proper angle|slight angle|held at)\b|"
        r"\b(?:focus on )?ensur(?:e|ing) (?:that )?(?:the )?feet are placed in a stable position\b"
    )
    body_term = r"\b(?:hips?|pelvis|chest|torso|trunk|back|spine|feet|foot|heels?)\b"
    movement_term = (
        r"\b(?:rise|rises|rising|shoot(?:s|ing)? up|drop(?:s|ped|ping)?|lower(?:s|ed|ing)?|"
        r"stay(?:s|ed|ing)?|remain(?:s|ed|ing)?|tip(?:s|ped|ping)?|round(?:s|ed|ing)?|"
        r"shift(?:s|ed|ing)?|mov(?:e|es|ed|ing)|lean(?:s|ed|ing)?|lift(?:s|ed|ing)?|"
        r"rock(?:s|ed|ing)?|wobbl(?:e|es|ed|ing)|collaps(?:e|es|ed|ing)|fold(?:s|ed|ing)?|"
        r"plant(?:s|ed|ing)?|pivot(?:s|ed|ing)?|rotat(?:e|es|ed|ing)|turn(?:s|ed|ing)|"
        r"shoot(?:s|ing)?|drive|push|keep|hold|starts? to|begins? to)\b"
    )
    body_movement = re.compile(
        r"(?i)(?:" + body_term + r".{0,90}" + movement_term + r"|"
        + movement_term + r".{0,55}" + body_term + r")"
    )
    rejected_claims = {"spinal_shape": False, "hip_forward": False, "back_bar_cue": False}

    def clean_field(lines):
        raw = prompt_echo.sub("", " ".join(lines)).strip()
        sentences = re.split(r"(?<=[.!?])\s+|\n+", raw)
        cleaned = []
        seen = set()
        for sentence in sentences:
            sentence = sentence.strip()
            key = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
            if not sentence or key in seen:
                continue
            is_spinal_shape_claim = bool(spinal_shape_claim.search(sentence))
            is_vague_hip_forward_claim = bool(vague_hip_forward_claim.search(sentence))
            if is_spinal_shape_claim:
                rejected_claims["spinal_shape"] = True
            if is_vague_hip_forward_claim:
                rejected_claims["hip_forward"] = True
            if is_spinal_shape_claim or is_vague_hip_forward_claim:
                continue
            if (unsafe_coach_content.search(sentence)
                    or not supported_hips_first_claim(sentence)
                    or generic_coach_content.search(sentence)):
                continue
            seen.add(key)
            cleaned.append(sentence)
        return " ".join(cleaned).strip()

    overall = clean_field(values.get("OVERALL", []))
    next_set = clean_field(values.get("NEXT SET", []))
    drill = clean_field(values.get("DRILL", []))
    if ambiguous_back_bar_cue.search(next_set):
        rejected_claims["back_bar_cue"] = True
        next_set = ""
    phase_values = {
        phase.lower(): clean_field(values.get(phase, []))
        for phase in ("SETUP", "DESCENT", "BOTTOM", "ASCENT")
    }
    phase_values = {
        phase: " ".join(
            sentence for sentence in re.split(r"(?<=[.!?])\s+", text)
            if body_movement.search(sentence)
        )
        for phase, text in phase_values.items()
    }
    overall_sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", overall)
                         if body_movement.search(sentence)]
    overall = " ".join(overall_sentences)
    # A set summary may synthesize phase observations, but the UI should not
    # print the same sentence again under a phase heading.
    seen_observations = {
        re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", overall) if sentence.strip()
    }
    for phase, text in phase_values.items():
        unique = []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            key = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
            if sentence.strip() and key not in seen_observations:
                seen_observations.add(key)
                unique.append(sentence.strip())
        phase_values[phase] = " ".join(unique)
    if not overall:
        overview_parts = [text for text in phase_values.values() if text]
        overall = " ".join(overview_parts)
    has_specific_observation = bool(overall) or any(phase_values.values())
    no_change_parts = [
        "I couldn't verify a specific form issue from the visible movement, so I won't invent a correction."
    ]
    if rejected_claims["spinal_shape"]:
        no_change_parts.append(
            "The model's back-rounding comment wasn't verified; LiftLens can't reliably judge spinal shape from this view."
        )
    if rejected_claims["hip_forward"]:
        no_change_parts.append("Hips moving forward by itself isn't enough to identify a form fault.")
    if rejected_claims["back_bar_cue"]:
        no_change_parts.append("I removed the unclear 'push your back into the bar' instruction.")
    no_change_parts.append("Don't change your technique based on this review.")
    no_change_message = " ".join(no_change_parts)
    if not overall or not has_specific_observation:
        withheld = True
        overall = no_change_message
        next_set = "No technique change is justified from this review."
        drill = "No drill recommended: no clear, repeatable form fault was verified."
        phase_values = {phase: "" for phase in phase_values}
        next_set = ""
    elif not next_set:
        # A verified movement description can be useful without forcing the model
        # to invent a correction when no clear fault was observed.
        withheld = False
        next_set = "No technique change is justified from the visible evidence in this review."
        drill = "No drill recommended: no clear, repeatable form fault was verified."
    else:
        withheld = False
    if next_set and not body_movement.search(next_set):
        next_set = "No technique change is justified from the visible evidence in this review."
        drill = "No drill recommended: no clear, repeatable form fault was verified."
    if not drill:
        drill = "No drill recommended: no clear, repeatable form fault was verified."
    signal_fallback = _movement_signal_coach(movement_signals)
    if possible_hips_first_reps and signal_fallback:
        # A cautious pose signal must not disappear behind a generic visual
        # summary such as “no change needed.” Preserve other grounded notes,
        # but surface the possible ascent pattern and its low-risk practice cue.
        model_ascent_claims = [
            sentence
            for text in [overall, *phase_values.values()]
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if hips_first_claim.search(sentence) and supported_hips_first_claim(sentence)
        ]
        if not model_ascent_claims:
            phase_values["ascent"] = signal_fallback["observations"][0]["observation"]
        if not overall or not has_specific_observation:
            overall = signal_fallback["reasoning"]
        if not next_set or re.search(
            r"(?i)\bno (?:technique change|hips-first correction|change)\b|\bnot justified\b",
            next_set,
        ):
            next_set = signal_fallback["primary_cue"]
        if not drill or re.search(r"(?i)\bno drill recommended\b|\bno clear.*fault\b", drill):
            drill = signal_fallback["drill"]
        withheld = False
    elif withheld and signal_fallback:
        return signal_fallback
    limitation = (
        "This is a 2-D pose estimate with sparse video samples. It cannot confirm spinal rounding "
        "or replace a qualified coach's full-speed review."
    )
    if reduced_sampling:
        limitation = "I used fewer video samples to fit local memory, so brief changes between them may be missed. " + limitation
    clean = lambda value: value.replace("\ufffd", "'").strip()[:700]
    observations = [
        {"rep": None, "timestamp_s": None, "area": phase, "observation": clean(text),
         "confidence": "medium", "cue": ""}
        for phase, text in phase_values.items() if text
    ]
    return {
        "headline": "No specific cue verified" if withheld else "Whole-set review",
        "primary_cue": clean(next_set)[:500],
        "reasoning": clean(overall),
        "next_set": clean(next_set)[:500],
        "drill": clean(drill)[:500],
        "limitation": clean(limitation)[:500],
        "observations": observations,
        "depth_note": "",
        "review_status": "withheld" if withheld else "grounded",
    }


def _prepare_coach_image(image_bytes: bytes, max_side: int = MAX_COACH_IMAGE_SIDE) -> bytes:
    """Downscale sampled frames to keep local coaching memory and context use bounded."""
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        # Keep mocked/legacy image bytes usable; actual analyzed frames are valid JPEGs.
        return image_bytes
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side > max_side:
        scale = max_side / float(longest_side)
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70]
    )
    return encoded.tobytes() if ok else image_bytes


def generate_visual_coaching(facts: Dict, frames=None, model: str = VISION_MODEL) -> Dict:
    """Ask local Ollama vision model to coach from selected raw video frames and measurements."""
    facts = dict(facts or {})
    embedded_frames = facts.pop("coach_frames", None)
    frames = list(frames if frames is not None else embedded_frames or [])
    frames = [item for item in frames if isinstance(item, dict) and isinstance(item.get("image"), bytes)]
    if not frames:
        raise ValueError("No usable video frames are available for visual coaching. Analyze or record a clip first.")
    if len(frames) > MAX_COACH_IMAGES:
        indices = [round(i * (len(frames) - 1) / (MAX_COACH_IMAGES - 1))
                   for i in range(MAX_COACH_IMAGES)]
        frames = [frames[i] for i in indices]

    system = (
        "You are LiftLens, an original, experienced strength-coach-style assistant: direct, calm, precise, and encouraging. "
        "Never claim you are a human coach or claim personal years of professional experience. "
        "You receive structured pose estimates and a sequence of raw, unannotated images from one local squat clip. "
        "Images are ordered by image_number in the user message. Ground every observation in visible evidence; do not invent a fault. "
        "Speak to the lifter like a practical coach, not like an analysis report. Translate measurements into plain gym language. "
        "Do not mention proxies, margins, statistics, coordinates, confidence labels, algorithms, or model limitations in the coaching cue. "
        "Never tell the lifter to keep a hip below a knee, and do not turn a proxy measurement into a technique instruction. "
        "Give at most two useful corrections, with one clear action the lifter can try next set. Avoid repeating the same cue in multiple fields. "
        "Do not mention bar path, midfoot, load shifting, or equipment details unless they are clearly visible in the images. "
        "Depth: the pose estimates use hip-joint and knee-joint proxies, not the official visible hip crease and top-of-knee landmarks. "
        "You may describe what the images appear to show about those visible landmarks only when clearly visible; otherwise say review is needed. "
        "LiftLens evaluates left and right proxy margins independently. A sufficiently visible side whose hip proxy is clearly lower than its same-side knee proxy counts as reaching the proxy threshold, even if the other side disagrees; name the disagreement but do not cancel that side's proxy pass. "
        "Back position: describe only clearly visible changes in the back/trunk silhouette across frames; clothing, camera angle and occlusion can make this unassessable. "
        "A torso lean angle is not proof of spinal rounding or butt wink. Do not diagnose a medical condition, infer pain, or state injury risk. "
        "If a depth estimate is borderline, inconsistent or missing, still offer one useful form/setup cue supported by the images, and explicitly keep depth unresolved. "
        "If a body area cannot be judged from this camera view, use low confidence and say what additional angle or visibility is needed. "
        "Return no more than three high-value, evidence-backed observations in priority order; do not pad the list with guesses. "
        "Avoid a generic laundry list. Never tell the lifter to force range or train through pain. "
        "Return only a JSON object matching the provided schema. Keep each string short, concrete, and conversational. Use a rep/time reference only when it helps the lifter act."
    )
    def make_payload(selected_frames, max_side, selected_facts):
        frame_map = [
            {"image_number": i + 1, "rep": item.get("rep"), "timestamp_seconds": item.get("timestamp_s")}
            for i, item in enumerate(selected_frames)
        ]
        user = (
            "Review this squat set. Treat measurements as estimates; use images as evidence. "
            "Give one useful cue even if depth is unclear. Image numbers match image order.\n"
            "FACTS:\n" + json.dumps(selected_facts, ensure_ascii=False) +
            "\nFRAME MAP:\n" + json.dumps(frame_map, ensure_ascii=False)
        )
        images = [
            base64.b64encode(_prepare_coach_image(item["image"], max_side)).decode("ascii")
            for item in selected_frames
        ]
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": images},
            ],
            "stream": False,
            "format": VISUAL_COACH_SCHEMA,
            "think": False,
            "options": {"temperature": 0.15, "num_predict": 512, "num_ctx": 4096, "num_gpu": 0},
            "keep_alive": "5m",
        }

    try:
        response = requests.post(
            OLLAMA_URL, json=make_payload(frames, MAX_COACH_IMAGE_SIDE, facts), timeout=600
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_error = exc.response
        try:
            error_body = response_error.json() if response_error is not None else {}
            error_detail = error_body.get("error", {})
            overflow = (
                error_detail.get("type") == "exceed_context_size_error"
                or "exceeds the available context size" in str(error_detail.get("message", "")).lower()
            )
        except (ValueError, AttributeError):
            overflow = False
        if not overflow:
            raise

        # Retry only a context overflow: one small image and a compact rep summary.
        retry_facts = dict(facts)
        for key in ("per_rep_side_joint_proxy_measurements", "detected_rep_bottom_times_seconds"):
            if isinstance(retry_facts.get(key), list):
                retry_facts[key] = retry_facts[key][:3]
        retry_frame = [frames[len(frames) // 2]]
        response = requests.post(
            OLLAMA_URL, json=make_payload(retry_frame, 224, retry_facts), timeout=600
        )
        response.raise_for_status()
    response_body = response.json()
    message = response_body.get("message") or {}
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        stop_reason = response_body.get("done_reason") or "unknown"
        raise ValueError(
            f"{model} returned no coaching text (stop reason: {stop_reason}). "
            "Try again after confirming the instruct model is installed."
        )
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen3-VL returned malformed JSON despite the output schema. Press Generate again.") from exc
    required = ("headline", "primary_cue", "reasoning", "observations", "depth_note", "limitation", "next_set")
    if not isinstance(result, dict) or any(key not in result for key in required):
        raise ValueError("The local visual model returned an incomplete response. Try again.")
    if not all(isinstance(result.get(key), str) for key in required if key != "observations"):
        raise ValueError("The local visual model returned an unexpected response. Try again.")
    if not isinstance(result.get("observations"), list):
        raise ValueError("The local visual model returned invalid observations. Try again.")
    def clean_text(value: str, limit: int) -> str:
        return value.replace("\ufffd", "—").strip()[:limit]

    clean_observations = []
    for item in result["observations"][:3]:
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in ("area", "observation", "confidence", "cue")):
            continue
        clean_observations.append({
            "rep": item.get("rep") if isinstance(item.get("rep"), int) else None,
            "timestamp_s": item.get("timestamp_s") if isinstance(item.get("timestamp_s"), (int, float)) else None,
            "area": clean_text(item["area"], 32),
            "observation": clean_text(item["observation"], 300),
            "confidence": item["confidence"] if item["confidence"] in ("high", "medium", "low") else "low",
            "cue": clean_text(item["cue"], 250),
        })
    return {
        "headline": clean_text(result["headline"], 200),
        "primary_cue": clean_text(result["primary_cue"], 400),
        "reasoning": clean_text(result["reasoning"], 500),
        "observations": clean_observations,
        "depth_note": clean_text(result["depth_note"], 400),
        "limitation": clean_text(result["limitation"], 400),
        "next_set": clean_text(result["next_set"], 400),
    }


def generate_coaching(facts: Dict, model: str = DEFAULT_MODEL) -> Dict[str, str]:
    """Generate a brief cue from analyzer facts only; the LLM never sees video."""
    system = (
        "You are LiftLens, a cautious squat technique explainer. You receive structured "
        "measurements from a computer-vision system, not the video itself. Treat those "
        "measurements as estimates. Use only the supplied facts. Never claim to diagnose, "
        "predict injury, or make an official referee call. If the result is uncertain, say so. "
        "Always respond based on the supplied result, including when depth is borderline or no reps were detected. "
        "If no reps were detected, do not invent a form fault or technique correction; explain the tracking limitation "
        "and give one useful recording/setup step. If left/right proxy measurements disagree or are uncertain, state that "
        "clearly and recommend frame review rather than declaring a pass/fail. A positive margin only means the hip-joint "
        "proxy appears lower than the same-side knee-joint proxy; it is not an IPF depth ruling. "
        "Return an object matching the supplied JSON schema exactly. "
        "Give one practical, conservative cue tied to the supplied evidence. Do not merely repeat a number or verdict in the cue. "
        "If the evidence does not support a technique correction, say that more footage is needed. "
        "When evidence suggests a high or limited depth proxy, you may suggest reducing load, using a box or support, "
        "or practicing a comfortable pain-free range; never state the cause or force a deeper squat. "
        "A torso lean angle is not proof of a rounded back or butt wink. Never claim the vision system detects pain or injury. "
        "If the user reports pain, numbness, dizziness, or an acute injury, advise stopping the set and seeking qualified help. "
        "Keep each value to one short sentence. Do not add markdown."
    )
    user = ("Create a coach explanation from these facts only. Keep each field brief and do not reason aloud. "
            "Return only the requested JSON object.\n" + json.dumps(facts, ensure_ascii=False))
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": COACH_SCHEMA,
            "think": False,
            "options": {"temperature": 0.1, "num_predict": 256},
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen3 returned malformed JSON despite the output schema. Press Generate again.") from exc
    required = ("headline", "cue", "why", "limitation")
    if not isinstance(result, dict) or any(not isinstance(result.get(key), str) for key in required):
        raise ValueError("The local model returned an unexpected response. Try again.")
    return {key: result[key].strip()[:400] for key in required}
