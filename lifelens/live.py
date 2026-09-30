"""Local, opt-in webcam squat monitor using MediaPipe Pose Landmarker."""

from __future__ import annotations

import atexit
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .analyzer import (_best_side, _bilateral_pose_metrics, _classify_bilateral_depth,
                       _draw_pose, _joint_angle_degrees)


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
INFERENCE_INTERVAL_S = 0.10
MIN_VISIBILITY = 0.50
REVIEW_BAND_BODY_SCALE = 0.02


class LiveSquatMonitor:
    """Owns the webcam thread; camera access begins only after start() is called."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame = None
        self._state = self._empty_state()
        self._last_landmarks = None

    @staticmethod
    def _empty_state():
        return {
            "running": False,
            "status": "Ready. The camera stays off until you start a session.",
            "phase": "WAITING",
            "rep_count": 0,
            "cue": "Set the camera side-on with your full body in frame, then stand still to calibrate.",
            "confidence": 0.0,
            "depth": "Waiting for a detected squat bottom.",
            "torso_angle_deg": None,
            "body_scale": None,
            "reps": [],
            "error": None,
            "tracked_video": None,
            "recording_error": None,
            "recording_enabled": False,
            "recorded_duration_s": 0.0,
            "raw_video_path": None,
            "raw_video_error": None,
            "coach_frames": [],
        }

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._state["running"])

    def start(self, camera_index: int = 0) -> None:
        with self._lock:
            if self._state["running"]:
                return
            self._state = self._empty_state()
            self._state["running"] = True
            self._state["status"] = "Opening the selected camera…"
            self._frame = None
            self._last_landmarks = None
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, args=(camera_index,), name="LiftLensCamera", daemon=True
            )
            self._thread.start()

    def begin_browser_session(self, recording_enabled: bool = False) -> None:
        """Reset shared state when the browser starts sending webcam frames."""
        with self._lock:
            previous_raw_video = self._state.get("raw_video_path")
            self._state = self._empty_state()
            self._state["running"] = True
            self._state["status"] = "Browser camera connected. Processing frames locally."
            self._state["recording_enabled"] = bool(recording_enabled)
            self._frame = None
            self._last_landmarks = None
        if previous_raw_video:
            try:
                os.unlink(previous_raw_video)
            except OSError:
                pass

    def end_browser_session(self, tracked_video=None, recording_error=None,
                            recorded_duration_s: float = 0.0, coach_frames=None,
                            raw_video_path=None, raw_video_error=None) -> None:
        """Publish the completed local recording and mark the session stopped."""
        with self._lock:
            self._state["running"] = False
            self._state["tracked_video"] = tracked_video
            self._state["recording_error"] = recording_error
            self._state["recorded_duration_s"] = float(recorded_duration_s)
            self._state["raw_video_path"] = raw_video_path
            self._state["raw_video_error"] = raw_video_error
            self._state["coach_frames"] = list(coach_frames or [])
            if not self._state["error"]:
                if tracked_video:
                    self._state["status"] = "Live session finished. Your annotated video is ready below."
                elif self._state["recording_enabled"]:
                    self._state["status"] = "Live session stopped, but no tracked video was produced."
                else:
                    self._state["status"] = "Live session stopped. Recording was off."

    def cleanup_recorded_video(self) -> None:
        """Delete the temporary raw live clip after coaching or before a new session."""
        with self._lock:
            path = self._state.get("raw_video_path")
            self._state["raw_video_path"] = None
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        with self._lock:
            self._state["running"] = False
            if not self._state["error"]:
                self._state["status"] = "Live session stopped. No video was saved."

    def snapshot(self):
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            state = dict(self._state)
            state["reps"] = [dict(rep) for rep in self._state["reps"]]
            return frame, state

    def coaching_facts(self) -> dict:
        _, state = self.snapshot()
        reps = state["reps"]
        return {
            "exercise": "barbell squat",
            "rep_count_estimate": len(reps),
            "coach_frames": state.get("coach_frames", []),
            "rep_timeline_seconds": [
                {"rep": rep["rep"], "bottom_seconds": round(rep["bottom_time_s"], 2)}
                for rep in reps[-8:]
            ],
        }

    @staticmethod
    def _smooth_landmarks(previous, current, alpha=0.65):
        if previous is None or len(previous) != len(current):
            return [SimpleNamespace(x=float(lm.x), y=float(lm.y),
                                    visibility=float(lm.visibility))
                    for lm in current]
        smoothed = []
        for old, new in zip(previous, current):
            smoothed.append(SimpleNamespace(
                x=alpha * float(new.x) + (1 - alpha) * old.x,
                y=alpha * float(new.y) + (1 - alpha) * old.y,
                visibility=float(new.visibility),
            ))
        return smoothed

    @staticmethod
    def _torso_angle(landmarks, shoulder_index, hip_index) -> float:
        shoulder, hip = landmarks[shoulder_index], landmarks[hip_index]
        return math.degrees(math.atan2(abs(float(shoulder.x) - float(hip.x)),
                                       max(abs(float(shoulder.y) - float(hip.y)), 1e-5)))

    def _run(self, camera_index: int) -> None:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            self._set_error("Could not open the camera. Check that it is connected and not being used by another app.")
            return

        if not MODEL_PATH.is_file():
            cap.release()
            self._set_error(f"MediaPipe pose model was not found at {MODEL_PATH}.")
            return

        vision = mp.tasks.vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        start_clock = time.monotonic()
        next_inference = 0.0
        last_timestamp_ms = -1
        landmarks = None
        side_indices = None
        rep_state = {
            "phase": "WAITING",
            "baseline_y": None,
            "baseline_torso": None,
            "baseline_scale": None,
            "baseline_knee_angle": None,
            "bottom_y": None,
            "bottom_gap": None,
            "bottom_left_gap": None,
            "bottom_right_gap": None,
            "bottom_left_confidence": 0.0,
            "bottom_right_confidence": 0.0,
            "bottom_left_body_scale": None,
            "bottom_right_body_scale": None,
            "bottom_torso": None,
            "bottom_time": None,
            "bottom_visibility": None,
            "bottom_knee_angle": None,
            "cycle_start_time": None,
            "descent_frames": 0,
            "ascent_frames": 0,
            "standing_frames": 0,
            "previous_y": None,
            "rep_count": 0,
            "reps": [],
            "cue": self._empty_state()["cue"],
            "depth": "Waiting for a detected squat bottom.",
        }
        last_update = None
        processed = 0

        try:
            with vision.PoseLandmarker.create_from_options(options) as model:
                with self._lock:
                    self._state["status"] = "Camera connected. Stand still briefly to calibrate."
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        self._set_error("The camera stopped returning frames. Check its connection and permissions.")
                        break
                    height, width = frame.shape[:2]
                    if max(height, width) > 720:
                        scale = 720.0 / max(height, width)
                        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                    elapsed = time.monotonic() - start_clock
                    if elapsed >= next_inference:
                        timestamp_ms = max(last_timestamp_ms + 1, int(elapsed * 1000))
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        detected = model.detect_for_video(image, timestamp_ms)
                        last_timestamp_ms = timestamp_ms
                        next_inference = elapsed + INFERENCE_INTERVAL_S
                        processed += 1
                        if detected.pose_landmarks:
                            landmarks = self._smooth_landmarks(landmarks, detected.pose_landmarks[0])
                            if side_indices is None:
                                side_indices = _best_side(landmarks)
                            shoulder_i, hip_i, knee_i = side_indices
                            bilateral = _bilateral_pose_metrics(landmarks, width, height)
                            visibility = min(float(landmarks[i].visibility) for i in side_indices)
                            hip_y = float(landmarks[hip_i].y)
                            knee_y = float(landmarks[knee_i].y)
                            ankle_i = 27 if hip_i == 23 else 28
                            body_scale = abs(float(landmarks[ankle_i].y) - float(landmarks[shoulder_i].y))
                            knee_angle = (
                                _joint_angle_degrees(landmarks[hip_i], landmarks[knee_i], landmarks[ankle_i], width, height)
                                if min(float(landmarks[i].visibility) for i in (hip_i, knee_i, ankle_i)) >= 0.50
                                else None
                            )
                            gap = hip_y - knee_y
                            torso_angle = self._torso_angle(landmarks, shoulder_i, hip_i)
                            rep_state = self._update_rep_state(
                                rep_state, elapsed, hip_y, knee_y, gap, visibility, torso_angle,
                                body_scale, knee_angle, **bilateral
                            )
                            last_update = {
                                "confidence": visibility,
                                "torso_angle_deg": torso_angle,
                                "body_scale": body_scale,
                                "knee_angle_deg": knee_angle,
                                "depth": rep_state["depth"],
                                "cue": rep_state["cue"],
                                "phase": rep_state["phase"],
                                "rep_count": rep_state["rep_count"],
                                "reps": rep_state["reps"],
                            }
                        else:
                            landmarks = None
                            last_update = {
                                "confidence": 0.0,
                                "torso_angle_deg": None,
                                "body_scale": None,
                                "depth": "No usable pose in this frame.",
                                "cue": "Step back and keep your shoulder, hip, knee, and ankle visible.",
                                "phase": rep_state["phase"],
                                "rep_count": rep_state["rep_count"],
                                "reps": rep_state["reps"],
                            }

                    if landmarks is not None and side_indices is not None:
                        frame = _draw_pose(frame, landmarks, side_indices[1], side_indices[2])
                    else:
                        cv2.putText(frame, "POSE NOT DETECTED", (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    elapsed_label = f"{rep_state['phase']}  |  REPS {rep_state['rep_count']}  |  {elapsed:.1f}s"
                    cv2.putText(frame, elapsed_label, (12, 62), cv2.FONT_HERSHEY_SIMPLEX,
                                0.58, (255, 255, 255), 2, cv2.LINE_AA)
                    with self._lock:
                        self._frame = frame
                        if last_update:
                            self._state.update(last_update)
                        self._state["status"] = "Live — processed locally; no recording is being saved."
        except Exception as exc:
            self._set_error(f"Live pose processing failed: {exc}")
        finally:
            cap.release()
            with self._lock:
                self._state["running"] = False
                if not self._state["error"]:
                    self._state["status"] = "Live session stopped. No video was saved."

    @staticmethod
    def _update_rep_state(state, elapsed, hip_y, knee_y, gap, visibility, torso_angle,
                          body_scale=0.50, knee_angle_deg=None, left_hip_y=None,
                          left_knee_y=None, left_gap=None, left_confidence=0.0,
                          left_body_scale=None, left_knee_angle_deg=None,
                          right_hip_y=None, right_knee_y=None, right_gap=None,
                          right_confidence=0.0, right_body_scale=None,
                          right_knee_angle_deg=None):
        """Conservative live state machine: a rep needs descent, reversal and stable return."""
        if visibility < MIN_VISIBILITY:
            # A landmark dropout during movement invalidates the incomplete cycle;
            # preserving it could turn a jump in pose tracking into a rep.
            state.update({
                "phase": "WAITING", "baseline_y": None, "baseline_torso": None,
                "baseline_scale": None, "baseline_knee_angle": None,
                "bottom_y": None, "bottom_gap": None,
                "bottom_left_gap": None, "bottom_right_gap": None,
                "bottom_left_confidence": 0.0, "bottom_right_confidence": 0.0,
                "bottom_left_body_scale": None, "bottom_right_body_scale": None,
                "bottom_torso": None, "bottom_time": None,
                "bottom_visibility": None, "bottom_knee_angle": None, "cycle_start_time": None,
                "descent_frames": 0, "ascent_frames": 0,
                "standing_frames": 0, "previous_y": None,
            })
            state["cue"] = "Pose tracking paused. Re-center so the selected side stays visible."
            state["depth"] = "Not enough landmark visibility for a useful estimate."
            return state

        if not np.isfinite(body_scale) or body_scale < 0.10:
            body_scale = 0.50
        body_scale = float(body_scale)

        if state.get("baseline_y") is None:
            state.update({
                "phase": "WAITING", "baseline_y": hip_y,
                "baseline_torso": torso_angle, "baseline_scale": body_scale,
                "baseline_knee_angle": knee_angle_deg,
                "previous_y": hip_y, "descent_frames": 0,
                "ascent_frames": 0, "standing_frames": 0,
            })
            state["cue"] = "Calibrated. Perform one comfortable squat at your normal pace."
            return state

        baseline = float(state["baseline_y"])
        scale = float(state.get("baseline_scale") or body_scale)
        displacement = hip_y - baseline
        phase = state["phase"]
        start_threshold = max(0.05, scale * 0.11)
        required_excursion = max(0.075, scale * 0.16)
        return_threshold = max(0.022, scale * 0.045)
        direction_slack = max(0.006, scale * 0.012)
        previous_y = float(state.get("previous_y") if state.get("previous_y") is not None else hip_y)

        if phase == "WAITING":
            if abs(displacement) <= max(0.025, scale * 0.04):
                state["baseline_y"] = 0.98 * baseline + 0.02 * hip_y
                state["baseline_torso"] = 0.98 * state["baseline_torso"] + 0.02 * torso_angle
                state["baseline_scale"] = 0.98 * scale + 0.02 * body_scale
                if knee_angle_deg is not None:
                    old_angle = state.get("baseline_knee_angle")
                    state["baseline_knee_angle"] = (knee_angle_deg if old_angle is None
                                                     else 0.98 * old_angle + 0.02 * knee_angle_deg)
            if displacement >= start_threshold:
                state.update({
                    "phase": "DESCENDING", "bottom_y": hip_y,
                    "bottom_gap": gap, "bottom_torso": torso_angle,
                    "bottom_time": elapsed, "bottom_visibility": visibility,
                    "bottom_knee_angle": knee_angle_deg,
                    "bottom_left_gap": left_gap, "bottom_right_gap": right_gap,
                    "bottom_left_confidence": left_confidence,
                    "bottom_right_confidence": right_confidence,
                    "bottom_left_body_scale": left_body_scale,
                    "bottom_right_body_scale": right_body_scale,
                    "cycle_start_time": elapsed, "descent_frames": 1,
                    "ascent_frames": 0, "standing_frames": 0,
                })
                state["cue"] = "Rep started. LiftLens counts only after a full, visible return to standing."
        elif phase == "DESCENDING":
            if hip_y >= previous_y - direction_slack:
                state["descent_frames"] = state.get("descent_frames", 0) + 1
            if hip_y >= state["bottom_y"]:
                state.update({"bottom_y": hip_y, "bottom_gap": gap,
                              "bottom_torso": torso_angle, "bottom_time": elapsed,
                              "bottom_visibility": visibility, "bottom_knee_angle": knee_angle_deg,
                              "bottom_left_gap": left_gap, "bottom_right_gap": right_gap,
                              "bottom_left_confidence": left_confidence,
                              "bottom_right_confidence": right_confidence,
                              "bottom_left_body_scale": left_body_scale,
                              "bottom_right_body_scale": right_body_scale})
            if (state.get("descent_frames", 0) >= 3
                    and hip_y < state["bottom_y"] - max(0.02, scale * 0.04)):
                state["phase"] = "ASCENDING"
                state["ascent_frames"] = 1
        elif phase == "ASCENDING":
            if hip_y >= state["bottom_y"]:
                state.update({"bottom_y": hip_y, "bottom_gap": gap,
                              "bottom_torso": torso_angle, "bottom_time": elapsed,
                              "bottom_visibility": visibility, "bottom_knee_angle": knee_angle_deg,
                              "bottom_left_gap": left_gap, "bottom_right_gap": right_gap,
                              "bottom_left_confidence": left_confidence,
                              "bottom_right_confidence": right_confidence,
                              "bottom_left_body_scale": left_body_scale,
                              "bottom_right_body_scale": right_body_scale,
                              "phase": "DESCENDING",
                              "descent_frames": 1, "ascent_frames": 0})
            else:
                if hip_y <= previous_y + direction_slack:
                    state["ascent_frames"] = state.get("ascent_frames", 0) + 1
                if hip_y <= baseline + return_threshold:
                    state["standing_frames"] = state.get("standing_frames", 0) + 1
                else:
                    state["standing_frames"] = 0

                cycle_duration = elapsed - float(state.get("cycle_start_time") or elapsed)
                complete = (
                    state.get("standing_frames", 0) >= 3
                    and state.get("ascent_frames", 0) >= 3
                    and cycle_duration >= 0.80
                    and state["bottom_y"] - baseline >= required_excursion
                    and state.get("bottom_visibility", 0.0) >= MIN_VISIBILITY
                    and state.get("baseline_knee_angle") is not None
                    and state.get("bottom_knee_angle") is not None
                    and state["baseline_knee_angle"] >= 150.0
                    and state["bottom_knee_angle"] <= 155.0
                    and state["baseline_knee_angle"] - state["bottom_knee_angle"] >= 15.0
                )
                if complete:
                    bottom_gap = float(state["bottom_gap"])
                    normalized_gap = bottom_gap / scale
                    left_scale = state.get("bottom_left_body_scale")
                    right_scale = state.get("bottom_right_body_scale")
                    left_margin = (float(state["bottom_left_gap"]) / left_scale * 100
                                   if state.get("bottom_left_gap") is not None and left_scale else None)
                    right_margin = (float(state["bottom_right_gap"]) / right_scale * 100
                                    if state.get("bottom_right_gap") is not None and right_scale else None)
                    depth_status, passing_sides, side_disagreement = _classify_bilateral_depth(
                        left_margin, state.get("bottom_left_confidence", 0.0),
                        right_margin, state.get("bottom_right_confidence", 0.0),
                    )
                    if passing_sides:
                        cue = "LiftLens marked its depth estimate as reached for this rep. This is not an official referee decision."
                    elif depth_status == "HIP PROXY ABOVE KNEE PROXY ON BOTH SIDES":
                        cue = "LiftLens did not mark its depth estimate as reached; review the frame before changing technique."
                    else:
                        cue = "The depth estimate is unclear; review this rep before changing technique."
                    state["rep_count"] += 1
                    state["reps"].append({
                        "rep": state["rep_count"],
                        "bottom_time_s": float(state["bottom_time"]),
                        "hip_minus_knee_y": bottom_gap,
                        "hip_minus_knee_pct_body_scale": float(normalized_gap * 100),
                        "left_margin_pct_body": left_margin,
                        "left_confidence": float(state.get("bottom_left_confidence", 0.0)),
                        "right_margin_pct_body": right_margin,
                        "right_confidence": float(state.get("bottom_right_confidence", 0.0)),
                        "passing_sides": passing_sides,
                        "side_disagreement": side_disagreement,
                        "knee_flexion_estimate_deg": float(state["baseline_knee_angle"] - state["bottom_knee_angle"]),
                        "torso_lean_deg": float(state["bottom_torso"]),
                        "torso_change_deg": float(state["bottom_torso"] - state["baseline_torso"]),
                        "visibility": float(state.get("bottom_visibility", visibility)),
                        "depth_status": depth_status,
                        "cycle_duration_s": float(cycle_duration),
                    })
                    state.update({
                        "depth": depth_status, "cue": cue, "phase": "WAITING",
                        "baseline_y": hip_y, "baseline_torso": torso_angle,
                        "baseline_scale": body_scale, "bottom_y": None,
                        "baseline_knee_angle": knee_angle_deg,
                        "bottom_gap": None, "bottom_torso": None, "bottom_time": None,
                        "bottom_left_gap": None, "bottom_right_gap": None,
                        "bottom_left_confidence": 0.0, "bottom_right_confidence": 0.0,
                        "bottom_left_body_scale": None, "bottom_right_body_scale": None,
                        "bottom_visibility": None, "bottom_knee_angle": None, "cycle_start_time": None,
                        "descent_frames": 0, "ascent_frames": 0, "standing_frames": 0,
                    })
        state["previous_y"] = hip_y
        return state

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._state["error"] = message
            self._state["status"] = "Camera or pose model unavailable."
            self._state["running"] = False


_monitor = LiveSquatMonitor()
atexit.register(_monitor.stop)
atexit.register(_monitor.cleanup_recorded_video)


def get_live_monitor() -> LiveSquatMonitor:
    return _monitor
