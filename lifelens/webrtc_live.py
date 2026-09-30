"""Browser-captured webcam frames for LiftLens live squat analysis."""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace

import av
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from .analyzer import _best_side, _bilateral_pose_metrics, _draw_pose, _joint_angle_degrees
from .live import INFERENCE_INTERVAL_S, MODEL_PATH, LiveSquatMonitor, get_live_monitor


class LivePoseProcessor(VideoProcessorBase):
    """Analyze frames supplied by the browser; never opens a server-side camera."""

    RECORDING_FPS = 10.0

    def __init__(self, record_video: bool = False) -> None:
        self.monitor: LiveSquatMonitor = get_live_monitor()
        self.record_video = bool(record_video)
        self.monitor.begin_browser_session(recording_enabled=self.record_video)
        self.started_at = time.monotonic()
        self.next_inference = 0.0
        self.last_timestamp_ms = -1
        self.landmarks = None
        self.side_indices = None
        self.last_update = None
        self.rep_state = {
            "phase": "WAITING",
            "baseline_y": None,
            "baseline_torso": None,
            "baseline_scale": None,
            "baseline_knee_angle": None,
            "bottom_y": None,
            "bottom_gap": None,
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
            "cue": self.monitor._empty_state()["cue"],
            "depth": "Waiting for a detected squat bottom.",
        }
        self.model = None
        self.model_error = None
        self.video_writer = None
        self.video_path = None
        self.raw_video_writer = None
        self.raw_video_path = None
        self.raw_recording_error = None
        self.recording_error = None
        self.recorded_frames = 0
        self.last_recorded_at = -1.0
        self.current_bottom_frame = None
        self.last_completed_count = 0
        self.coach_frames = []
        self.reference_frames = []
        self.reference_frame_count = 0
        self.last_reference_capture = -1.0
        try:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"MediaPipe pose model was not found at {MODEL_PATH}.")
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.model = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:
            self.model_error = str(exc)
            self.monitor._set_error(f"Live pose model could not start: {exc}")

    @staticmethod
    def _torso_angle(landmarks, shoulder_index: int, hip_index: int) -> float:
        shoulder, hip = landmarks[shoulder_index], landmarks[hip_index]
        return math.degrees(
            math.atan2(
                abs(float(shoulder.x) - float(hip.x)),
                max(abs(float(shoulder.y) - float(hip.y)), 1e-5),
            )
        )

    @staticmethod
    def _smooth_landmarks(previous, current, alpha=0.65):
        if previous is None or len(previous) != len(current):
            return [
                SimpleNamespace(x=float(lm.x), y=float(lm.y), visibility=float(lm.visibility))
                for lm in current
            ]
        return [
            SimpleNamespace(
                x=alpha * float(new.x) + (1 - alpha) * old.x,
                y=alpha * float(new.y) + (1 - alpha) * old.y,
                visibility=float(new.visibility),
            )
            for old, new in zip(previous, current)
        ]

    @staticmethod
    def _jpeg_payload(image):
        h, w = image.shape[:2]
        if max(h, w) > 640:
            scale = 640.0 / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 76])
        return encoded.tobytes() if ok else None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        if max(image.shape[:2]) > 720:
            scale = 720.0 / max(image.shape[:2])
            image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        height, width = image.shape[:2]
        if width % 2 or height % 2:
            image = image[:height - (height % 2), :width - (width % 2)]
        source_image = image.copy()

        # Keep a few raw, unannotated frames so the local visual coach can still
        # review the movement if the conservative rep counter abstains.
        elapsed = time.monotonic() - self.started_at
        if elapsed - self.last_reference_capture >= 2.0:
            payload = self._jpeg_payload(source_image)
            if payload:
                candidate = {"rep": None, "timestamp_s": round(elapsed, 2), "image": payload}
                self.reference_frame_count += 1
                if len(self.reference_frames) < 6:
                    self.reference_frames.append(candidate)
                else:
                    # Reservoir sampling keeps a representative, bounded set
                    # from the entire live session without saving its video.
                    slot = random.randrange(self.reference_frame_count)
                    if slot < 6:
                        self.reference_frames[slot] = candidate
                self.last_reference_capture = elapsed

        if self.model_error is None and self.model is not None:
            if elapsed >= self.next_inference:
                try:
                    timestamp_ms = max(self.last_timestamp_ms + 1, int(elapsed * 1000))
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = self.model.detect_for_video(mp_image, timestamp_ms)
                    self.last_timestamp_ms = timestamp_ms
                    self.next_inference = elapsed + INFERENCE_INTERVAL_S
                    if result.pose_landmarks:
                        self.landmarks = self._smooth_landmarks(self.landmarks, result.pose_landmarks[0])
                        if self.side_indices is None:
                            self.side_indices = _best_side(self.landmarks)
                        shoulder_i, hip_i, knee_i = self.side_indices
                        bilateral = _bilateral_pose_metrics(self.landmarks, width, height)
                        visibility = min(float(self.landmarks[i].visibility) for i in self.side_indices)
                        hip_y = float(self.landmarks[hip_i].y)
                        knee_y = float(self.landmarks[knee_i].y)
                        ankle_i = 27 if hip_i == 23 else 28
                        body_scale = abs(float(self.landmarks[ankle_i].y) - float(self.landmarks[shoulder_i].y))
                        knee_angle = (
                            _joint_angle_degrees(self.landmarks[hip_i], self.landmarks[knee_i],
                                                 self.landmarks[ankle_i], width, height)
                            if min(float(self.landmarks[i].visibility) for i in (hip_i, knee_i, ankle_i)) >= 0.50
                            else None
                        )
                        torso_angle = self._torso_angle(self.landmarks, shoulder_i, hip_i)
                        before_count = self.rep_state["rep_count"]
                        self.rep_state = self.monitor._update_rep_state(
                            self.rep_state,
                            elapsed,
                            hip_y,
                            knee_y,
                            hip_y - knee_y,
                            visibility,
                            torso_angle,
                            body_scale,
                            knee_angle,
                            **bilateral,
                        )
                        bottom_time = self.rep_state.get("bottom_time")
                        if (bottom_time is not None and abs(bottom_time - elapsed) < 1e-6
                                and self.rep_state["phase"] in ("DESCENDING", "ASCENDING")):
                            payload = self._jpeg_payload(source_image)
                            if payload:
                                self.current_bottom_frame = {
                                    "rep": self.rep_state["rep_count"] + 1,
                                    "timestamp_s": round(float(bottom_time), 2),
                                    "image": payload,
                                }
                        if self.rep_state["rep_count"] > before_count:
                            if self.current_bottom_frame:
                                self.coach_frames.append(self.current_bottom_frame)
                                self.coach_frames = self.coach_frames[-12:]
                            self.current_bottom_frame = None
                            self.last_completed_count = self.rep_state["rep_count"]
                        self.last_update = {
                            "confidence": visibility,
                            "torso_angle_deg": torso_angle,
                            "body_scale": body_scale,
                            "knee_angle_deg": knee_angle,
                            "depth": self.rep_state["depth"],
                            "cue": self.rep_state["cue"],
                            "phase": self.rep_state["phase"],
                            "rep_count": self.rep_state["rep_count"],
                            "reps": self.rep_state["reps"],
                        }
                    else:
                        self.landmarks = None
                        self.last_update = {
                            "confidence": 0.0,
                            "torso_angle_deg": None,
                            "body_scale": None,
                            "knee_angle_deg": None,
                            "depth": "No usable pose in this frame.",
                            "cue": "Step back and keep your shoulder, hip, knee, and ankle visible.",
                            "phase": self.rep_state["phase"],
                            "rep_count": self.rep_state["rep_count"],
                            "reps": self.rep_state["reps"],
                        }
                except Exception as exc:
                    self.model_error = str(exc)
                    self.monitor._set_error(f"Live pose processing failed: {exc}")

            if self.landmarks is not None and self.side_indices is not None:
                image = _draw_pose(image, self.landmarks, self.side_indices[1], self.side_indices[2])
            else:
                cv2.putText(
                    image,
                    "POSE NOT DETECTED",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                image,
                f"{self.rep_state['phase']}  |  REPS {self.rep_state['rep_count']}",
                (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                image,
                "POSE MODEL UNAVAILABLE",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            with self.monitor._lock:
                self.monitor._frame = image.copy()
                if self.last_update:
                    self.monitor._state.update(self.last_update)
                self.monitor._state["running"] = True
                if self.record_video:
                    self.monitor._state["status"] = "Live — processing and recording locally. Stop to create the tracked video."
                else:
                    self.monitor._state["status"] = "Live — browser camera frames processed locally; recording is off."

        if self.record_video and self.recording_error is None:
            elapsed = time.monotonic() - self.started_at
            if self.last_recorded_at < 0 or elapsed - self.last_recorded_at >= 1.0 / self.RECORDING_FPS:
                if self.video_writer is None:
                    temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                    self.video_path = temp.name
                    temp.close()
                    self.video_writer = cv2.VideoWriter(
                        self.video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.RECORDING_FPS,
                        (image.shape[1], image.shape[0]),
                    )
                    if not self.video_writer.isOpened():
                        self.video_writer.release()
                        self.video_writer = None
                        self.recording_error = "Could not initialize the local MP4 encoder."
                        try:
                            os.unlink(self.video_path)
                        except OSError:
                            pass
                        self.video_path = None
                    else:
                        raw_temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                        self.raw_video_path = raw_temp.name
                        raw_temp.close()
                        self.raw_video_writer = cv2.VideoWriter(
                            self.raw_video_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            self.RECORDING_FPS,
                            (image.shape[1], image.shape[0]),
                        )
                        if not self.raw_video_writer.isOpened():
                            self.raw_video_writer.release()
                            self.raw_video_writer = None
                            self.raw_recording_error = "Could not initialize the raw local replay encoder."
                            try:
                                os.unlink(self.raw_video_path)
                            except OSError:
                                pass
                            self.raw_video_path = None
                if self.video_writer is not None:
                    self.video_writer.write(image)
                    if self.raw_video_writer is not None:
                        self.raw_video_writer.write(source_image)
                    self.recorded_frames += 1
                    self.last_recorded_at = elapsed

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def on_ended(self) -> None:
        if self.model is not None:
            self.model.close()
            self.model = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.raw_video_writer is not None:
            self.raw_video_writer.release()
            self.raw_video_writer = None

        tracked_video = None
        if self.record_video and self.video_path:
            try:
                if self.recorded_frames:
                    tracked_video = Path(self.video_path).read_bytes()
                    capture = cv2.VideoCapture(self.video_path)
                    decodes = capture.isOpened() and capture.read()[0]
                    capture.release()
                    if not decodes:
                        tracked_video = None
                        self.recording_error = "The MP4 file was created but could not be decoded for playback."
                else:
                    self.recording_error = "No camera frames arrived, so no tracked video was created."
            except OSError as exc:
                self.recording_error = f"Could not read the completed MP4: {exc}"
            finally:
                try:
                    os.unlink(self.video_path)
                except OSError:
                    pass
                self.video_path = None
        elif self.record_video and not self.recording_error:
            self.recording_error = "No camera frames arrived, so no tracked video was created."

        raw_video_path = None
        if self.record_video and self.raw_video_path:
            try:
                capture = cv2.VideoCapture(self.raw_video_path)
                decodes = capture.isOpened() and capture.read()[0]
                capture.release()
                if decodes:
                    raw_video_path = self.raw_video_path
                else:
                    self.raw_recording_error = "The raw local replay could not be decoded."
            except Exception as exc:
                self.raw_recording_error = f"Could not verify the raw local replay: {exc}"
            if raw_video_path is None:
                try:
                    os.unlink(self.raw_video_path)
                except OSError:
                    pass
            self.raw_video_path = None
        elif self.raw_video_path:
            try:
                os.unlink(self.raw_video_path)
            except OSError:
                pass
            self.raw_video_path = None

        if len(self.coach_frames) >= 6:
            frame_indexes = np.linspace(0, len(self.coach_frames) - 1, 6).round().astype(int)
            selected_coach_frames = [self.coach_frames[i] for i in sorted(set(frame_indexes.tolist()))]
        else:
            remaining = 6 - len(self.coach_frames)
            if self.reference_frames and remaining:
                ref_indexes = np.linspace(0, len(self.reference_frames) - 1,
                                          min(remaining, len(self.reference_frames))).round().astype(int)
                references = [self.reference_frames[i] for i in sorted(set(ref_indexes.tolist()))]
            else:
                references = []
            selected_coach_frames = list(self.coach_frames) + references
        selected_coach_frames.sort(key=lambda item: item.get("timestamp_s", 0.0))
        self.monitor.end_browser_session(
            tracked_video=tracked_video,
            recording_error=self.recording_error,
            recorded_duration_s=self.recorded_frames / self.RECORDING_FPS,
            coach_frames=selected_coach_frames,
            raw_video_path=raw_video_path,
            raw_video_error=self.raw_recording_error,
        )
