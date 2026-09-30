"""Local squat-video analysis using MediaPipe pose landmarks and explicit rules."""

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np


POSE_CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (27, 29), (29, 31), (27, 31), (24, 26), (26, 28),
    (28, 30), (30, 32), (28, 32),
)


@dataclass
class FrameResult:
    timestamp_s: float
    hip_y: Optional[float]
    knee_y: Optional[float]
    visibility: float
    image: np.ndarray
    landmarks: Optional[list]
    hip_index: Optional[int] = None
    knee_index: Optional[int] = None
    left_hip_y: Optional[float] = None
    left_knee_y: Optional[float] = None
    left_confidence: float = 0.0
    left_hip_index: Optional[int] = None
    left_knee_index: Optional[int] = None
    right_hip_y: Optional[float] = None
    right_knee_y: Optional[float] = None
    right_confidence: float = 0.0
    right_hip_index: Optional[int] = None
    right_knee_index: Optional[int] = None
    left_body_scale: Optional[float] = None
    right_body_scale: Optional[float] = None
    body_scale: Optional[float] = None
    left_knee_angle_deg: Optional[float] = None
    right_knee_angle_deg: Optional[float] = None
    knee_angle_deg: Optional[float] = None


@dataclass
class Analysis:
    verdict: str
    reason: str
    evidence: str
    bottom_timestamp_s: float
    bottom_frame: np.ndarray
    frames_analyzed: int
    frame_count: int
    fps: float
    hip_vs_knee: Optional[float]
    confidence: float
    rep_count: int
    cues: List[str]
    rep_bottom_times_s: List[float]
    tracking_video: Optional[bytes] = None
    tracking_video_error: Optional[str] = None
    rep_measurements: Optional[List[dict]] = None
    tracking_side: Optional[str] = None
    coach_frames: Optional[List[dict]] = None
    coach_movement_signals: Optional[List[dict]] = None

    def coach_facts(self) -> dict:
        """Send only set-level timing metadata to the coach, never side proxies."""
        rep_timeline = []
        for measurement in self.rep_measurements or []:
            rep_timeline.append({
                "rep": measurement["rep"],
                "descent_start_seconds": round(measurement["cycle_start_s"], 2),
                "bottom_seconds": round(measurement["timestamp_s"], 2),
                "standing_return_seconds": round(measurement["cycle_end_s"], 2),
            })
        return {
            "exercise": "barbell squat",
            "rep_count_estimate": self.rep_count,
            "rep_timeline_seconds": rep_timeline,
            "movement_signals": list(self.coach_movement_signals or []),
        }


def _best_side(landmarks):
    """Pick the clearer same-side shoulder/hip/knee triplet for live tracking."""
    sides = ((11, 23, 25), (12, 24, 26))
    return max(sides, key=lambda ids: min(float(landmarks[i].visibility) for i in ids))


def _bilateral_pose_metrics(landmarks, width: int, height: int) -> dict:
    """Measure both same-side hip/knee proxies for live depth while preserving confidence."""
    output = {}
    for side, shoulder_i, hip_i, knee_i, ankle_i in (
        ("left", 11, 23, 25, 27), ("right", 12, 24, 26, 28),
    ):
        confidence = min(float(landmarks[i].visibility) for i in (shoulder_i, hip_i, knee_i))
        hip_y, knee_y = float(landmarks[hip_i].y), float(landmarks[knee_i].y)
        body_scale = abs(float(landmarks[ankle_i].y) - float(landmarks[shoulder_i].y))
        knee_angle = (
            _joint_angle_degrees(landmarks[hip_i], landmarks[knee_i], landmarks[ankle_i], width, height)
            if min(float(landmarks[i].visibility) for i in (hip_i, knee_i, ankle_i)) >= 0.50 else None
        )
        output.update({
            f"{side}_hip_y": hip_y,
            f"{side}_knee_y": knee_y,
            f"{side}_gap": hip_y - knee_y,
            f"{side}_confidence": confidence,
            f"{side}_body_scale": body_scale if np.isfinite(body_scale) and body_scale > 0 else None,
            f"{side}_knee_angle_deg": knee_angle,
        })
    return output


def _joint_angle_degrees(point_a, point_b, point_c, width: int, height: int) -> Optional[float]:
    """2D angle at point B, with normalized landmarks corrected to image aspect ratio."""
    try:
        def xy(point):
            if hasattr(point, "x") and hasattr(point, "y"):
                return float(point.x), float(point.y)
            return float(point[0]), float(point[1])
        ax, ay = xy(point_a)
        bx, by = xy(point_b)
        cx, cy = xy(point_c)
        a = np.asarray([ax * width, ay * height], dtype=np.float64)
        b = np.asarray([bx * width, by * height], dtype=np.float64)
        c = np.asarray([cx * width, cy * height], dtype=np.float64)
        v1, v2 = a - b, c - b
        denominator = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denominator < 1e-6:
            return None
        cosine = float(np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))
    except (AttributeError, TypeError, ValueError):
        return None


def _draw_pose(frame: np.ndarray, landmarks, hip_index: int = None, knee_index: int = None) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    points = {}
    for index, lm in enumerate(landmarks):
        if lm.visibility >= 0.35:
            points[index] = (int(lm.x * width), int(lm.y * height))
    for start, end in POSE_CONNECTIONS:
        if start in points and end in points:
            cv2.line(output, points[start], points[end], (45, 205, 120), 2, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(output, point, 4, (50, 210, 255), -1, cv2.LINE_AA)
    _draw_side_markers(output, points, ((23, 25, "L"), (24, 26, "R")))
    return output


def _draw_side_markers(output: np.ndarray, points: dict, sides) -> None:
    """Draw separately labeled hip/knee joint proxies for both sides."""
    h, _ = output.shape[:2]
    colors = {"L": ((255, 150, 0), (255, 255, 0)),
              "R": ((0, 0, 255), (0, 165, 255))}
    for hip_i, knee_i, label in sides:
        hip_color, knee_color = colors[label]
        if hip_i in points and knee_i in points:
            cv2.line(output, points[hip_i], points[knee_i], (235, 235, 235), 2, cv2.LINE_AA)
        for index, color, name in ((hip_i, hip_color, f"{label} HIP"),
                                   (knee_i, knee_color, f"{label} KNEE")):
            if index in points:
                x, y = points[index]
                cv2.circle(output, (x, y), 8, color, -1, cv2.LINE_AA)
                cv2.putText(output, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.43, color, 2, cv2.LINE_AA)
    cv2.putText(output, "L: blue/cyan   R: red/orange | joint proxies", (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


def _lock_tracking_side(frames: List[FrameResult]) -> str:
    """Choose one side for rep tracking across the entire clip to avoid side switching."""
    left_scores = [frame.left_confidence for frame in frames if frame.left_hip_y is not None]
    right_scores = [frame.right_confidence for frame in frames if frame.right_hip_y is not None]
    side = "left" if (np.mean(left_scores) if left_scores else 0.0) >= (
        np.mean(right_scores) if right_scores else 0.0) else "right"
    for frame in frames:
        if side == "left":
            frame.hip_y, frame.knee_y = frame.left_hip_y, frame.left_knee_y
            frame.visibility = frame.left_confidence
            frame.hip_index, frame.knee_index = frame.left_hip_index, frame.left_knee_index
            frame.body_scale = frame.left_body_scale
            frame.knee_angle_deg = frame.left_knee_angle_deg
        else:
            frame.hip_y, frame.knee_y = frame.right_hip_y, frame.right_knee_y
            frame.visibility = frame.right_confidence
            frame.hip_index, frame.knee_index = frame.right_hip_index, frame.right_knee_index
            frame.body_scale = frame.right_body_scale
            frame.knee_angle_deg = frame.right_knee_angle_deg
    return side


def _classify_bilateral_depth(left_margin_pct_body, left_confidence,
                              right_margin_pct_body, right_confidence,
                              review_band=2.0, min_visibility=0.50):
    """A clear proxy pass on either visible side counts; fail only if both miss."""
    values = {
        "left": (left_margin_pct_body, float(left_confidence or 0.0)),
        "right": (right_margin_pct_body, float(right_confidence or 0.0)),
    }
    reliable = {side: margin for side, (margin, confidence) in values.items()
                if margin is not None and confidence >= min_visibility}
    passing = [side for side, margin in reliable.items() if margin > review_band]
    if passing:
        status = ("HIP PROXY BELOW KNEE PROXY ON BOTH SIDES" if len(passing) == 2
                  else f"HIP PROXY BELOW KNEE PROXY ON {passing[0].upper()} SIDE")
    elif len(reliable) == 2 and all(margin < -review_band for margin in reliable.values()):
        status = "HIP PROXY ABOVE KNEE PROXY ON BOTH SIDES"
    elif not reliable:
        status = "REVIEW — BOTH SIDES UNCLEAR"
    elif any(abs(margin) <= review_band for margin in reliable.values()):
        status = "REVIEW — NEAR PROXY LINE"
    else:
        status = "REVIEW — OTHER SIDE UNCLEAR"
    margins = list(reliable.values())
    disagreement = (any(margin > review_band for margin in margins)
                    and any(margin < -review_band for margin in margins))
    return status, passing, disagreement


def _rep_depth_measurements(usable: List[FrameResult], rep_cycles: List[tuple], draw_func,
                           tracking_side: str) -> List[dict]:
    measurements = []
    for rep_number, (start_index, index, end_index) in enumerate(rep_cycles, start=1):
        frame = usable[index]
        # A short median around the bottom is less sensitive to a one-frame
        # landmark jump than selecting one exact frame.
        window = usable[max(start_index, index - 2):min(end_index + 1, index + 3)]

        def side_measurements(side):
            hip_attr, knee_attr = f"{side}_hip_y", f"{side}_knee_y"
            confidence_attr, scale_attr = f"{side}_confidence", f"{side}_body_scale"
            side_confidence = float(np.median(
                [getattr(candidate, confidence_attr) for candidate in window]
            )) if window else 0.0
            side_frames = [candidate for candidate in window
                           if getattr(candidate, confidence_attr) >= 0.50
                           and getattr(candidate, hip_attr) is not None
                           and getattr(candidate, knee_attr) is not None
                           and getattr(candidate, scale_attr) is not None
                           and getattr(candidate, scale_attr) > 0]
            if len(side_frames) < 2:
                return None, None, side_confidence
            gap = float(np.median([getattr(candidate, hip_attr) - getattr(candidate, knee_attr)
                                   for candidate in side_frames]))
            scale = float(np.median([getattr(candidate, scale_attr) for candidate in side_frames]))
            margin = gap / scale * 100 if scale > 0 else None
            return gap, margin, side_confidence

        left_gap, left_margin, left_confidence = side_measurements("left")
        right_gap, right_margin, right_confidence = side_measurements("right")
        selected_gap = left_gap if tracking_side == "left" else right_gap
        selected_margin = left_margin if tracking_side == "left" else right_margin
        selected_confidence = left_confidence if tracking_side == "left" else right_confidence
        status, passing_sides, side_disagreement = _classify_bilateral_depth(
            left_margin, left_confidence, right_margin, right_confidence
        )

        overlay = draw_func(frame.image, frame.landmarks, frame.hip_index, frame.knee_index)
        measurements.append({
            "rep": rep_number,
            "timestamp_s": frame.timestamp_s,
            "cycle_start_s": usable[start_index].timestamp_s,
            "cycle_end_s": usable[end_index].timestamp_s,
            "left_margin_pct": left_gap * 100 if left_gap is not None else None,
            "left_margin_pct_body": left_margin,
            "left_confidence": left_confidence,
            "right_margin_pct": right_gap * 100 if right_gap is not None else None,
            "right_margin_pct_body": right_margin,
            "right_confidence": right_confidence,
            "selected_side": tracking_side,
            "selected_margin_pct_body": selected_margin,
            "selected_confidence": selected_confidence,
            "status": status,
            "side_disagreement": side_disagreement,
            "passing_sides": passing_sides,
            "frame": overlay,
        })
    return measurements


def _find_rep_cycles(frames: List[FrameResult]) -> List[tuple]:
    """Return conservative top-to-bottom-to-top squat cycles as usable-frame indexes.

    A count requires a clearly visible complete excursion and return to standing.
    This deliberately favors missed/uncertain reps over counting a small bounce.
    """
    valid = [(i, f) for i, f in enumerate(frames)
             if f.hip_y is not None and f.visibility >= 0.50
             and f.body_scale is not None and f.body_scale > 0
             and f.knee_angle_deg is not None]
    if len(valid) < 6:
        return []

    times = np.asarray([f.timestamp_s for _, f in valid], dtype=np.float64)
    positions = np.asarray([f.hip_y for _, f in valid], dtype=np.float64)
    knee_angles = [f.knee_angle_deg for _, f in valid]
    scales = np.asarray([f.body_scale for _, f in valid], dtype=np.float64)
    # Close short pose dropouts into separate sequences instead of bridging them.
    breaks = [0] + [i for i in range(1, len(valid))
                    if times[i] - times[i - 1] > 0.70
                    or valid[i][0] - valid[i - 1][0] > 3] + [len(valid)]
    cycles = []
    usable_offset = 0

    for segment_number in range(len(breaks) - 1):
        left, right = breaks[segment_number], breaks[segment_number + 1]
        segment = positions[left:right]
        segment_times = times[left:right]
        if len(segment) < 6:
            usable_offset += len(segment)
            continue
        # Median filtering suppresses single-frame pose spikes while preserving bottoms.
        smooth = segment.copy()
        if len(segment) >= 3:
            for i in range(1, len(segment) - 1):
                smooth[i] = float(np.median(segment[i - 1:i + 2]))
        scale = float(np.median(scales[left:right]))
        baseline = float(np.percentile(smooth, 10))
        excursion = float(np.percentile(smooth, 95) - baseline)
        start_threshold = max(0.045, scale * 0.11)
        required_excursion = max(0.075, scale * 0.16)
        return_threshold = max(0.022, scale * 0.045)
        if excursion < required_excursion:
            usable_offset += len(segment)
            continue

        active_start = None
        bottom = None
        top_count = 0
        for i, value in enumerate(smooth):
            if active_start is None:
                if value >= baseline + start_threshold:
                    active_start = i
                    bottom = i
                    top_count = 0
                continue

            if value > smooth[bottom]:
                bottom = i
            if value <= baseline + return_threshold:
                top_count += 1
            else:
                top_count = 0

            if top_count >= 3:
                end = i
                cycle_duration = float(segment_times[end] - segment_times[active_start])
                rise_after_bottom = float(smooth[bottom] - smooth[bottom:min(end + 1, len(smooth))][-1])
                valid_cycle = (
                    bottom - active_start >= 2
                    and end - bottom >= 2
                    and cycle_duration >= 0.80
                    and smooth[bottom] - baseline >= required_excursion
                    and rise_after_bottom >= required_excursion * 0.70
                )
                top_angles = [knee_angles[left + j] for j, value in enumerate(smooth)
                              if knee_angles[left + j] is not None
                              and value <= baseline + return_threshold]
                bottom_angles = [knee_angles[j] for j in range(
                    max(left, left + bottom - 1), min(right, left + bottom + 2))
                    if knee_angles[j] is not None]
                knee_flexion_confirmed = bool(
                    top_angles and bottom_angles
                    and float(np.median(top_angles)) >= 150.0
                    and float(np.median(bottom_angles)) <= 155.0
                    and float(np.median(top_angles)) - float(np.median(bottom_angles)) >= 15.0
                )
                if valid_cycle and knee_flexion_confirmed:
                    cycles.append((usable_offset + active_start, usable_offset + bottom, usable_offset + end))
                active_start = None
                bottom = None
                top_count = 0
        usable_offset += len(segment)
    return cycles


def _render_tracking_video(video_path: str, frames: List[FrameResult], fps: float,
                           draw_pose_func=None):
    """Render the whole source clip with the latest sampled pose over each frame."""
    if not frames:
        return None, "No sampled frames were available to render."
    draw_pose_func = draw_pose_func or _draw_pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "OpenCV could not reopen the uploaded clip after pose analysis."
    ok, first = cap.read()
    if not ok:
        cap.release()
        return None, "OpenCV reopened the clip but could not decode its first frame for export."
    h, w = first.shape[:2]
    if max(h, w) > 720:
        scale = 720.0 / max(h, w)
        w, h = int(w * scale), int(h * scale)
    # MP4 encoders commonly require even dimensions; resize every source frame
    # to this exact size so portrait clips and odd source dimensions also work.
    w -= w % 2
    h -= h % 2
    if w < 2 or h < 2:
        cap.release()
        return None, "The source video dimensions are too small for MP4 export."
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_output:
        output_path = temp_output.name
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        try:
            os.unlink(output_path)
        except OSError:
            pass
        return None, f"OpenCV could not initialize the MP4V encoder for {w}×{h} at {fps:.2f} FPS."

    frame_index = 0
    pose_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / fps
            while pose_index + 1 < len(frames) and frames[pose_index + 1].timestamp_s <= timestamp:
                pose_index += 1
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            pose = frames[pose_index]
            if pose.landmarks is not None:
                frame = draw_pose_func(frame, pose.landmarks, pose.hip_index, pose.knee_index)
            else:
                cv2.putText(frame, "POSE NOT DETECTED", (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{timestamp:.2f}s", (12, 62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    browser_output_path = None
    try:
        output = Path(output_path).read_bytes()
        if not output or frame_index == 0:
            return None, "The MP4 encoder opened but produced no video frames."
        check = cv2.VideoCapture(output_path)
        decodes = check.isOpened() and check.read()[0]
        check.release()
        if not decodes:
            return None, "The MP4 encoder produced a file, but OpenCV could not decode it afterward."

        # OpenCV's MP4V output is decodable locally but unsupported by some
        # browser players. Re-encode to H.264/AVC so Streamlit can preview it.
        project_ffmpeg = Path(__file__).resolve().parent.parent / ".runtime" / "ffmpeg" / "bin" / "ffmpeg.exe"
        ffmpeg = str(project_ffmpeg) if project_ffmpeg.is_file() else shutil.which("ffmpeg")
        if not ffmpeg:
            return None, "A browser-compatible H.264 encoder (FFmpeg) was not found."
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as browser_output:
            browser_output_path = browser_output.name
        converted = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", output_path,
                "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                browser_output_path,
            ],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if converted.returncode != 0 or not Path(browser_output_path).is_file():
            detail = (converted.stderr or "").strip()[-300:]
            return None, "FFmpeg could not encode a browser-compatible H.264 preview." + (f" {detail}" if detail else "")
        browser_output = Path(browser_output_path).read_bytes()
        if not browser_output:
            return None, "FFmpeg created an empty H.264 preview."
        return browser_output, None
    except subprocess.TimeoutExpired:
        return None, "Creating the browser-compatible H.264 preview took too long."
    except OSError:
        return None, "The rendered tracking video could not be read or converted."
    finally:
        for temp_path in (output_path, browser_output_path):
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


def _find_rep_bottoms(frames: List[FrameResult]) -> List[int]:
    """Compatibility wrapper returning only bottom indexes among usable frames."""
    return [cycle[1] for cycle in _find_rep_cycles(frames)]


def _coach_frame_payload(frames: List[FrameResult], cycles: List[tuple], limit: int = 6) -> List[dict]:
    """Encode raw phase frames for local coaching (never annotated frames)."""
    if not frames:
        return []
    picks = []
    usable = [f for f in frames if f.hip_y is not None]
    if cycles:
        chosen_cycles = list(enumerate(cycles, start=1))
        if len(chosen_cycles) * 2 > limit:
            slots = max(1, limit // 2)
            indices = np.linspace(0, len(chosen_cycles) - 1, slots).round().astype(int)
            chosen_cycles = [chosen_cycles[i] for i in sorted(set(indices.tolist()))]
        for rep_number, (start, bottom, end) in chosen_cycles:
            # The full video remains in the prompt. These crisp frames add
            # evidence at the deepest position and during ascent, where a
            # hips-before-chest pattern can otherwise be lost in sparse samples.
            ascent = bottom + max(1, round((end - bottom) * 0.40))
            picks.extend(((rep_number, "bottom", frames[bottom]),
                          (rep_number, "early_ascent", frames[min(ascent, end)])))
    else:
        # Even when the rep detector abstains, the coach can still inspect the
        # set-up/movement frames and explain what visual evidence is missing.
        available = [frame for frame in frames if frame.image is not None]
        if available:
            indices = np.linspace(0, len(available) - 1, min(limit, len(available))).round().astype(int)
            picks = [(None, "set_sequence", available[i]) for i in sorted(set(indices.tolist()))]

    output = []
    for rep_number, phase, frame in picks[:limit]:
        image = frame.image
        h, w = image.shape[:2]
        if max(h, w) > 640:
            scale = 640.0 / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 76])
        if ok:
            output.append({"rep": rep_number, "phase": phase,
                           "timestamp_s": round(frame.timestamp_s, 2),
                           "image": encoded.tobytes()})
    return output


def _summarize_ascent_coordination(frames: List[FrameResult], cycles: List[tuple]) -> List[dict]:
    """Return conservative whole-body ascent signals without exposing side/joint metrics.

    This is a screening signal from 2-D pose landmarks, not a diagnosis or a
    validated judging rule. A hips-first cue is emitted only when the pelvis
    rises materially ahead of the shoulder line across consecutive frames.
    """
    summaries = []

    def center(landmarks, indices):
        points = []
        for index in indices:
            try:
                lm = landmarks[index]
                visibility = float(lm.visibility)
                if visibility >= 0.65:
                    points.append((float(lm.x), float(lm.y), visibility))
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
        if not points:
            return None
        weights = [item[2] ** 2 for item in points]
        total = sum(weights)
        return (sum(p[0] * w for p, w in zip(points, weights)) / total,
                sum(p[1] * w for p, w in zip(points, weights)) / total,
                min(p[2] for p in points))

    for rep_number, (start, bottom, end) in enumerate(cycles, start=1):
        if bottom < 0 or end >= len(frames) or end - bottom < 3:
            summaries.append({"rep": rep_number, "ascent_coordination": "unclear"})
            continue
        samples = []
        for frame in frames[bottom:end + 1]:
            landmarks = frame.landmarks
            if not landmarks:
                continue
            shoulders = center(landmarks, (11, 12))
            pelvis = center(landmarks, (23, 24))
            if shoulders is None or pelvis is None:
                continue
            samples.append((frame.timestamp_s, shoulders, pelvis))
        if len(samples) < 5:
            summaries.append({"rep": rep_number, "ascent_coordination": "unclear"})
            continue

        shoulder_bottom, pelvis_bottom = samples[0][1], samples[0][2]
        shoulder_end, pelvis_end = samples[-1][1], samples[-1][2]
        torso_scale = abs(pelvis_bottom[1] - shoulder_bottom[1])
        total_pelvis_rise = pelvis_bottom[1] - pelvis_end[1]
        if torso_scale < 0.08 or total_pelvis_rise < torso_scale * 0.20:
            summaries.append({"rep": rep_number, "ascent_coordination": "unclear"})
            continue

        early_lags = []
        min_visibility = 1.0
        for _, shoulders, pelvis in samples[1:-1]:
            pelvis_rise = pelvis_bottom[1] - pelvis[1]
            progress = pelvis_rise / total_pelvis_rise
            if 0.12 <= progress <= 0.70:
                shoulder_rise = shoulder_bottom[1] - shoulders[1]
                early_lags.append((pelvis_rise - shoulder_rise) / torso_scale)
                min_visibility = min(min_visibility, shoulders[2], pelvis[2])

        if len(early_lags) < 3 or min_visibility < 0.65:
            coordination = "unclear"
        else:
            threshold = 0.15
            above = [lag >= threshold for lag in early_lags]
            consecutive = any(above[i] and above[i + 1] for i in range(len(above) - 1))
            coordination = "possible_hips_first" if consecutive else "no_clear_hips_first_signal"
        summaries.append({"rep": rep_number, "ascent_coordination": coordination})
    return summaries


def analyze_video(video_bytes: bytes, suffix: str = ".mp4", sample_every: int = 2) -> Analysis:
    """Analyze a local clip. OpenCV's temporary decode file is deleted afterward."""
    if not video_bytes:
        raise ValueError("The selected video is empty.")
    safe_suffix = suffix.lower() if suffix.lower() in (".mp4", ".mov", ".avi", ".m4v") else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=safe_suffix, delete=False) as temp:
        temp.write(video_bytes)
        temp_path = temp.name

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        cap.release()
        os.unlink(temp_path)
        raise ValueError("Could not decode this clip. Try an MP4 video encoded with H.264.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps < 1:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    results: List[FrameResult] = []
    tracking_video = None
    tracking_video_error = None
    model_path = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model not found: {model_path}. See the README setup steps.")
    vision = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        with vision.PoseLandmarker.create_from_options(options) as model:
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % max(1, sample_every) == 0:
                    h, w = frame.shape[:2]
                    if max(h, w) > 720:
                        scale = 720.0 / max(h, w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp = frame_index / fps
                    detected = model.detect_for_video(mp_image, int(timestamp * 1000))
                    if detected.pose_landmarks:
                        lms = detected.pose_landmarks[0]
                        left_conf = min(float(lms[i].visibility) for i in (23, 25))
                        right_conf = min(float(lms[i].visibility) for i in (24, 26))
                        results.append(FrameResult(
                            timestamp, None, None, 0.0, frame, lms,
                            left_hip_y=float(lms[23].y), left_knee_y=float(lms[25].y),
                            left_confidence=left_conf, left_hip_index=23, left_knee_index=25,
                            right_hip_y=float(lms[24].y), right_knee_y=float(lms[26].y),
                            right_confidence=right_conf, right_hip_index=24, right_knee_index=26,
                            left_body_scale=(
                                abs(float(lms[27].y) - float(lms[11].y))
                                if min(float(lms[i].visibility) for i in (11, 27)) >= 0.50 else None
                            ),
                            right_body_scale=(
                                abs(float(lms[28].y) - float(lms[12].y))
                                if min(float(lms[i].visibility) for i in (12, 28)) >= 0.50 else None
                            ),
                            left_knee_angle_deg=(
                                _joint_angle_degrees(lms[23], lms[25], lms[27], frame.shape[1], frame.shape[0])
                                if min(float(lms[i].visibility) for i in (23, 25, 27)) >= 0.50 else None
                            ),
                            right_knee_angle_deg=(
                                _joint_angle_degrees(lms[24], lms[26], lms[28], frame.shape[1], frame.shape[0])
                                if min(float(lms[i].visibility) for i in (24, 26, 28)) >= 0.50 else None
                            ),
                        ))
                    else:
                        results.append(FrameResult(timestamp, None, None, 0.0, frame, None))
                frame_index += 1
        # Release the decoder before reopening the same temporary file. On
        # Windows, keeping both capture handles alive can block the second read.
        cap.release()
        tracking_video, tracking_video_error = _render_tracking_video(temp_path, results, fps)
    finally:
        cap.release()
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    tracking_side = _lock_tracking_side(results)
    usable = [frame for frame in results
              if frame.hip_y is not None and frame.visibility >= 0.50
              and frame.body_scale is not None and frame.body_scale > 0
              and frame.knee_angle_deg is not None]
    rep_cycles = _find_rep_cycles(results)
    rep_bottom_times = [usable[cycle[1]].timestamp_s for cycle in rep_cycles] if usable else []
    rep_measurements = (_rep_depth_measurements(usable, rep_cycles, _draw_pose, tracking_side)
                        if usable and rep_cycles else [])
    coach_frames = _coach_frame_payload(usable if usable else results, rep_cycles)
    coach_movement_signals = (_summarize_ascent_coordination(usable, rep_cycles)
                              if usable and rep_cycles else [])
    rep_count = len(rep_bottom_times)
    if not usable:
        best = results[0] if results else None
        if best is None:
            raise ValueError("No frames could be analyzed.")
        return Analysis("UNABLE TO ASSESS", "The pose model could not locate the lifter.",
                        "No usable shoulder, hip and knee landmarks were found.",
                        best.timestamp_s, best.image, len(results), frame_count,
                        fps, None, 0.0, rep_count,
                        ["Try a fixed side view with the full body visible."], rep_bottom_times,
                        tracking_video, tracking_video_error, rep_measurements, tracking_side, coach_frames)

    if not rep_measurements:
        bottom = max(usable, key=lambda frame: frame.hip_y)
        gap = float(bottom.hip_y - bottom.knee_y)
        confidence = bottom.visibility
        verdict = "REP COUNT UNCONFIRMED — REVIEW VIDEO"
        reason = ("LiftLens did not confirm a complete, high-confidence squat cycle, so it is withholding a rep count "
                  "and per-rep depth verdict.")
        evidence = "No complete top-to-bottom-to-top cycle passed the conservative motion and visibility checks."
        cues = ["Keep the whole body visible and record a clear standing start and return; review the clip manually."]
        selected_time = bottom.timestamp_s
        overlay = _draw_pose(bottom.image, bottom.landmarks, bottom.hip_index, bottom.knee_index)
    else:
        passed = [item for item in rep_measurements if item["passing_sides"]]
        if len(passed) == len(rep_measurements):
            verdict = "DEPTH PROXY REACHED ON AT LEAST ONE SIDE IN ALL DETECTED REPS"
            reason = "LiftLens estimates that its depth threshold was reached on every detected rep. This is a training estimate, not an official referee decision."
            cues = ["Keep using a consistent camera view; the coach reviews form separately from the depth estimate."]
            evidence = f"LiftLens marked its depth estimate as reached for all {len(rep_measurements)} detected reps."
        elif passed:
            verdict = f"DEPTH PROXY REACHED ON AT LEAST ONE SIDE IN {len(passed)} OF {len(rep_measurements)} REPS"
            reason = f"LiftLens estimates its depth threshold was reached on {len(passed)} of {len(rep_measurements)} detected reps; the rest need review."
            cues = ["Review the reps marked uncertain before making a depth-related technique change."]
            evidence = f"LiftLens marked its depth estimate as reached for {len(passed)} of {len(rep_measurements)} detected reps."
        elif all(item["status"] == "HIP PROXY ABOVE KNEE PROXY ON BOTH SIDES" for item in rep_measurements):
            verdict = "HIP PROXY ABOVE KNEE PROXY ON BOTH SIDES IN ALL DETECTED REPS"
            reason = "LiftLens did not mark its depth threshold as reached on the detected reps. This is a training estimate, not an official referee decision."
            cues = ["Review the bottom frames before changing your technique based on this estimate."]
            evidence = f"LiftLens did not mark its depth estimate as reached for any of the {len(rep_measurements)} detected reps."
        else:
            verdict = "DEPTH PROXY NEEDS REP-BY-REP REVIEW"
            reason = "LiftLens could not make a dependable depth estimate for the detected reps."
            cues = ["Review the bottom frames before changing your technique based on this estimate."]
            evidence = "One or more reps need visual review before LiftLens can summarize the depth estimate."

        # Make the lead frame the least certain/highest rep, not an unrelated
        # deepest frame elsewhere in the clip.
        candidate = min(
            rep_measurements,
            key=lambda item: (0 if item["status"].startswith("REVIEW") else 1,
                              item["selected_margin_pct_body"]
                              if item["selected_margin_pct_body"] is not None else 0.0),
        )
        bottom = next(frame for frame in usable if abs(frame.timestamp_s - candidate["timestamp_s"]) < 1e-6)
        gap = float(bottom.hip_y - bottom.knee_y)
        confidence = bottom.visibility
        selected_time = bottom.timestamp_s
        overlay = candidate["frame"]

    return Analysis(verdict, reason, evidence, selected_time, overlay,
                    len(results), frame_count, fps, gap, confidence, rep_count, cues,
                    rep_bottom_times, tracking_video, tracking_video_error,
                    rep_measurements, tracking_side, coach_frames, coach_movement_signals)
