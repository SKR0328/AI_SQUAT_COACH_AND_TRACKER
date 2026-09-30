import unittest
import base64
import os
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import cv2
import av
import numpy as np
import requests

from lifelens.analyzer import (FrameResult, _classify_bilateral_depth, _find_rep_cycles,
                               _rep_depth_measurements, _coach_frame_payload,
                               _summarize_ascent_coordination, analyze_video)
from lifelens.live import LiveSquatMonitor
from lifelens.coach import (_coach_prompt_facts, generate_video_coaching,
                            generate_visual_coaching)
from lifelens.webrtc_live import LivePoseProcessor


def make_frame(t, hip_y, visibility=0.95):
    knee_angle = (max(80.0, 175.0 - max(0.0, (float(hip_y) - 0.4) * 500.0))
                  if hip_y is not None else None)
    return FrameResult(
        timestamp_s=float(t), hip_y=float(hip_y) if hip_y is not None else None,
        knee_y=0.52 if hip_y is not None else None,
        visibility=visibility, image=np.zeros((64, 64, 3), dtype=np.uint8), landmarks=None,
        left_hip_y=float(hip_y) if hip_y is not None else None,
        left_knee_y=0.52 if hip_y is not None else None,
        left_confidence=visibility, left_body_scale=0.50,
        right_hip_y=float(hip_y) if hip_y is not None else None,
        right_knee_y=0.52 if hip_y is not None else None,
        right_confidence=visibility, right_body_scale=0.50,
        body_scale=0.50,
        left_knee_angle_deg=knee_angle, right_knee_angle_deg=knee_angle,
        knee_angle_deg=knee_angle,
    )


def cycle(start, standing=0.4, bottom=0.56, step=0.10):
    values = [standing] * 5
    values += np.linspace(standing + 0.01, bottom, 6).tolist()
    values += [bottom] * 3
    values += np.linspace(bottom - 0.02, standing, 7).tolist()
    values += [standing] * 6
    return [(start + i * step, value) for i, value in enumerate(values)]


def live_state():
    return {
        "phase": "WAITING", "baseline_y": None, "baseline_torso": None,
        "baseline_scale": None, "bottom_y": None, "bottom_gap": None,
        "bottom_torso": None, "bottom_time": None, "bottom_visibility": None,
        "cycle_start_time": None, "descent_frames": 0, "ascent_frames": 0,
        "standing_frames": 0, "previous_y": None, "rep_count": 0,
        "reps": [], "cue": "", "depth": "",
    }


class RepQualityTests(unittest.TestCase):
    def test_bilateral_depth_accepts_a_clear_pass_on_either_side(self):
        status, passing, disagreement = _classify_bilateral_depth(
            left_margin_pct_body=12.0, left_confidence=0.92,
            right_margin_pct_body=-10.0, right_confidence=0.95,
        )
        self.assertEqual(status, "HIP PROXY BELOW KNEE PROXY ON LEFT SIDE")
        self.assertEqual(passing, ["left"])
        self.assertTrue(disagreement)

    def test_bilateral_depth_only_reports_both_sides_miss_when_both_are_clear(self):
        status, passing, _ = _classify_bilateral_depth(-9.0, 0.9, -11.0, 0.92)
        self.assertEqual(status, "HIP PROXY ABOVE KNEE PROXY ON BOTH SIDES")
        self.assertEqual(passing, [])
        status, passing, _ = _classify_bilateral_depth(-9.0, 0.9, None, 0.0)
        self.assertEqual(status, "REVIEW — OTHER SIDE UNCLEAR")
        self.assertEqual(passing, [])

    def test_offline_depth_does_not_discard_non_tracking_side_pass(self):
        samples = cycle(0.0)
        frames = [make_frame(t, y) for t, y in samples]
        for frame in frames:
            frame.left_hip_y = frame.hip_y
            frame.left_knee_y = frame.hip_y - 0.12
            frame.right_hip_y = frame.hip_y
            frame.right_knee_y = frame.hip_y + 0.08
        cycles = _find_rep_cycles(frames)
        measurement = _rep_depth_measurements(frames, cycles, lambda image, *_: image, "right")[0]
        self.assertEqual(measurement["selected_side"], "right")
        self.assertEqual(measurement["passing_sides"], ["left"])
        self.assertEqual(measurement["status"], "HIP PROXY BELOW KNEE PROXY ON LEFT SIDE")

    def test_offline_accepts_three_complete_cycles(self):
        samples = []
        for start in (0.0, 3.0, 6.0):
            samples.extend(cycle(start))
        frames = [make_frame(t, y) for t, y in samples]
        detected = _find_rep_cycles(frames)
        self.assertEqual(len(detected), 3)
        times = [frames[cycle_indices[1]].timestamp_s for cycle_indices in detected]
        self.assertEqual(times, sorted(times))

    def test_offline_rejects_small_bounce_and_single_frame_spike(self):
        values = [0.4] * 12 + [0.42, 0.44, 0.43, 0.41] + [0.4] * 10
        values += [0.4] * 10 + [0.58] + [0.4] * 10
        frames = [make_frame(i * 0.1, y) for i, y in enumerate(values)]
        self.assertEqual(_find_rep_cycles(frames), [])

    def test_offline_rejects_cycle_without_full_return(self):
        samples = [(i * 0.1, 0.4) for i in range(10)]
        samples += [(1.0 + i * 0.1, value) for i, value in enumerate(np.linspace(0.41, 0.57, 8))]
        samples += [(1.8 + i * 0.1, 0.57) for i in range(10)]
        frames = [make_frame(t, y) for t, y in samples]
        self.assertEqual(_find_rep_cycles(frames), [])

    def test_offline_rejects_hip_motion_without_knee_flexion(self):
        samples = []
        for start, hip in cycle(0.0):
            frame = make_frame(start, hip)
            frame.knee_angle_deg = 170.0
            frame.left_knee_angle_deg = 170.0
            frame.right_knee_angle_deg = 170.0
            samples.append(frame)
        self.assertEqual(_find_rep_cycles(samples), [])

    def test_live_counts_only_after_complete_return(self):
        state = live_state()
        t = 0.0
        for hip in [0.4] * 5:
            knee_angle = max(80.0, 175.0 - max(0.0, (hip - 0.4) * 500.0))
            LiveSquatMonitor._update_rep_state(
                state, t, hip, 0.52, -0.12, 0.95, 20.0, 0.50, knee_angle,
                left_gap=0.10, left_confidence=0.95, left_body_scale=0.50,
                right_gap=-0.10, right_confidence=0.95, right_body_scale=0.50,
            )
            t += 0.1
        for hip in [0.43, 0.46, 0.49, 0.52, 0.55, 0.57, 0.57,
                    0.54, 0.50, 0.46, 0.43, 0.41, 0.40, 0.40, 0.40]:
            knee_angle = max(80.0, 175.0 - max(0.0, (hip - 0.4) * 500.0))
            LiveSquatMonitor._update_rep_state(
                state, t, hip, 0.52, hip - 0.52, 0.95, 20.0, 0.50, knee_angle,
                left_gap=0.10, left_confidence=0.95, left_body_scale=0.50,
                right_gap=-0.10, right_confidence=0.95, right_body_scale=0.50,
            )
            t += 0.1
        self.assertEqual(state["rep_count"], 1)
        self.assertEqual(len(state["reps"]), 1)
        self.assertEqual(state["reps"][0]["passing_sides"], ["left"])

    def test_live_does_not_count_shallow_bounce_or_dropout(self):
        state = live_state()
        t = 0.0
        for hip in [0.4] * 5 + [0.42, 0.44, 0.42, 0.4] * 2:
            knee_angle = max(80.0, 175.0 - max(0.0, (hip - 0.4) * 500.0))
            LiveSquatMonitor._update_rep_state(state, t, hip, 0.52, hip - 0.52, 0.95, 20.0, 0.50, knee_angle)
            t += 0.1
        self.assertEqual(state["rep_count"], 0)
        # A tracking dropout discards an in-progress partial cycle.
        for hip in [0.46, 0.50]:
            LiveSquatMonitor._update_rep_state(state, t, hip, 0.52, hip - 0.52, 0.95, 20.0, 0.50)
            t += 0.1
        LiveSquatMonitor._update_rep_state(state, t, 0.50, 0.52, -0.02, 0.1, 20.0, 0.50, 120.0)
        self.assertEqual(state["rep_count"], 0)
        self.assertEqual(state["phase"], "WAITING")

    def test_live_rejects_hip_motion_without_knee_flexion(self):
        state = live_state()
        t = 0.0
        for hip in [0.4] * 5 + [0.43, 0.46, 0.49, 0.52, 0.55, 0.57, 0.57,
                                0.54, 0.50, 0.46, 0.43, 0.41, 0.40, 0.40, 0.40]:
            LiveSquatMonitor._update_rep_state(state, t, hip, 0.52, hip - 0.52,
                                               0.95, 20.0, 0.50, 170.0)
            t += 0.1
        self.assertEqual(state["rep_count"], 0)

    @patch("lifelens.coach.requests.post")
    def test_visual_coach_sends_selected_images_to_local_cpu_model(self, post):
        class MockResponse:
            def raise_for_status(self):
                return None

            @staticmethod
            def json():
                return {"message": {"content": (
                    '{"headline":"Set review","primary_cue":"Brace before descent.",'
                    '"reasoning":"The frames show the start and bottom.","observations":[],'
                    '"depth_note":"Joint proxy only.","limitation":"One camera view.",'
                    '"next_set":"Keep the camera side-on."}'
                )}}

        post.return_value = MockResponse()
        result = generate_visual_coaching({
            "result": "DEPTH PROXY NEEDS REVIEW",
            "coach_frames": [{"rep": 1, "timestamp_s": 1.2, "image": b"jpeg-bytes"}],
        })
        self.assertEqual(result["primary_cue"], "Brace before descent.")
        post.assert_called_once()
        url, = post.call_args.args
        payload = post.call_args.kwargs["json"]
        self.assertEqual(url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["options"]["num_gpu"], 0)
        self.assertEqual(len(payload["messages"][1]["images"]), 1)
        self.assertNotIn("coach_frames", payload["messages"][1]["content"])

    @patch("lifelens.coach.requests.post")
    def test_visual_coach_caps_and_downscales_frames_for_4096_context(self, post):
        class MockResponse:
            def raise_for_status(self):
                return None

            @staticmethod
            def json():
                return {"message": {"content": (
                    '{"headline":"Set review","primary_cue":"Brace.",'
                    '"reasoning":"Visible evidence.","observations":[],"depth_note":"Review.",'
                    '"limitation":"One view.","next_set":"Keep side view."}'
                )}}

        post.return_value = MockResponse()
        ok, encoded = cv2.imencode(".jpg", np.zeros((1080, 1920, 3), dtype=np.uint8))
        self.assertTrue(ok)
        image_bytes = encoded.tobytes()
        frames = [
            {"rep": i // 2 + 1, "timestamp_s": i, "image": image_bytes}
            for i in range(6)
        ]

        generate_visual_coaching({"result": "review"}, frames)

        payload = post.call_args.kwargs["json"]
        images = payload["messages"][1]["images"]
        self.assertEqual(len(images), 2)
        for image in images:
            decoded = cv2.imdecode(
                np.frombuffer(base64.b64decode(image), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            self.assertLessEqual(max(decoded.shape[:2]), 320)
        self.assertEqual(payload["options"]["num_ctx"], 4096)

    @patch("lifelens.coach.requests.post")
    def test_visual_coach_retries_context_overflow_with_one_smaller_frame(self, post):
        class OverflowResponse:
            def raise_for_status(self):
                error = requests.HTTPError("context overflow")
                error.response = self
                raise error

            @staticmethod
            def json():
                return {"error": {
                    "type": "exceed_context_size_error",
                    "message": "request exceeds the available context size",
                }}

        class SuccessResponse:
            def raise_for_status(self):
                return None

            @staticmethod
            def json():
                return {"message": {"content": (
                    '{"headline":"Set review","primary_cue":"Brace.",'
                    '"reasoning":"Visible evidence.","observations":[],"depth_note":"Review.",'
                    '"limitation":"One view.","next_set":"Keep side view."}'
                )}}

        post.side_effect = [OverflowResponse(), SuccessResponse()]
        ok, encoded = cv2.imencode(".jpg", np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertTrue(ok)
        frames = [
            {"rep": i + 1, "timestamp_s": float(i), "image": encoded.tobytes()}
            for i in range(4)
        ]

        result = generate_visual_coaching({"result": "review"}, frames)

        self.assertEqual(result["headline"], "Set review")
        self.assertEqual(post.call_count, 2)
        retry_payload = post.call_args.kwargs["json"]
        self.assertEqual(len(retry_payload["messages"][1]["images"]), 1)
        retry_image = cv2.imdecode(
            np.frombuffer(base64.b64decode(retry_payload["messages"][1]["images"][0]), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertLessEqual(max(retry_image.shape[:2]), 224)

    @patch("lifelens.coach.requests.post")
    def test_visual_coach_handles_empty_local_model_response(self, post):
        class MockResponse:
            def raise_for_status(self):
                return None

            @staticmethod
            def json():
                return {"done_reason": "length", "message": {"content": "", "thinking": "internal"}}

        post.return_value = MockResponse()
        with self.assertRaisesRegex(ValueError, "returned no coaching text"):
            generate_visual_coaching(
                {"exercise": "squat"},
                [{"rep": 1, "timestamp_s": 1.0, "image": b"jpeg-bytes"}],
            )

    @patch("lifelens.coach.subprocess.run")
    @patch("lifelens.coach._video_runtime_paths")
    def test_video_coach_uses_phase_frames_and_hides_proxy_claims(self, runtime_paths, run):
        with tempfile.TemporaryDirectory() as root:
            root = os.path.abspath(root)
            executable = Path(root) / "llama-mtmd-cli.exe"
            ffmpeg_dir = Path(root) / "ffmpeg"
            model = Path(root) / "model.gguf"
            mmproj = Path(root) / "mmproj.gguf"
            for path in (executable, model, mmproj):
                with open(path, "wb") as target:
                    target.write(b"fixture")
            os.makedirs(ffmpeg_dir)
            runtime_paths.return_value = (root, executable, ffmpeg_dir, model, mmproj)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as source:
                source.write(b"input video")
                source_path = source.name
            model_responses = [
                "OVERALL: Your evidence-based read of the visible set in one or two natural sentences. "
                "The right side shows a clear hip-knee proxy difference, with the right hip lower than the right knee, "
                "indicating a potential issue with knee tracking. The left side is not visible, so it cannot be assessed.\n"
                "SETUP: The lifter is standing with a stable and balanced posture.\n"
                "DESCENT: The descent is smooth and controlled, with a neutral spine.\n"
                "BOTTOM: The back remains straight and the knees are slightly bent.\n"
                "ASCENT: The lifter returns to standing in a controlled manner.\n"
                "NEXT SET: Focus on ensuring the right knee tracks over the right foot during the descent.\n"
                "LIMIT: None.\n",
                "OVERALL: The pelvis rises before the chest during the first part of the ascent.\n"
                "SETUP: The lifter keeps his back straight.\n"
                "DESCENT: Not clear from this view.\n"
                "BOTTOM: The lifter's hips are slightly lowered, and his knees are at a similar angle to his hips.\n"
                "ASCENT: The pelvis rises before the chest during the first part of the ascent.\n"
                "NEXT SET: Keep your chest rising with your hips through the first part of the ascent.\n"
                "LIMIT: Brief movement between sampled moments may be missed.\n",
                "OVERALL: Neutral. No clear technical fault observed across the rep.\n"
                "SETUP: Neutral.\nDESCENT: Neutral.\nBOTTOM: Neutral.\nASCENT: Neutral.\n"
                "NEXT SET: No change is justified.\nLIMIT: None.\n",
                "OVERALL: The movement appears to be controlled, with the torso and pelvis moving in a coordinated manner.\n"
                "SETUP: Not clear from this view.\nDESCENT: Not clear from this view.\n"
                "BOTTOM: The lifter's hips are slightly lowered, and his knees are at a similar angle to his hips.\n"
                "ASCENT: The lifter begins to rise from the bottom position, keeping his back straight and his knees slightly bent.\n"
                "NEXT SET: No technique change is justified from the visible evidence in this review.\n"
                "LIMIT: The lifter's movement is consistent, and there are no visible limitations that affect the execution of the set.\n",
                "OVERALL: Your pelvis rises before the chest during the ascent.\n"
                "SETUP: Not clear from this view.\nDESCENT: Not clear from this view.\n"
                "BOTTOM: Not clear from this view.\n"
                "ASCENT: Your pelvis rises before the chest during the first part of the ascent.\n"
                "NEXT SET: Keep your chest rising with your hips through the ascent.\nLIMIT: None.\n",
                "OVERALL: The lifter's back is slightly rounded, and the hips are moving forward as the lifter descends. "
                "The lifter's back is slightly rounded, and the hips are moving forward as the lifter descends. "
                "The lifter's back is slightly rounded, and the hips are moving forward as the lifter rises.\n"
                "SETUP: Not clear from this view.\n"
                "DESCENT: The lifter's back is slightly rounded, and the hips are moving forward as the lifter descends.\n"
                "BOTTOM: The lifter's back is slightly rounded, and the hips are moving forward as the lifter descends.\n"
                "ASCENT: The lifter's back is slightly rounded, and the hips are moving forward as the lifter rises.\n"
                "NEXT SET: Push your back into the bar!\nLIMIT: None.\n",
                "OVERALL: The pelvis is visibly moving forward during the descent, and the chest is not rising significantly before the barbell reaches the bottom.\n"
                "SETUP: The barbell is held in the hands, and the lifter's back is slightly rounded.\n"
                "DESCENT: The hips are moving forward during the descent, and the barbell is moving down from the shoulders.\n"
                "BOTTOM: Not clear from this view.\nASCENT: Not clear from this view.\n"
                "NEXT SET: Push your back into the barbell as you rise from the bottom position.\nLIMIT: None.\n",
            ]
            response_index = 0

            def fake_run(command, **kwargs):
                nonlocal response_index
                if command[0].endswith("ffmpeg.exe"):
                    with open(command[-1], "wb") as target:
                        target.write(b"normalized video")
                    return __import__("subprocess").CompletedProcess(command, 0, "", "")
                self.assertIn("--video", command)
                self.assertIn("--image", command)
                prompt = command[command.index("-p") + 1]
                self.assertIn("time-ordered video", prompt)
                self.assertIn("squat as a whole", prompt)
                self.assertIn("cannot reliably verify spinal shape", prompt)
                self.assertIn("Never copy the same observation", prompt)
                self.assertNotIn("left_hip_vs_knee", prompt)
                self.assertNotIn("right_hip_vs_knee", prompt)
                self.assertNotIn("private-frame-bytes", prompt)
                answer = model_responses[response_index]
                response_index += 1
                return __import__("subprocess").CompletedProcess(
                    command, 0, "2.0.0.0 I mtmd batch encoding done in 3 ms\n" + answer, ""
                )

            run.side_effect = fake_run
            try:
                result = generate_video_coaching(
                    {
                        "result": "DEPTH PROXY NEEDS REP-BY-REP REVIEW",
                        "per_rep_side_joint_proxy_measurements": [
                            {"left_hip_vs_knee_proxy_margin_pct_of_body_scale": 3.4,
                             "right_hip_vs_knee_proxy_margin_pct_of_body_scale": -2.1}
                        ],
                    },
                    source_path,
                    frames=[
                        {"rep": 1, "timestamp_s": 1.0, "image": b"private-frame-bytes"},
                        {"rep": 1, "timestamp_s": 2.0, "image": b"private-frame-bytes"},
                    ],
                )
                grounded_result = generate_video_coaching(
                    {"rep_count_estimate": 1, "movement_signals": [
                        {"rep": 1, "ascent_coordination": "possible_hips_first"}
                    ]}, source_path,
                    frames=[
                        {"rep": 1, "timestamp_s": 1.0, "image": b"private-frame-bytes"},
                        {"rep": 1, "timestamp_s": 2.0, "image": b"private-frame-bytes"},
                    ],
                )
                neutral_result = generate_video_coaching(
                    {"rep_count_estimate": 1}, source_path,
                    frames=[
                        {"rep": 1, "timestamp_s": 1.0, "image": b"private-frame-bytes"},
                        {"rep": 1, "timestamp_s": 2.0, "image": b"private-frame-bytes"},
                    ],
                )
                actual_style_result = generate_video_coaching(
                    {"rep_count_estimate": 1, "movement_signals": [
                        {"rep": 1, "ascent_coordination": "no_clear_hips_first_signal"}
                    ]}, source_path,
                    frames=[
                        {"rep": 1, "timestamp_s": 1.0, "image": b"private-frame-bytes"},
                        {"rep": 1, "timestamp_s": 2.0, "image": b"private-frame-bytes"},
                    ],
                )
                contradictory_result = generate_video_coaching(
                    {"rep_count_estimate": 1, "movement_signals": [
                        {"rep": 1, "ascent_coordination": "no_clear_hips_first_signal"}
                    ]}, source_path,
                    frames=[
                        {"rep": 1, "timestamp_s": 1.0, "image": b"private-frame-bytes"},
                        {"rep": 1, "timestamp_s": 2.0, "image": b"private-frame-bytes"},
                    ],
                )
                bad_coaching_result = generate_video_coaching(
                    {"rep_count_estimate": 1}, source_path,
                    frames=[
                        {"rep": 1, "timestamp_s": 1.0, "image": b"private-frame-bytes"},
                        {"rep": 1, "timestamp_s": 2.0, "image": b"private-frame-bytes"},
                    ],
                )
                barbell_cue_result = generate_video_coaching(
                    {"rep_count_estimate": 1}, source_path,
                    frames=[
                        {"rep": 1, "timestamp_s": 1.0, "image": b"private-frame-bytes"},
                        {"rep": 1, "timestamp_s": 2.0, "image": b"private-frame-bytes"},
                    ],
                )
            finally:
                os.unlink(source_path)

        self.assertEqual(run.call_count, 14)
        self.assertEqual(result["review_status"], "withheld")
        self.assertEqual(result["headline"], "No specific cue verified")
        self.assertNotIn("right", result["reasoning"].lower())
        self.assertNotIn("proxy", result["reasoning"].lower())
        self.assertEqual(result["observations"], [])
        self.assertEqual(grounded_result["review_status"], "grounded")
        self.assertIn("hips may start rising before your chest", grounded_result["reasoning"])
        self.assertEqual(grounded_result["observations"][0]["area"], "ascent")
        self.assertIn("hips may rise ahead of your chest", grounded_result["observations"][0]["observation"])
        self.assertIn("chest and hips up together", grounded_result["primary_cue"])
        self.assertEqual(neutral_result["review_status"], "withheld")
        self.assertEqual(neutral_result["observations"], [])
        self.assertEqual(actual_style_result["review_status"], "grounded")
        self.assertIn("tracked ascent", actual_style_result["reasoning"].lower())
        self.assertNotIn("knee", actual_style_result["reasoning"].lower())
        self.assertIn("cannot confirm spinal rounding", actual_style_result["limitation"])
        self.assertEqual(contradictory_result["review_status"], "grounded")
        self.assertIn("doesn't justify changing", contradictory_result["reasoning"])
        self.assertNotIn("pelvis rises before the chest", contradictory_result["reasoning"])
        self.assertEqual(bad_coaching_result["review_status"], "withheld")
        self.assertEqual(bad_coaching_result["headline"], "No specific cue verified")
        self.assertEqual(bad_coaching_result["observations"], [])
        self.assertNotIn("rounded", bad_coaching_result["reasoning"].lower())
        self.assertNotIn("the hips are moving forward as the lifter", bad_coaching_result["reasoning"].lower())
        self.assertNotIn("push your back", bad_coaching_result["primary_cue"].lower())
        self.assertIn("wasn't verified", bad_coaching_result["reasoning"])
        self.assertIn("isn't enough to identify a form fault", bad_coaching_result["reasoning"])
        self.assertEqual(barbell_cue_result["review_status"], "withheld")
        self.assertIn("wasn't verified", barbell_cue_result["reasoning"])
        self.assertIn("isn't enough to identify a form fault", barbell_cue_result["reasoning"])
        self.assertIn("removed the unclear", barbell_cue_result["reasoning"])
        self.assertNotIn("barbell reaches the bottom", barbell_cue_result["reasoning"])

    @patch("lifelens.coach.subprocess.run")
    @patch("lifelens.coach._video_runtime_paths")
    def test_video_coach_retries_context_exhaustion_with_reduced_sampling(self, runtime_paths, run):
        with tempfile.TemporaryDirectory() as root:
            root = os.path.abspath(root)
            executable = Path(root) / "llama-mtmd-cli.exe"
            ffmpeg_dir = Path(root) / "ffmpeg"
            model = Path(root) / "model.gguf"
            mmproj = Path(root) / "mmproj.gguf"
            for path in (executable, model, mmproj):
                path.write_bytes(b"fixture")
            ffmpeg_dir.mkdir()
            runtime_paths.return_value = (root, executable, ffmpeg_dir, model, mmproj)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as source:
                source.write(b"input video")
                source_path = source.name

            def fake_run(command, **kwargs):
                if command[0].endswith("ffmpeg.exe"):
                    Path(command[-1]).write_bytes(b"normalized video")
                    return __import__("subprocess").CompletedProcess(command, 0, "", "")
                if "--video-fps" in command:
                    fps = command[command.index("--video-fps") + 1]
                    if fps == "0.5":
                        return __import__("subprocess").CompletedProcess(
                            command, 1, "failed to find a memory slot for batch of size 64", ""
                        )
                    self.assertEqual(fps, "0.25")
                    self.assertEqual(command[command.index("-c") + 1], "12288")
                    self.assertEqual(command[command.index("-b") + 1], "24")
                    self.assertIn("reduced rate", command[command.index("-p") + 1])
                    self.assertIn("--image", command)
                    return __import__("subprocess").CompletedProcess(
                        command, 0,
                        "2.0.0.0 I mtmd batch encoding done in 3 ms\n"
                        "OVERALL: The set stays controlled across the reps.\n"
                        "SETUP: Your starting position is stable.\n"
                        "DESCENT: You lower under control.\n"
                        "BOTTOM: The turn is clear.\n"
                        "ASCENT: You return to standing smoothly.\n"
                        "NEXT SET: Exhale and reset between reps.\n"
                        "LIMIT: Brief movement between frames may be missed.\n",
                        "",
                    )
                self.fail("Expected a llama.cpp video command")

            run.side_effect = fake_run
            try:
                result = generate_video_coaching(
                    {"result": "DEPTH PROXY NEEDS REP-BY-REP REVIEW"},
                    source_path,
                    frames=[
                        {"rep": 1, "timestamp_s": 1.0, "image": b"frame-one"},
                        {"rep": 2, "timestamp_s": 3.0, "image": b"frame-two"},
                    ],
                )
            finally:
                os.unlink(source_path)

        self.assertEqual(run.call_count, 3)  # normalize + initial review + fallback
        self.assertEqual(result["review_status"], "withheld")
        self.assertEqual(result["primary_cue"], "")
        self.assertIn("couldn't verify", result["reasoning"])
        self.assertIn("fewer video samples", result["limitation"])

    def test_coach_frame_payload_samples_bottom_and_ascent_for_each_rep(self):
        frames = [make_frame(i / 10, 0.42 + 0.01 * i) for i in range(15)]
        selected = _coach_frame_payload(frames, [(0, 4, 7), (7, 10, 14)])
        self.assertEqual([item["phase"] for item in selected],
                         ["bottom", "early_ascent", "bottom", "early_ascent"])
        self.assertEqual([item["rep"] for item in selected], [1, 1, 2, 2])

    def test_whole_body_ascent_signal_is_conservative_and_has_no_side_measurements(self):
        def pose_frame(t, shoulder_y, pelvis_y, visibility=0.95):
            landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=visibility) for _ in range(33)]
            for index in (11, 12):
                landmarks[index] = SimpleNamespace(x=0.45 if index == 11 else 0.55,
                                                   y=shoulder_y, visibility=visibility)
            for index in (23, 24):
                landmarks[index] = SimpleNamespace(x=0.45 if index == 23 else 0.55,
                                                   y=pelvis_y, visibility=visibility)
            return FrameResult(t, pelvis_y, 0.6, visibility, np.zeros((24, 24, 3)), landmarks)

        together = [pose_frame(i, shoulder, pelvis) for i, (shoulder, pelvis) in enumerate((
            (0.50, 0.70), (0.47, 0.67), (0.44, 0.64), (0.40, 0.60), (0.35, 0.55), (0.30, 0.50)
        ))]
        hips_first = [pose_frame(i, shoulder, pelvis) for i, (shoulder, pelvis) in enumerate((
            (0.50, 0.70), (0.50, 0.66), (0.49, 0.62), (0.48, 0.57), (0.42, 0.53), (0.30, 0.50)
        ))]
        unclear = [pose_frame(i, shoulder, pelvis, visibility=0.5) for i, (shoulder, pelvis) in enumerate((
            (0.50, 0.70), (0.47, 0.67), (0.44, 0.64), (0.40, 0.60), (0.35, 0.55), (0.30, 0.50)
        ))]

        self.assertEqual(_summarize_ascent_coordination(together, [(0, 0, 5)])[0]["ascent_coordination"],
                         "no_clear_hips_first_signal")
        self.assertEqual(_summarize_ascent_coordination(hips_first, [(0, 0, 5)])[0]["ascent_coordination"],
                         "possible_hips_first")
        self.assertEqual(_summarize_ascent_coordination(unclear, [(0, 0, 5)])[0]["ascent_coordination"],
                         "unclear")

        facts = _coach_prompt_facts({
            "rep_count_estimate": 1,
            "movement_signals": [{"rep": 1, "ascent_coordination": "no_clear_hips_first_signal"}],
            "per_rep_side_joint_proxy_measurements": [{"left_hip_vs_knee_proxy_margin": 3.0}],
        })
        self.assertEqual(facts["movement_signals"], [
            {"rep": 1, "ascent_coordination": "no_clear_hips_first_signal"}
        ])
        self.assertNotIn("left", str(facts).lower())
        self.assertNotIn("proxy", str(facts).lower())

    def test_video_pipeline_gracefully_handles_clip_without_a_detectable_person(self):
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as temp:
            video_path = temp.name
        try:
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (320, 240))
            self.assertTrue(writer.isOpened(), "MJPG test video encoder did not open")
            for _ in range(24):
                writer.write(np.full((240, 320, 3), 235, dtype=np.uint8))
            writer.release()
            with open(video_path, "rb") as video_file:
                result = analyze_video(video_file.read(), suffix=".avi", sample_every=3)
            self.assertEqual(result.rep_count, 0)
            self.assertEqual(result.verdict, "UNABLE TO ASSESS")
            self.assertIsNotNone(result.tracking_video)
            self.assertIsNone(result.tracking_video_error)
        finally:
            try:
                os.unlink(video_path)
            except OSError:
                pass

    def test_live_processor_accepts_local_frame_without_opening_a_camera(self):
        processor = LivePoseProcessor(record_video=False)
        try:
            input_frame = av.VideoFrame.from_ndarray(
                np.full((240, 320, 3), 235, dtype=np.uint8), format="bgr24"
            )
            output_frame = processor.recv(input_frame)
            output = output_frame.to_ndarray(format="bgr24")
            self.assertEqual(output.shape, (240, 320, 3))
        finally:
            processor.on_ended()

    def test_live_processor_saves_a_temporary_raw_video_for_post_set_coaching(self):
        processor = LivePoseProcessor(record_video=True)
        try:
            source = np.full((240, 320, 3), 235, dtype=np.uint8)
            output_frame = processor.recv(av.VideoFrame.from_ndarray(source, format="bgr24"))
            output = output_frame.to_ndarray(format="bgr24")
            self.assertEqual(output.shape, source.shape)
            processor.on_ended()
            _, state = processor.monitor.snapshot()
            raw_path = state["raw_video_path"]
            self.assertTrue(raw_path and os.path.isfile(raw_path))
            raw = cv2.VideoCapture(raw_path)
            ok, raw_frame = raw.read()
            raw.release()
            self.assertTrue(ok)
            self.assertGreater(np.mean(np.abs(raw_frame.astype(float) - source.astype(float))), 0)
        finally:
            processor.on_ended()
            processor.monitor.cleanup_recorded_video()


if __name__ == "__main__":
    unittest.main()
