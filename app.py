import cv2
import importlib
import os
import tempfile
import streamlit as st
from PIL import Image

import lifelens.analyzer as analyzer_module
import lifelens.coach as coach_module
from lifelens.live import get_live_monitor
from lifelens.webrtc_live import LivePoseProcessor
from streamlit_webrtc import webrtc_streamer

# Streamlit reruns this script while keeping imported modules in sys.modules.
# Reload the analysis and coach modules so local edits are used immediately
# without restarting the server or discarding an uploaded video from the session.
analyzer_module = importlib.reload(analyzer_module)
analyze_video = analyzer_module.analyze_video
coach_module = importlib.reload(coach_module)
generate_video_coaching = coach_module.generate_video_coaching


st.set_page_config(page_title="LiftLens · Squat Coach", page_icon="🏋️", layout="wide")

# Drop any comparison result left in a Streamlit session by an older version.
st.session_state.pop("comparison", None)

ANALYSIS_SCHEMA_VERSION = 20
if st.session_state.get("analysis_schema_version") != ANALYSIS_SCHEMA_VERSION:
    st.session_state.pop("analysis", None)
    st.session_state.pop("coach", None)
    st.session_state.pop("live_coach", None)
    st.session_state["analysis_schema_version"] = ANALYSIS_SCHEMA_VERSION

COACH_REVIEW_VERSION = 7
if st.session_state.get("coach_review_version") != COACH_REVIEW_VERSION:
    st.session_state.pop("coach", None)
    st.session_state.pop("live_coach", None)
    st.session_state["coach_review_version"] = COACH_REVIEW_VERSION


def clear_previous_result():
    st.session_state.pop("analysis", None)
    st.session_state.pop("coach", None)


def _depth_label(status: str) -> str:
    status = str(status or "").upper()
    if "REVIEW" in status or "UNCLEAR" in status:
        return "Needs review"
    if "BELOW KNEE PROXY" in status:
        return "Estimate reached"
    if "ABOVE KNEE PROXY" in status:
        return "Estimate not reached"
    return "Not assessed"


def _live_rep_summary(reps):
    """Show simple per-rep results without side-by-side measurements."""
    rows = []
    for rep in reps:
        status = "Estimate reached" if rep.get("passing_sides") else _depth_label(rep.get("depth_status"))
        rows.append({
            "Rep": rep.get("rep"),
            "Bottom time": f"{float(rep.get('bottom_time_s', 0.0)):.2f}s",
            "Depth estimate": status,
        })
    return rows


st.title("🏋️ LiftLens")
st.subheader("Your local AI squat coach")
st.write("Analyze a squat clip on your computer, inspect the depth estimate, and get a local AI review of your whole-squat form.")

live_monitor = get_live_monitor()
st.divider()
st.header("Live squat session")
st.write("Start the browser camera for a live pose overlay and rep count. After you stop, the local video coach can review the recorded movement and suggest practical form cues.")
record_live = st.checkbox(
    "Record a local replay for full-video coaching and tracked-video download",
    value=True,
    disabled=live_monitor.is_running(),
    help="When enabled, LiftLens records both an unannotated replay for local AI review and a tracked replay for download. Both stay on this computer; the temporary raw replay is deleted after coaching or when a new live session starts.",
)
live_ctx = webrtc_streamer(
    key="liftlens_live_camera",
    video_processor_factory=lambda: LivePoseProcessor(record_video=record_live),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    media_toggle_controls=False,
)

with st.expander("How LiftLens estimates live reps and squat depth"):
    st.markdown("""
    **Rep count**

    MediaPipe estimates body landmarks. LiftLens follows the clearest movement signal for the session and counts only after a clear descent, direction change, and stable return to standing. A pose dropout or a movement that is too small cancels the pending count. This errs toward missing an unclear rep rather than counting a small bounce.

    **Depth estimate**

    For depth, LiftLens evaluates pose landmarks around each detected rep's bottom. A clear pass from either internal landmark path is enough for that rep to count as reaching the app's depth estimate; if the signal is unclear, LiftLens asks for a visual review. The page reports the overall result without separate per-side measurements.

    **What this means**

    The depth estimate is not an IPF referee decision, and LiftLens cannot promise zero counting errors. After the set, local Qwen3-VL reviews the time-ordered recording and representative raw frames from the bottom and early ascent. That review runs after capture, so it does not interrupt live pose tracking. It focuses on visible whole-squat form, independently of the app's depth result.
    """)

live_running = bool(live_ctx.state.playing)
if live_running:
    st.session_state.pop("live_coach", None)
if live_running:
    if record_live:
        st.caption("Live camera connected. Frames are analyzed and the annotated video is recorded locally; Stop to finish the video.")
    else:
        st.caption("Live camera connected. Pose analysis runs locally; video recording is off.")
else:
    st.caption("Click START above and allow camera access when your browser asks. The camera stays off until then.")

@st.fragment(run_every=0.3 if live_running else None, key="live_pose_metrics")
def render_live_metrics():
    _, current = live_monitor.snapshot()
    if current["error"]:
        st.error(current["error"])
    if current["rep_count"] or current["confidence"]:
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Reps", current["rep_count"])
        metric_col2.metric("Movement phase", current["phase"])
        metric_col3.metric("Pose visibility", f"{current['confidence']:.0%}")
        st.write(f"**Current depth estimate:** {_depth_label(current['depth'])}")
        st.info(current["cue"])
    elif not current["error"] and not live_running:
        st.info("Start a session to turn on the camera. No camera frames are processed before you start.")
    if current["reps"]:
        st.markdown("#### Live rep review")
        st.dataframe(_live_rep_summary(current["reps"]), use_container_width=True, hide_index=True)

render_live_metrics()

_, current_live_state = live_monitor.snapshot()
if current_live_state["reps"] or current_live_state.get("raw_video_path"):
    if not current_live_state["running"]:
        if current_live_state["reps"]:
            st.markdown("#### Live rep results")
            st.dataframe(_live_rep_summary(current_live_state["reps"]), use_container_width=True, hide_index=True)
    if not current_live_state["running"]:
        if st.button("Get local post-set coaching", key="live_coach_button"):
            raw_video_path = current_live_state.get("raw_video_path")
            if not raw_video_path:
                if current_live_state.get("raw_video_error"):
                    st.warning(f"Full-video coaching is unavailable: {current_live_state['raw_video_error']}")
                else:
                    st.info("The full-video coach needs a recorded local replay. Turn on recording before starting the next live session.")
            else:
                with st.spinner("Rechecking the recorded set locally, then reviewing it with Qwen3-VL on your GPU…"):
                    try:
                        with open(raw_video_path, "rb") as recorded_video:
                            live_analysis = analyze_video(
                                recorded_video.read(), suffix=os.path.splitext(raw_video_path)[1] or ".mp4"
                            )
                        st.session_state.live_coach = generate_video_coaching(
                            live_analysis.coach_facts(), raw_video_path, frames=live_analysis.coach_frames
                        )
                    except (ValueError, RuntimeError, TimeoutError, FileNotFoundError) as exc:
                        st.error(f"Could not get a local video coaching response: {exc}")
                    except Exception as exc:
                        st.error(f"Local video coaching failed: {exc}")
                    finally:
                        live_monitor.cleanup_recorded_video()
        live_coach = st.session_state.get("live_coach")
        if live_coach:
            if live_coach.get("review_status") == "withheld":
                st.warning(live_coach["reasoning"])
            else:
                st.markdown(f"#### {live_coach['headline']}")
                st.write(live_coach["reasoning"])
                st.info(f"**Try this next set:** {live_coach['primary_cue']}")
                if live_coach.get("drill"):
                    st.write(f"**Form drill:** {live_coach['drill']}")
                for observation in live_coach["observations"]:
                    st.write(f"**{observation['area'].replace('_', ' ').title()}:** {observation['observation']}")
                if live_coach["depth_note"]:
                    st.write(f"**Depth note:** {live_coach['depth_note']}")
                if live_coach["next_set"] and live_coach["next_set"] != live_coach["primary_cue"]:
                    st.write(f"**Next set:** {live_coach['next_set']}")
            st.caption(live_coach["limitation"])

if not current_live_state["running"]:
    if current_live_state["tracked_video"]:
        st.subheader("Completed live set")
        st.video(current_live_state["tracked_video"], format="video/mp4")
        st.download_button(
            "Download tracked squat video",
            data=current_live_state["tracked_video"],
            file_name="liftlens_live_squat_tracked.mp4",
            mime="video/mp4",
            key="download_live_tracked_video",
        )
        st.caption(
            f"Annotated live session · approximately {current_live_state['recorded_duration_s']:.1f} seconds. "
            "Pose lines and rep count are burned into the video."
        )
    elif current_live_state["recording_error"]:
        st.info(f"Tracked video unavailable: {current_live_state['recording_error']}")

st.caption("This single-person 2D view estimates squat depth and visible movement. It cannot diagnose pain or injury, and some form details may be hidden by the camera angle or clothing. Stop if you feel pain or dizziness; the camera is not a safety monitor.")
st.divider()

with st.expander("How LiftLens uses AI", expanded=False):
    st.markdown("""
    1. **MediaPipe Pose** estimates body landmarks in the video.
    2. **LiftLens rules** estimate rep count and provide a separate overall depth result.
    3. **Qwen3-VL 2B through llama.cpp** reviews the local video as a time-ordered sequence sampled at 0.5 frames per second, plus raw bottom and early-ascent frames from representative reps. It receives rep timing and a conservative whole-body ascent timing check, never separate left/right proxy measurements. It uses the GTX 1650 when available. The video stays on this computer; no cloud AI model API is used. Live pose tracking runs separately; a recorded live set is rechecked locally before video coaching.

    This is a training aid, not an official referee or medical tool. The model can still miss a fault or describe an unclear frame incorrectly; confidence and evidence are shown for review.

    **MediaPipe privacy note:** MediaPipe Tasks may send API performance and usage metrics to Google. Its privacy notice says input images/video are processed on-device and are not sent to Google. [Read the notice](https://github.com/google-ai-edge/mediapipe#privacy-notice).
    """)

uploaded = st.file_uploader(
    "Choose a squat video",
    type=["mp4", "mov", "avi", "m4v"],
    key="video_uploader",
    on_change=clear_previous_result,
)

if uploaded:
    left, right = st.columns([1.1, 1])
    with left:
        st.video(uploaded)
    with right:
        st.info("For best results: one lifter, full body visible, fixed side view, one complete set.")
        st.caption("Video is analyzed locally. On Windows, a temporary local copy is deleted after decoding.")
        if st.button("Analyze squat", type="primary", use_container_width=True, disabled=live_monitor.is_running()):
            with st.spinner("Tracking pose locally and measuring the squat…"):
                try:
                    suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ".mp4"
                    st.session_state.analysis = analyze_video(uploaded.getvalue(), suffix=suffix)
                    st.session_state.pop("coach", None)
                except Exception as exc:
                    st.error(str(exc))

analysis = st.session_state.get("analysis")
if analysis:
    st.divider()
    st.header(analysis.verdict)
    st.write(analysis.reason)
    evidence_col, details_col = st.columns([1.15, 1])
    with evidence_col:
        rgb_frame = cv2.cvtColor(analysis.bottom_frame, cv2.COLOR_BGR2RGB)
        st.image(Image.fromarray(rgb_frame),
                 caption=f"Apparent bottom · {analysis.bottom_timestamp_s:.2f}s · left: blue/cyan · right: red/orange",
                 use_container_width=True)
    with details_col:
        st.markdown("#### What the vision model found")
        st.write(analysis.evidence)
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Estimated reps", analysis.rep_count)
        metric_b.metric("Landmark visibility", f"{analysis.confidence:.0%}")
        if analysis.rep_bottom_times_s:
            times = ", ".join(f"{time_s:.2f}s" for time_s in analysis.rep_bottom_times_s)
            st.caption(f"Detected bottom frames: {times}")
        else:
            st.caption("No distinct squat bottoms found in the hip-motion trace.")
        st.markdown("**Rule-based note**")
        for cue in analysis.cues:
            st.write(f"• {cue}")
        st.caption(f"Analyzed {analysis.frames_analyzed} sampled frames. This prototype checks squat depth only; it does not verify every judging rule.")

    st.subheader("Full-clip pose tracking")
    tracking_video = getattr(analysis, "tracking_video", None)
    if tracking_video:
        st.video(tracking_video, format="video/mp4")
        st.download_button(
            "Download MediaPipe tracking video",
            data=tracking_video,
            file_name="liftlens_mediapipe_tracking.mp4",
            mime="video/mp4",
            key="download_mediapipe_tracking",
        )
        st.caption("The pose model updates on sampled frames; the latest detected pose is drawn over the frames between updates. The export has no audio.")
    else:
        export_error = getattr(analysis, "tracking_video_error", None)
        message = "LiftLens could not create the annotated video for this run."
        if export_error:
            message += f" Reason: {export_error}"
        st.info(message + " The still-frame analysis is still available.")

    st.divider()
    if analysis.rep_count == 0:
        st.subheader("Rep tracking unavailable")
        st.warning("LiftLens could not find distinct squat reps in this video. That does not mean you performed zero reps.")
        st.write("Try a fixed side view with your full body, especially hips, knees, and feet, unobstructed. Analyze again after changing the camera position.")
        st.caption("The vision model has no reliable rep measurements; Qwen can still help troubleshoot the recording setup.")
    elif any(item["status"].startswith("REVIEW") for item in (analysis.rep_measurements or [])):
        st.subheader("Some reps need visual review")
        st.warning("LiftLens keeps unclear depth estimates in review instead of forcing a pass/fail.")
        st.write("The coach reviews set-up, descent, bottom position, and ascent separately from the depth estimate.")
        st.caption("Use this depth estimate as a training aid, not an official judging decision.")

    if "coach" not in st.session_state:
        quick_pose_coach = coach_module._movement_signal_coach(
            analysis.coach_facts().get("movement_signals", [])
        )
        if quick_pose_coach:
            st.session_state.coach = quick_pose_coach

    st.subheader("Ask your local AI coach")
    st.write("Qwen3-VL reviews the whole uploaded set as a time-ordered video with raw frames from the bottom and early ascent of representative reps. It reviews set-up, descent, bottom, and ascent; the depth estimate is separate. Processing stays on this computer.")
    st.caption("The coach receives no separate left/right joint-proxy measurements. If it verifies a clear fault—or finds a possible movement pattern worth practicing—it can suggest one targeted drill and label any uncertainty. Otherwise it won't invent one.")
    if st.button("Review video with local Qwen3-VL coach", type="secondary", key="qwen_coach_button"):
        suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ".mp4"
        if suffix not in (".mp4", ".mov", ".avi", ".m4v"):
            suffix = ".mp4"
        with st.spinner("Qwen3-VL is reviewing the complete clip locally on your GPU. This can take a few minutes…"):
            try:
                local_video = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                local_video_path = local_video.name
                try:
                    local_video.write(uploaded.getvalue())
                    local_video.close()
                    st.session_state.coach = generate_video_coaching(
                        analysis.coach_facts(), local_video_path, frames=analysis.coach_frames
                    )
                finally:
                    if not local_video.closed:
                        local_video.close()
                    try:
                        os.unlink(local_video_path)
                    except OSError:
                        pass
            except (ValueError, RuntimeError, TimeoutError, FileNotFoundError) as exc:
                pose_coach = coach_module._movement_signal_coach(
                    analysis.coach_facts().get("movement_signals", [])
                )
                if pose_coach:
                    st.session_state.coach = pose_coach
                    st.warning(
                        "The local video model could not finish this review, so this is a cautious pose-based coaching note—not a Qwen video review."
                    )
                else:
                    st.error(f"Could not get a local video coaching response: {exc}")
            except Exception as exc:
                st.error(f"Local video coaching failed: {exc}")

    coach = st.session_state.get("coach")
    if coach:
        if coach.get("review_source") == "pose_signal":
            st.caption("Quick pose-based note; the full-video Qwen review has not run or did not finish.")
        if coach.get("review_status") == "withheld":
            st.warning(coach["reasoning"])
        else:
            st.markdown(f"### {coach['headline']}")
            st.write(coach["reasoning"])
            st.info(f"**Try this next set:** {coach['primary_cue']}")
            if coach.get("drill"):
                st.write(f"**Form drill:** {coach['drill']}")
            for observation in coach["observations"]:
                st.write(f"**{observation['area'].replace('_', ' ').title()}:** {observation['observation']}")
            if coach["depth_note"]:
                st.caption(coach["depth_note"])
            if coach["next_set"] and coach["next_set"] != coach["primary_cue"]:
                st.write(f"**For the next set:** {coach['next_set']}")
        st.caption(coach["limitation"])

st.divider()
st.caption("Pose and AI inference run locally. Uploaded and recorded video stays on this computer. MediaPipe Tasks may separately send performance/usage metrics to Google; it does not send the images/video. See ‘How LiftLens uses AI’ above.")
