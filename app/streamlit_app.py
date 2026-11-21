"""Web demo: upload a squat video, get the verdict for every rep.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from liftjudge import analyze_pose  # noqa: E402
from liftjudge.keypoints import PoseSequence, extract_pose  # noqa: E402
from liftjudge.rules import SquatConfig, Status  # noqa: E402

DL_MODEL = ROOT / "models" / "rep_fault_net.keras"
STATUS_ICON = {"pass": "✅", "fail": "❌", "borderline": "⚠️", "unknown": "❔"}
CARD_ICON = {"red": "🟥", "blue": "🟦", "yellow": "🟨"}

st.set_page_config(page_title="Powerlifting Lift Judge", page_icon="🏋️", layout="wide")


# ---------------------------------------------------------------- cached steps
@st.cache_data(show_spinner=False, max_entries=8)
def pose_from_video(data: bytes, suffix: str, model: str) -> tuple[str, dict]:
    """Run the pose model once per (video, model). Returns keypoints CSV text + meta."""
    with tempfile.TemporaryDirectory() as d:
        video = Path(d) / f"upload{suffix}"
        video.write_bytes(data)
        pose = extract_pose(video, model=model)
        csv = Path(d) / "k.csv"
        pose.save_csv(csv)
        return csv.read_text(), {}


def load_pose(csv_text: str) -> PoseSequence:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv_text)
    return PoseSequence.load_csv(f.name)


@st.cache_resource(show_spinner=False)
def dl_model():
    try:
        from liftjudge.dl import load_model
        return load_model(DL_MODEL) if DL_MODEL.exists() else None
    except Exception:  # Keras or a backend not installed
        return None


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    lift = st.selectbox("Lift", ["squat", "deadlift"], format_func=str.title,
                        help="Bench press is on the roadmap (needs bar tracking).")
    size = st.select_slider("Pose model", ["n", "s", "m", "l"], value="n",
                            help="Bigger = more precise keypoints, slower. 'n' is fine for a first look; "
                                 "use 's' or 'm' for close depth calls.")
    if lift != "squat":
        st.caption("Depth settings apply to the squat only.")
    offset = st.slider("Crease offset (thighs)", 0.0, 0.2, SquatConfig.crease_offset, 0.01,
                       help="How far below the knee joint the hip joint must go for the hip crease to pass "
                            "the top of the knee. Calibrate it on labelled videos (notebook 02).")
    band = st.slider("Borderline band (± thighs)", 0.0, 0.1, SquatConfig.borderline_band, 0.01)
    st.divider()
    st.markdown("**Recording tips**\n- Side view, camera at hip height\n- Camera square to the lifter\n"
                "- Whole body + bar in frame\n- Phone on a tripod\n\n"
                "Full guide: docs/filming_guide.md")

st.title("🏋️ Powerlifting Lift Judge")
st.caption("Judges every rep of a squat or deadlift video against the IPF rules and explains the call "
           "(plus a neural-network second opinion for squats). A training aid, not a replacement for referees.")

col_a, col_b = st.columns([3, 1])
upload = col_a.file_uploader(f"Side-view {lift} video", type=["mp4", "mov", "avi", "m4v", "mkv"])
sample = col_b.button("Try a sample set", help="Synthetic keypoints: 3 reps, one good and two with faults")

# ---------------------------------------------------------------- run
pose, video_path = None, None
if upload is not None:
    data = upload.getvalue()
    with st.spinner("Finding the lifter and tracking 17 body points... (first run downloads the model)"):
        csv_text, _ = pose_from_video(data, Path(upload.name).suffix, f"yolo26{size}-pose.pt")
    pose = load_pose(csv_text)
    video_path = Path(tempfile.gettempdir()) / f"lj_{hashlib.md5(data).hexdigest()[:10]}{Path(upload.name).suffix}"
    video_path.write_bytes(data)
    st.session_state.pop("sample", None)
elif sample or st.session_state.get("sample"):
    st.session_state["sample"] = True
    from liftjudge.synthetic import sample_set
    pose = sample_set(lift)
    st.info("Sample mode: synthetic keypoints, so there's no video. Upload your own set to see the annotated video.")

if pose is None:
    st.stop()

cfg = SquatConfig(crease_offset=offset, borderline_band=band)
try:
    res = analyze_pose(pose, lift, config=cfg)
except ValueError as e:
    st.error(f"Couldn't read the lifter's legs: {e} Is the whole body in frame, filmed from the side?")
    st.stop()

if res.view and res.view.kind != "side":
    (st.error if res.view.kind == "front" else st.warning)(f"📷 {res.view.message}")

if not res.verdicts:
    st.warning("No reps found. Check the camera angle: side view, whole body in frame for the whole set.")
    st.stop()

# ---------------------------------------------------------------- summary
statuses = [v.status for v in res.verdicts]
m = st.columns(4)
m[0].metric("Reps", len(statuses))
m[1].metric("Good lifts", sum(s == Status.PASS for s in statuses))
m[2].metric("No lifts", sum(s == Status.FAIL for s in statuses))
m[3].metric("Borderline", sum(s == Status.BORDERLINE for s in statuses))

dl = None
model = dl_model() if lift == "squat" else None
if model is not None:
    from liftjudge.dl import FAULT_TEXT, FAULTS, predict_faults
    dl = predict_faults(model, res)

left, right = st.columns([1, 1])
with left:
    st.subheader("Verdict per rep")
    for v in res.verdicts:
        cards = " ".join(CARD_ICON[c.value] for c in v.cards)
        with st.expander(f"Rep {v.rep + 1}: **{v.label}** {cards}", expanded=v.status != Status.PASS):
            for c in v.checks:
                st.markdown(f"{STATUS_ICON[c.status.value]} {c.message}")
            met = v.metrics
            if lift == "squat":
                st.caption(f"descent {met['descent_s']:.2f}s · ascent {met['ascent_s']:.2f}s · "
                           f"sticking speed {met['sticking_speed']:.2f} thighs/s · sway {met['shoulder_drift']:.2f} thighs")
            else:
                st.caption(f"pull {met['pull_s']:.2f}s · lockout knee {met['lockout_knee']:.0f}° · "
                           f"hip {met['lockout_hip']:.0f}° · bar drift {met['bar_drift']:.2f} thighs")
            if dl is not None:
                r = dl.iloc[v.rep]
                found = [f"{FAULT_TEXT[f]} ({r[f]:.0%})" for f in FAULTS if r[f] >= 0.5]
                agree = (r.dl_label == "NO LIFT") == (v.status == Status.FAIL) or v.status == Status.BORDERLINE
                st.markdown(f"🧠 **Neural network:** {r.dl_label}" + (f": {'; '.join(found)}" if found else "")
                            + ("" if agree else "  \n⚠️ *Disagrees with the rules: worth a human look.*"))

with right:
    if video_path is not None:
        st.subheader("Annotated video")
        from liftjudge.render import render_video
        out = video_path.with_name(video_path.stem + f"_judged_{lift}_{offset:.2f}_{band:.2f}.mp4")
        if not out.exists():
            with st.spinner("Drawing the skeleton and verdicts..."):
                render_video(video_path, out, res.pose, res.signals, res.verdicts, cfg.crease_offset, lift=lift)
        st.video(str(out))

    sig = res.signals
    fig, ax = plt.subplots(figsize=(7, 3))
    if lift == "squat":
        st.subheader("Depth over the set")
        curve = sig.depth.to_numpy()
        ax.plot(sig.time_s, curve, lw=1.5, label="hip below knee (thighs)")
        ax.axhline(cfg.crease_offset, color="tab:orange", lw=1, label="depth line")
        ax.axhspan(cfg.crease_offset - cfg.borderline_band, cfg.crease_offset + cfg.borderline_band,
                   color="tab:orange", alpha=0.15)
    else:
        from liftjudge.rules.deadlift import bar_height
        st.subheader("Bar height over the set")
        curve = bar_height(sig)
        ax.plot(sig.time_s, curve, lw=1.5, label="bar above the feet (thighs)")
    for v in res.verdicts:
        col = {"pass": "tab:green", "fail": "tab:red"}.get(v.status.value, "tab:orange")
        ax.axvspan(sig.time_s[v.start], sig.time_s[v.end], color=col, alpha=0.08)
        ax.annotate(f"{v.rep + 1}", (sig.time_s[v.bottom], curve[v.bottom]), textcoords="offset points",
                    xytext=(0, 6), ha="center")
    ax.set_xlabel("time (s)"); ax.legend(loc="lower right", fontsize=8)
    st.pyplot(fig, clear_figure=True)

if dl is None and lift == "squat":
    st.caption("🧠 Neural-network second opinion is off: run `liftjudge train-dl` to create "
               "models/rep_fault_net.keras (needs Keras).")

st.divider()
d1, d2 = st.columns(2)
d1.download_button("Verdicts (JSON)", json.dumps([v.to_dict() for v in res.verdicts], indent=2),
                   "verdicts.json", "application/json")
d2.download_button("Per-frame signals (CSV)", res.signals.to_csv(index=False), "signals.csv", "text/csv")
