from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from liftjudge.synthetic import make_squat

ROOT = Path(__file__).resolve().parents[1]


def _standin_video(folder: Path, name: str, **kw) -> Path:
    """Grey frames + cached keypoints: tests everything after pose detection."""
    cv2 = pytest.importorskip("cv2")
    pose = make_squat(**kw)
    path = folder / f"{name}.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), pose.fps, (pose.width, pose.height))
    for _ in range(len(pose)):
        w.write(np.full((pose.height, pose.width, 3), 70, np.uint8))
    w.release()
    pose.save_csv(folder / f"{name}_keypoints.csv")
    return path


def test_draft_labels_prefills_clear_calls_and_leaves_close_ones(tmp_path):
    from liftjudge.cli import main

    raw = tmp_path / "raw"
    raw.mkdir()
    _standin_video(raw, "deep", depth_deg=12, reps=2, seed=1)
    _standin_video(raw, "close", depth_deg=3, reps=1, seed=2)
    _standin_video(raw, "high", depth_deg=-8, reps=1, seed=3)

    out = tmp_path / "draft.csv"
    assert main(["label", str(raw), "--out", str(out), "--snapshots", str(tmp_path / "snaps")]) == 0
    df = pd.read_csv(out)
    assert len(df) == 4
    assert (df.loc[df.video_id == "deep", "good_lift"] == 1).all()
    assert df.loc[df.video_id == "high", "depth"].item() == 1
    close = df[df.video_id == "close"].iloc[0]
    assert pd.isna(close.good_lift) and pd.isna(close.depth_ok)   # a human decides
    assert close.notes.startswith("CHECK")
    assert len(list((tmp_path / "snaps").glob("*.jpg"))) == 4


def test_annotated_video_is_browser_playable(tmp_path):
    from liftjudge import analyze_pose
    from liftjudge.keypoints import PoseSequence
    from liftjudge.render import ffmpeg_exe, render_video

    if not ffmpeg_exe():
        pytest.skip("no ffmpeg")
    import subprocess
    video = _standin_video(tmp_path, "v", depth_deg=12, seed=4)
    pose = PoseSequence.load_csv(tmp_path / "v_keypoints.csv")
    res = analyze_pose(pose)
    out = render_video(video, tmp_path / "judged.mp4", pose, res.signals, res.verdicts)
    probe = subprocess.run([ffmpeg_exe(), "-i", str(out)], capture_output=True, text=True).stderr
    assert "h264" in probe
    assert not (tmp_path / "judged_raw.mp4").exists()


def test_streamlit_app_sample_mode():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {"Reps": "3", "Good lifts": "1", "No lifts": "2", "Borderline": "0"}
