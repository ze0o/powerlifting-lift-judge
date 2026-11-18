"""End-to-end run of the real pose model on a short clip.

Slow (downloads yolo26n-pose.pt on first run), so it only runs when asked:
    LIFTJUDGE_E2E=1 pytest tests/test_video_pipeline.py
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("LIFTJUDGE_E2E"), reason="set LIFTJUDGE_E2E=1 to run")


def test_cli_on_sample_clip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    ultralytics = pytest.importorskip("ultralytics")
    from liftjudge.cli import main
    from liftjudge.keypoints import PoseSequence

    img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    h, w = img.shape[:2]
    video = tmp_path / "clip.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for t in range(30):
        m = np.float32([[1, 0, 0], [0, 1, int(20 * np.sin(t / 5))]])
        out.write(cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE))
    out.release()

    assert main(["analyze", str(video), "--out", str(tmp_path / "res")]) == 0
    pose = PoseSequence.load_csv(tmp_path / "res" / "clip_keypoints.csv")
    assert len(pose) == 30
    assert (pose.conf.sum(1) > 0).all(), "the same person should be tracked in every frame"
    assert (tmp_path / "res" / "clip_judged.mp4").exists()
