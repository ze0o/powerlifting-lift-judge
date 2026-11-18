"""Rep-level feature table for the ML / calibration step.

Each row is one rep: every measured check value plus the coaching metrics.
Join it with your human labels (data/labels.csv) to train a classifier or to
calibrate the rule thresholds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .rules.base import RepVerdict

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def rep_features(v: RepVerdict, video_id: str = "") -> dict:
    row = {"video_id": video_id, "lift": v.lift, "rep": v.rep + 1,
           "rule_status": v.status.value, "rule_cards": "+".join(c.value for c in v.cards)}
    for c in v.checks:
        row[c.code] = c.value
        row[f"{c.code}_status"] = c.status.value
    row.update(v.metrics)
    return row


def verdicts_to_frame(verdicts: list[RepVerdict], video_id: str = "") -> pd.DataFrame:
    return pd.DataFrame([rep_features(v, video_id) for v in verdicts])


def build_dataset(folder: str | Path, lift: str = "squat", cache: bool = True) -> pd.DataFrame:
    """Analyse every video in `folder` and return one row per rep.

    Keypoints are cached next to each video as <name>_keypoints.csv, so the
    slow pose model runs only once per video.
    """
    from . import analyze_pose
    from .keypoints import PoseSequence, extract_pose

    rows = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        kp = path.with_name(f"{path.stem}_keypoints.csv")
        if cache and kp.exists():
            pose = PoseSequence.load_csv(kp)
        else:
            pose = extract_pose(path)
            if cache:
                pose.save_csv(kp)
        res = analyze_pose(pose, lift)
        rows.append(verdicts_to_frame(res.verdicts, path.stem))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
