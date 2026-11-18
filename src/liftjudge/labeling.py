"""Make labelling fast: let the judge draft the labels, then a human checks them.

`draft_labels` runs every video in a folder and writes one row per rep with
the rule engine's verdict already filled in, the time of each rep (so you can
find it in the video) and a snapshot of the bottom position with the hip and
knee lines drawn. You then only have to check and correct, not label from zero.

Borderline / unclear calls are left EMPTY on purpose: those are the reps a
human must decide, and they're the most valuable labels for training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dl import FAULTS, _rule_faults
from .features import VIDEO_EXTS
from .keypoints import KP, SKELETON, PoseSequence
from .rules.base import Status

LABEL_COLUMNS = ["video_id", "rep", "lift", "judge", "start_s", "bottom_s", "end_s",
                 "depth_ok", "good_lift", "cards", *FAULTS, "camera", "rule_depth_margin", "notes"]


def _pose_for(video: Path, cache: bool = True, model: str | None = None) -> PoseSequence:
    from .keypoints import DEFAULT_MODEL, extract_pose
    kp = video.with_name(f"{video.stem}_keypoints.csv")
    if cache and kp.exists():
        return PoseSequence.load_csv(kp)
    pose = extract_pose(video, model=model or DEFAULT_MODEL)
    if cache:
        pose.save_csv(kp)
    return pose


def rep_rows(analysis, video_id: str) -> list[dict]:
    fps = analysis.pose.fps
    rows = []
    for v in analysis.verdicts:
        depth = next(c for c in v.checks if c.code == "squat_depth")
        unsure = {c.code for c in v.checks if c.status in (Status.BORDERLINE, Status.UNKNOWN)}
        faults = _rule_faults(v)
        row = {
            "video_id": video_id, "rep": v.rep + 1, "lift": v.lift, "judge": "auto",
            "start_s": round(v.start / fps, 2), "bottom_s": round(v.bottom / fps, 2), "end_s": round(v.end / fps, 2),
            "depth_ok": "" if "squat_depth" in unsure else int(depth.status == Status.PASS),
            "good_lift": "" if v.status == Status.BORDERLINE else int(v.status == Status.PASS),
            "cards": "+".join(c.value for c in v.cards),
            "camera": "side",
            "rule_depth_margin": round(float(depth.value), 3) if depth.value is not None else "",
            "notes": "CHECK: " + "; ".join(c.message for c in v.checks if c.code in unsure) if unsure else "",
        }
        for f in FAULTS:
            row[f] = "" if (f == "depth" and "squat_depth" in unsure) else int(f in faults)
        rows.append(row)
    return rows


def bottom_snapshot(video: Path, analysis, rep_index: int, out_path: Path, width: int = 480) -> Path | None:
    """Save the bottom frame of a rep, cropped to the lifter, with hip and knee lines."""
    import cv2

    v = analysis.verdicts[rep_index]
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, v.bottom)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    sig, side = analysis.signals, analysis.signals.attrs.get("side", "left")
    pts = analysis.pose.xy[v.bottom]
    for a, b in SKELETON:
        pa, pb = pts[KP[a]], pts[KP[b]]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(img, tuple(int(x) for x in pa), tuple(int(x) for x in pb), (255, 255, 255), 2, cv2.LINE_AA)
    hip = (sig["hip_x"].iloc[v.bottom], sig["hip_y"].iloc[v.bottom])
    knee = (sig["knee_x"].iloc[v.bottom], sig["knee_y"].iloc[v.bottom])
    thigh = float(sig["thigh_px"].iloc[0])
    for (x, y), col, name in ((hip, (0, 190, 255), "hip"), (knee, (80, 200, 80), "knee")):
        cv2.line(img, (int(x - thigh), int(y)), (int(x + thigh), int(y)), col, 2, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 6, col, -1, cv2.LINE_AA)
        cv2.putText(img, name, (int(x + thigh) - 45, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

    valid = pts[np.isfinite(pts).all(1)]
    if len(valid):
        pad = 0.6 * thigh
        x0, y0 = np.maximum(valid.min(0) - pad, 0).astype(int)
        x1, y1 = valid.max(0) + pad
        img = img[y0:int(min(y1, img.shape[0])), x0:int(min(x1, img.shape[1]))]
    scale = width / img.shape[1]
    img = cv2.resize(img, (width, int(img.shape[0] * scale)))
    depth = next(c for c in v.checks if c.code == "squat_depth")
    text = f"rep {v.rep + 1}  {depth.status.value}  margin {depth.value:+.2f}"
    cv2.rectangle(img, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path


def draft_labels(folder: str | Path, out_csv: str | Path, lift: str = "squat",
                 snapshots: str | Path | None = None, model: str | None = None) -> pd.DataFrame:
    """Analyse every video under `folder`; write a draft labels file (+ snapshots)."""
    from . import analyze_pose

    rows = []
    videos = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    for video in videos:
        res = analyze_pose(_pose_for(video, model=model), lift)
        rows += rep_rows(res, video.stem)
        if snapshots:
            for k in range(len(res.verdicts)):
                bottom_snapshot(video, res, k, Path(snapshots) / f"{video.stem}_rep{k + 1}.jpg")
        print(f"{video.name}: {len(res.verdicts)} reps")
    df = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
