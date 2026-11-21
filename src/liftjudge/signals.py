"""Keypoints -> clean time-series signals used by the rules and the ML features.

All lengths are normalised by the lifter's thigh length (hip-to-knee distance),
so the numbers do not depend on camera distance or video resolution.
Image y points DOWN, so a larger y means lower in the frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .keypoints import PoseSequence

SIDE_JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


def choose_side(seq: PoseSequence) -> str:
    """Pick the body side facing the camera (higher mean confidence on the leg)."""
    def score(side: str) -> float:
        idx = [f"{side}_{j}" for j in ("hip", "knee", "ankle")]
        from .keypoints import KP
        return float(np.nanmean(seq.conf[:, [KP[n] for n in idx]]))
    return "left" if score("left") >= score("right") else "right"


def fill_and_smooth(x: np.ndarray, fps: float, window_s: float = 0.2) -> np.ndarray:
    """Interpolate gaps, then Savitzky-Golay smooth (keeps the bottom of the rep sharp)."""
    s = pd.Series(x, dtype=float)
    if s.notna().sum() < 2:
        return s.to_numpy()
    s = s.interpolate(limit_direction="both")
    win = max(5, int(round(window_s * fps)) | 1)  # odd, at least 5
    if len(s) <= win:
        return s.to_numpy()
    return savgol_filter(s.to_numpy(), win, polyorder=2)


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for arrays of points (T, 2)."""
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _joint(seq: PoseSequence, side: str, joint: str, min_conf: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """(T, 2) points and (T,) confidence of a joint; side "both" = confidence-weighted
    average of left and right (used for front views, where one side isn't 'nearer')."""
    from .keypoints import KP
    if side != "both":
        return seq.point(f"{side}_{joint}"), seq.conf[:, KP[f"{side}_{joint}"]]
    l, r = seq.point(f"left_{joint}"), seq.point(f"right_{joint}")
    cl, cr = seq.conf[:, KP[f"left_{joint}"]], seq.conf[:, KP[f"right_{joint}"]]
    wl = np.where((cl >= min_conf) & np.isfinite(l).all(1), cl, 0.0)
    wr = np.where((cr >= min_conf) & np.isfinite(r).all(1), cr, 0.0)
    tot = wl + wr
    pts = (np.nan_to_num(l) * wl[:, None] + np.nan_to_num(r) * wr[:, None]) / np.where(tot > 0, tot, 1)[:, None]
    pts[tot == 0] = np.nan
    return pts, np.maximum(cl, cr)


def body_signals(seq: PoseSequence, side: str | None = None,
                 smooth_window_s: float = 0.2) -> pd.DataFrame:
    """Per-frame signals for one side of the body ("left", "right", or "both").

    Columns
    -------
    time_s, <joint>_x, <joint>_y   smoothed pixel coordinates
    thigh_px                        robust thigh length in pixels (constant)
    depth                           (hip_y - knee_y) / thigh. > 0 means hip joint below knee joint
    hip_height                      (ankle_y - hip_y) / thigh. how high the hips are above the feet
    hip_vel                         d(hip_y)/dt in thighs per second. > 0 means hips moving DOWN
    knee_angle, hip_angle, elbow_angle   joint angles in degrees (180 = straight)
    leg_conf                        mean confidence of hip, knee, ankle
    """
    side = side or choose_side(seq)
    out = {"time_s": seq.t}
    pts, confs = {}, {}
    for j in SIDE_JOINTS:
        p, confs[j] = _joint(seq, side, j)
        sx = fill_and_smooth(p[:, 0], seq.fps, smooth_window_s)
        sy = fill_and_smooth(p[:, 1], seq.fps, smooth_window_s)
        pts[j] = np.stack([sx, sy], axis=1)
        out[f"{j}_x"], out[f"{j}_y"] = sx, sy

    thigh = np.nanmedian(np.linalg.norm(pts["hip"] - pts["knee"], axis=1))
    if not np.isfinite(thigh) or thigh <= 0:
        raise ValueError("Could not measure thigh length: hip/knee not detected.")
    out["thigh_px"] = np.full(len(seq), thigh)
    out["depth"] = (pts["hip"][:, 1] - pts["knee"][:, 1]) / thigh
    out["hip_height"] = (pts["ankle"][:, 1] - pts["hip"][:, 1]) / thigh
    out["hip_vel"] = np.gradient(pts["hip"][:, 1] / thigh) * seq.fps
    out["wrist_vel"] = np.gradient(pts["wrist"][:, 1] / thigh) * seq.fps
    out["knee_angle"] = angle(pts["hip"], pts["knee"], pts["ankle"])
    out["hip_angle"] = angle(pts["shoulder"], pts["hip"], pts["knee"])
    out["elbow_angle"] = angle(pts["shoulder"], pts["elbow"], pts["wrist"])
    out["leg_conf"] = np.mean([confs[j] for j in ("hip", "knee", "ankle")], axis=0)
    df = pd.DataFrame(out)
    df.attrs.update(side=side, fps=seq.fps)
    return df
