"""Which way is the camera looking at the lifter?

Measured on competition front-view footage: 0.52 standing, 0.73 in the hole.

From the SIDE the left and right shoulders (and hips) sit almost on top of each
other in the image. From the FRONT or BACK they are a full body-width apart.
So the ratio  body width / torso length  tells the camera angle:

    < 0.25     side view     -> all checks reliable
    0.25-0.45  angled (~45°) -> depth less reliable: wider borderline band
    > 0.45     front / back  -> depth can't be judged: hip and knee overlap
                                when the thighs point at the camera
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .keypoints import KP, PoseSequence

SIDE_MAX = 0.25
ANGLED_MAX = 0.45

MESSAGES = {
    "side": "Side view: good for judging.",
    "angled": "Angled view (about 45°): depth is less precise, so more reps will be borderline. "
              "Move the camera to the side of the lifter.",
    "front": "Front/back view: depth can't be judged from the front, because the hip and knee "
             "overlap when the thighs point at the camera. Film from the side.",
    "unknown": "Couldn't tell the camera angle: shoulders or hips not visible.",
}


@dataclass
class CameraView:
    kind: str        # side | angled | front | unknown
    ratio: float     # body width / torso length (median over the video)
    message: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "ratio": None if not np.isfinite(self.ratio) else round(self.ratio, 3),
                "message": self.message}


def width_ratio(seq: PoseSequence, min_conf: float = 0.3) -> np.ndarray:
    """Per-frame body width / torso length (NaN where the joints aren't seen)."""
    ls, rs = seq.xy[:, KP["left_shoulder"]], seq.xy[:, KP["right_shoulder"]]
    lh, rh = seq.xy[:, KP["left_hip"]], seq.xy[:, KP["right_hip"]]
    c = seq.conf[:, [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]]
    width = (np.linalg.norm(ls - rs, axis=1) + np.linalg.norm(lh - rh, axis=1)) / 2
    torso = np.linalg.norm((ls + rs) / 2 - (lh + rh) / 2, axis=1)
    r = width / np.where(torso > 1, torso, np.nan)
    r[(c < min_conf).any(1)] = np.nan
    return r


def detect_view(seq: PoseSequence) -> CameraView:
    r = width_ratio(seq)
    if np.isfinite(r).sum() < 3:
        return CameraView("unknown", float("nan"), MESSAGES["unknown"])
    med = float(np.nanmedian(r))
    kind = "side" if med < SIDE_MAX else "angled" if med < ANGLED_MAX else "front"
    return CameraView(kind, med, MESSAGES[kind])
