"""Synthetic side-view squat keypoints, for tests and demos.

A 2-segment leg (shin + thigh) driven by a progress value s in [0, 1]:
s = 0 standing tall, s = 1 the bottom of the squat. At the bottom the thigh is
`depth_deg` degrees below horizontal (negative = above parallel).
"""

from __future__ import annotations

import numpy as np

from .keypoints import KP, KEYPOINTS, PoseSequence

THIGH = 200.0
SHIN = 200.0
TORSO = 260.0
ANKLE = np.array([600.0, 900.0])


def pose_at(s: float, depth_deg: float, ankle=ANKLE) -> dict[str, np.ndarray]:
    alpha = np.radians(35.0 * s)                    # shin leans forward
    gamma = np.radians(90.0 - (90.0 + depth_deg) * s)  # thigh angle from horizontal (backward)
    knee = ankle + SHIN * np.array([np.sin(alpha), -np.cos(alpha)])
    hip = knee + THIGH * np.array([-np.cos(gamma), -np.sin(gamma)])
    lean = np.radians(45.0 * s)                     # torso leans forward
    shoulder = hip + TORSO * np.array([np.sin(lean), -np.cos(lean)])
    elbow = shoulder + np.array([-40.0, 60.0])
    wrist = shoulder + np.array([-10.0, 10.0])
    return dict(hip=hip, knee=knee, ankle=ankle, shoulder=shoulder, elbow=elbow, wrist=wrist,
                nose=shoulder + np.array([30.0, -70.0]))


def ramp(a: float, b: float, seconds: float, fps: float) -> np.ndarray:
    n = max(2, int(seconds * fps))
    w = (1 - np.cos(np.linspace(0, np.pi, n))) / 2  # smooth ease in/out
    return a + (b - a) * w


def squat_progress(fps: float = 30.0, reps: int = 1, bounce: bool = False,
                   ascent_dip: float = 0.0, finish_s: float = 0.0, tempo: float = 1.0) -> np.ndarray:
    """Progress curve s(t) for a set of reps.

    tempo       speed multiplier (2.0 = twice as fast)
    bounce      rise out of the hole and drop back in before standing up
    ascent_dip  size of a downward movement half way up (in progress units)
    finish_s    progress where the lifter stops at the top (0 = fully locked)
    """
    fps = fps / tempo  # only used for segment lengths
    parts = [np.zeros(int(0.8 * fps))]
    for r in range(reps):
        parts.append(ramp(0, 1, 1.0, fps))
        if bounce:
            parts += [ramp(1, 0.8, 0.2, fps), ramp(0.8, 1.02, 0.2, fps)]
        else:
            parts.append(np.ones(int(0.1 * fps)))
        top = finish_s if r == reps - 1 else 0.0
        bottom = 1.02 if bounce else 1.0
        if ascent_dip:
            parts += [ramp(bottom, 0.5, 0.6, fps), ramp(0.5, 0.5 + ascent_dip, 0.25, fps),
                      ramp(0.5 + ascent_dip, top, 0.6, fps)]
        else:
            parts.append(ramp(bottom, top, 1.2, fps))
        parts.append(np.full(int(0.8 * fps), top))
    return np.concatenate(parts)


def make_squat(depth_deg: float = 15.0, fps: float = 30.0, reps: int = 1, bounce: bool = False,
               ascent_dip: float = 0.0, finish_s: float = 0.0, step_px: float = 0.0,
               noise_px: float = 1.5, side: str = "left", tempo: float = 1.0,
               drop_frac: float = 0.0, seed: int = 0) -> PoseSequence:
    """drop_frac: fraction of frames where the detector misses the lifter (NaN, conf 0)."""
    rng = np.random.default_rng(seed)
    s = squat_progress(fps, reps, bounce, ascent_dip, finish_s, tempo)
    T = len(s)
    xy = np.full((T, len(KEYPOINTS), 2), np.nan)
    conf = np.zeros((T, len(KEYPOINTS)))
    for t in range(T):
        ankle = ANKLE + np.array([step_px if t > T // 2 else 0.0, 0.0])
        p = pose_at(s[t], depth_deg, ankle)
        for j, pt in p.items():
            name = j if j == "nose" else f"{side}_{j}"
            xy[t, KP[name]] = pt + rng.normal(0, noise_px, 2)
            conf[t, KP[name]] = 0.9
            if j != "nose":  # far side: same point, lower confidence
                other = f"{'right' if side == 'left' else 'left'}_{j}"
                xy[t, KP[other]] = pt + rng.normal(0, noise_px * 3, 2)
                conf[t, KP[other]] = 0.5
    if drop_frac:
        miss = rng.random(T) < drop_frac
        xy[miss], conf[miss] = np.nan, 0.0
    return PoseSequence(xy=xy, conf=conf, fps=fps, width=1280, height=1000)
