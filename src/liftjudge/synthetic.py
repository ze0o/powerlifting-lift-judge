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
# half the distance between left and right joints seen from the front (pixels)
HALF_WIDTH = {"shoulder": 90.0, "elbow": 100.0, "wrist": 95.0, "hip": 55.0, "knee": 60.0, "ankle": 65.0}
VIEW_SPREAD = {"side": 0.0, "angled": 0.6, "front": 1.0}


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
               drop_frac: float = 0.0, view: str = "side", seed: int = 0) -> PoseSequence:
    """drop_frac: fraction of frames where the detector misses the lifter (NaN, conf 0).
    view: "side", "angled" (~45°) or "front": how far apart left and right joints appear."""
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
            if j != "nose":
                other = f"{'right' if side == 'left' else 'left'}_{j}"
                spread = VIEW_SPREAD[view] * HALF_WIDTH[j]
                if spread:  # left and right apart, both clearly visible
                    xy[t, KP[name]] = pt + np.array([-spread, 0]) + rng.normal(0, noise_px, 2)
                    xy[t, KP[other]] = pt + np.array([spread, 0]) + rng.normal(0, noise_px, 2)
                    conf[t, KP[other]] = 0.9
                else:  # side view: far side hidden behind the near side
                    xy[t, KP[other]] = pt + rng.normal(0, noise_px * 3, 2)
                    conf[t, KP[other]] = 0.5
    if drop_frac:
        miss = rng.random(T) < drop_frac
        xy[miss], conf[miss] = np.nan, 0.0
    return PoseSequence(xy=xy, conf=conf, fps=fps, width=1280, height=1000)


# --------------------------------------------------------------------------
# deadlift
# --------------------------------------------------------------------------
ARM = 290.0  # shoulder to wrist, arms hanging straight


def deadlift_pose_at(s: float, ankle=ANKLE, lean_top: float = 0.0, knee_top: float = 0.0,
                     rebend: float = 0.0) -> dict[str, np.ndarray]:
    """Side view, lifter facing +x. s = 0 bar on the floor, s = 1 lockout.

    lean_top / knee_top: degrees of torso lean / knee bend left at lockout (faults).
    rebend: extra knee bend in degrees at this moment (hitching: knees go forward
    under the bar so it can rest on the thighs).
    """
    shin = np.radians(25.0 * (1 - s) + knee_top * 0.4 * s + rebend * 0.5)   # shin forward lean
    thigh = np.radians(45.0 * (1 - s) + knee_top * 0.6 * s + rebend * 0.5)  # thigh tilt back from vertical
    torso = np.radians(60.0 * (1 - s) + lean_top * s)                  # torso lean forward
    knee = ankle + SHIN * np.array([np.sin(shin), -np.cos(shin)])
    hip = knee + THIGH * np.array([-np.sin(thigh), -np.cos(thigh)])
    shoulder = hip + TORSO * np.array([np.sin(torso), -np.cos(torso)])
    wrist = shoulder + np.array([0.0, ARM])
    elbow = shoulder + np.array([0.0, ARM / 2])
    return dict(hip=hip, knee=knee, ankle=ankle, shoulder=shoulder, elbow=elbow, wrist=wrist,
                nose=shoulder + np.array([40.0, -60.0]))


def deadlift_progress(fps: float = 30.0, reps: int = 1, dip: float = 0.0, hitch: bool = False,
                      tempo: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """s(t) and knee re-bend(t): pull, hold at lockout, lower, rest on the floor.

    dip    downward movement of the bar half way up (progress units)
    hitch  stall at the thighs while the knees re-bend 18° and straighten again
    """
    fps = fps / tempo
    s_parts = [np.zeros(int(0.6 * fps))]
    for _ in range(reps):
        if dip:
            s_parts += [ramp(0, 0.5, 0.8, fps), ramp(0.5, 0.5 - dip, 0.25, fps), ramp(0.5 - dip, 1, 0.8, fps)]
        elif hitch:
            s_parts += [ramp(0, 0.75, 1.0, fps), np.full(int(0.7 * fps), 0.75), ramp(0.75, 1, 0.5, fps)]
        else:
            s_parts.append(ramp(0, 1, 1.4, fps))
        s_parts += [np.ones(int(1.0 * fps)), ramp(1, 0, 1.0, fps), np.zeros(int(0.8 * fps))]
    s = np.concatenate(s_parts)
    bend = np.zeros_like(s)
    if hitch:  # place each stall where s sits at 0.75 before lockout
        flat = np.flatnonzero(np.isclose(s, 0.75))
        runs = np.split(flat, np.flatnonzero(np.diff(flat) > 1) + 1)
        for run in runs:
            if len(run) > 3:
                n = len(run)
                bend[run] = np.concatenate([np.linspace(0, 18, n // 2, endpoint=False), np.linspace(18, 0, n - n // 2)])
    return s, bend


def make_deadlift(fps: float = 30.0, reps: int = 1, dip: float = 0.0, hitch: bool = False,
                  lean_top: float = 0.0, knee_top: float = 0.0, step_px: float = 0.0,
                  noise_px: float = 1.5, side: str = "left", tempo: float = 1.0, seed: int = 0) -> PoseSequence:
    """Synthetic side-view deadlift. lean_top > ~15 = not erect; knee_top > ~15 = knees not locked."""
    rng = np.random.default_rng(seed)
    s, bend = deadlift_progress(fps, reps, dip, hitch, tempo)
    T = len(s)
    xy = np.full((T, len(KEYPOINTS), 2), np.nan)
    conf = np.zeros((T, len(KEYPOINTS)))
    for t in range(T):
        ankle = ANKLE + np.array([step_px if t > T // 2 else 0.0, 0.0])
        p = deadlift_pose_at(s[t], ankle, lean_top, knee_top, bend[t])
        for j, pt in p.items():
            name = j if j == "nose" else f"{side}_{j}"
            xy[t, KP[name]] = pt + rng.normal(0, noise_px, 2)
            conf[t, KP[name]] = 0.9
            if j != "nose":
                other = f"{'right' if side == 'left' else 'left'}_{j}"
                xy[t, KP[other]] = pt + rng.normal(0, noise_px * 3, 2)
                conf[t, KP[other]] = 0.5
    return PoseSequence(xy=xy, conf=conf, fps=fps, width=1280, height=1000)



def sample_set(lift: str = "squat") -> PoseSequence:
    """Demo 3-rep sets: squat (good, high, double bounce); deadlift (good, soft lockout, hitch)."""
    if lift == "deadlift":
        parts = [make_deadlift(seed=1), make_deadlift(lean_top=20, seed=2), make_deadlift(hitch=True, seed=3)]
    else:
        parts = [make_squat(depth_deg=d, bounce=b, seed=s) for d, b, s in ((14, False, 1), (-6, False, 2), (12, True, 3))]
    return PoseSequence(xy=np.concatenate([p.xy for p in parts]), conf=np.concatenate([p.conf for p in parts]),
                        fps=parts[0].fps, width=parts[0].width, height=parts[0].height)
