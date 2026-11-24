"""Development dataset: simulated lifters, squats and deadlifts, with the true faults known.

Real labelled lifting videos are still being collected, so the pipeline is developed and
evaluated on simulated sets first. Each simulated lifter has their own tempo, keypoint
noise, frame rate, filmed side and camera position, and some frames are dropped as if the
pose model lost them.

The LABELS come from what the simulator actually did (e.g. depth 2° above parallel, a
hitch), never from the rule engine, so comparing the rules with the labels is a fair test.

Two tables:
  reps    one row per rep: features the pipeline measures + target `good_lift`
  frames  one row per video frame of every rep: the signals over time (sequence data)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import analyze_pose
from .synthetic import make_deadlift, make_squat

TRUE_DEPTH_LINE_DEG = 3.0   # hidden referee depth line for simulated squats

FRAME_COLUMNS = ["depth", "hip_height", "bar_height", "knee_angle", "hip_angle", "elbow_angle",
                 "hip_vel", "ankle_dx", "leg_conf"]


def _lifters(n: int, rng: np.random.Generator) -> list[dict]:
    views = rng.choice(["side", "angled", "front"], size=n, p=[0.8, 0.12, 0.08])
    return [dict(id=f"L{i:03d}", tempo=rng.uniform(0.6, 1.8), noise=rng.uniform(0.8, 4.0),
                 side=str(rng.choice(["left", "right"])), fps=float(rng.choice([30, 30, 60])),
                 view=str(views[i]), drop=float(rng.choice([0, 0, 0.02, 0.06, 0.12])))
            for i in range(n)]


def _squat(L: dict, rng, seed: int):
    depth_deg = float(rng.uniform(-12, 18))
    f = {k: bool(rng.random() < p) for k, p in (("lockout", 0.10), ("bounce", 0.12), ("downward", 0.12), ("feet", 0.06))}
    pose = make_squat(depth_deg=depth_deg, fps=L["fps"], side=L["side"], seed=seed,
                      tempo=L["tempo"] * rng.uniform(0.85, 1.15), noise_px=L["noise"], drop_frac=L["drop"],
                      view=L["view"], bounce=f["bounce"],
                      ascent_dip=float(rng.uniform(0.08, 0.18)) if f["downward"] else 0.0,
                      finish_s=float(rng.uniform(0.22, 0.4)) if f["lockout"] else 0.0,
                      step_px=float(rng.uniform(40, 70)) if f["feet"] else 0.0)
    faults = [k for k, v in f.items() if v]
    if depth_deg <= TRUE_DEPTH_LINE_DEG:
        faults.insert(0, "depth")
    return pose, faults


def _deadlift(L: dict, rng, seed: int):
    f = {k: bool(rng.random() < p) for k, p in (("lockout", 0.16), ("downward", 0.10), ("hitching", 0.10), ("feet", 0.06))}
    lean = float(rng.uniform(16, 30)) if f["lockout"] and rng.random() < 0.6 else 0.0
    knee = float(rng.uniform(18, 28)) if f["lockout"] and lean == 0.0 else 0.0
    pose = make_deadlift(fps=L["fps"], side=L["side"], seed=seed, tempo=L["tempo"] * rng.uniform(0.85, 1.15),
                         noise_px=L["noise"], lean_top=lean, knee_top=knee,
                         dip=float(rng.uniform(0.08, 0.14)) if f["downward"] and not f["hitching"] else 0.0,
                         hitch=f["hitching"], step_px=float(rng.uniform(40, 70)) if f["feet"] else 0.0)
    if L["drop"]:  # detector misses
        miss = rng.random(len(pose)) < L["drop"]
        pose.xy[miss], pose.conf[miss] = np.nan, 0.0
    return pose, [k for k, v in f.items() if v]


def _value(v, code):
    c = next((c for c in v.checks if c.code == code), None)
    return float(c.value) if c is not None and c.value is not None else np.nan


def build_dev_dataset(n_squat: int = 1500, n_deadlift: int = 900, n_lifters: int = 60,
                      seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (reps, frames). About 2,400 reps and 300k frames with the defaults (~1-2 min)."""
    rng = np.random.default_rng(seed)
    lifters = _lifters(n_lifters, rng)
    rep_rows, frame_parts = [], []
    jobs = ["squat"] * n_squat + ["deadlift"] * n_deadlift
    rng.shuffle(jobs)
    i = 0
    for lift in jobs:
        for _attempt in range(5):  # the rep splitter can disagree on a broken clip: retry like a re-film
            i += 1
            L = lifters[int(rng.integers(n_lifters))]
            pose, faults = (_squat if lift == "squat" else _deadlift)(L, rng, i)
            res = analyze_pose(pose, lift)
            if len(res.verdicts) == 1:
                break
        else:
            continue
        v, sig = res.verdicts[0], res.signals
        rep_id = f"R{len(rep_rows):05d}"
        m = v.metrics
        raw_missing = float(np.isnan(pose.xy[:, 11:17, 0]).any(1).mean())
        rep_rows.append({
            "rep_id": rep_id, "lifter_id": L["id"], "lift": lift,
            "camera_view": res.view.kind, "filmed_side": L["side"], "fps": int(L["fps"]),
            "duration_s": round((v.end - v.start) / pose.fps, 3),
            "depth_margin": _value(v, "squat_depth"),
            "lockout_knee_angle": _value(v, "squat_lockout_finish") if lift == "squat" else _value(v, "deadlift_knees_locked"),
            "lockout_hip_angle": _value(v, "deadlift_erect"),
            "bounce": _value(v, "squat_bounce"),
            "downward_drop": _value(v, "squat_downward_movement") if lift == "squat" else _value(v, "deadlift_downward_movement"),
            "hitch_rebend_deg": _value(v, "deadlift_hitching"),
            "feet_shift": _value(v, "squat_feet") if lift == "squat" else _value(v, "deadlift_feet"),
            "down_phase_s": m.get("descent_s", np.nan),
            "up_phase_s": m.get("ascent_s", m.get("pull_s", np.nan)),
            "sticking_speed": m.get("sticking_speed", np.nan),
            "horizontal_drift": m.get("shoulder_drift", m.get("bar_drift", np.nan)),
            "keypoint_conf": float(sig["leg_conf"].iloc[v.start: v.end + 1].mean()),
            "missing_frames_pct": round(100 * raw_missing, 2),
            "rule_status": v.status.value,             # the rule engine's call (for comparison, not a feature)
            "fault_type": "none" if not faults else faults[0] if len(faults) == 1 else "multiple",
            "faults": "+".join(faults),
            "good_lift": int(not faults),               # TARGET: 1 = three white lights
        })
        # frames: the signals over the rep (+0.3 s either side)
        pad = int(0.3 * pose.fps)
        a, b = max(0, v.start - pad), min(len(sig) - 1, v.end + pad)
        w = sig.iloc[a: b + 1]
        thigh = float(sig["thigh_px"].iloc[0])
        raw_hip = pose.xy[a: b + 1, 11 if L["side"] == "left" else 12, 1]
        frame_parts.append(pd.DataFrame({
            "rep_id": rep_id, "lift": lift, "frame": np.arange(len(w)), "time_s": (w["time_s"] - w["time_s"].iloc[0]).round(3),
            "depth": w["depth"].round(4), "hip_height": w["hip_height"].round(4),
            "bar_height": ((w["ankle_y"] - w["wrist_y"]) / thigh).round(4),
            "knee_angle": w["knee_angle"].round(2), "hip_angle": w["hip_angle"].round(2),
            "elbow_angle": w["elbow_angle"].round(2), "hip_vel": w["hip_vel"].round(4),
            "ankle_dx": ((w["ankle_x"] - w["ankle_x"].iloc[0]) / thigh).round(4),
            "leg_conf": w["leg_conf"].round(3),
            "raw_hip_y": raw_hip,                       # before gap filling: NaN where the detector missed
            "good_lift": int(not faults),
        }))
    return pd.DataFrame(rep_rows), pd.concat(frame_parts, ignore_index=True)
