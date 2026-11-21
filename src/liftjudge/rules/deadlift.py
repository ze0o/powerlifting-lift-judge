"""IPF deadlift rules turned into measurable checks.

Card groups follow the IPF Technical Rulebook (2026):
  RED    - knees not locked straight at the completion of the lift
         - not standing erect with the shoulders back (measured as the hip angle)
  BLUE   - any downward movement of the bar before the final position
         - supporting the bar on the thighs (hitching: knees re-bend under the bar)
  YELLOW - stepping backward/forward or moving the feet laterally
           (lowering before the "Down" signal / dropping the bar can't be judged
           without the referee's signal)

The bar is tracked through the wrists (arms hang straight in a deadlift).
Needs a SIDE view: from the front, knee and hip angles can't be measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..reps import Rep, segment_reps
from .base import Card, Check, RepVerdict, Status


@dataclass
class DeadliftConfig:
    lockout_knee_angle: float = 165.0   # degrees, 2-D side view
    erect_hip_angle: float = 165.0      # shoulder-hip-knee: standing tall with shoulders back
    downward_tol: float = 0.05          # thighs of bar drop allowed as noise on the way up
    hitch_rebend: float = 8.0           # degrees the knees may re-bend during the pull
    top_zone: float = 0.06              # bar within this of its highest point = "at lockout"
    feet_tol: float = 0.15              # ankle displacement (thighs) that counts as a step


def bar_height(sig: pd.DataFrame) -> np.ndarray:
    """Bar height above the feet in thigh lengths, from the wrists."""
    return ((sig["ankle_y"] - sig["wrist_y"]) / sig["thigh_px"]).to_numpy()


def judge_rep(sig: pd.DataFrame, rep: Rep, cfg: DeadliftConfig | None = None) -> RepVerdict:
    cfg = cfg or DeadliftConfig()
    fps = sig.attrs.get("fps", 30.0)
    view = sig.attrs.get("view", "side")
    bar = bar_height(sig)
    v = RepVerdict(lift="deadlift", rep=rep.index, start=rep.start, bottom=rep.bottom, end=rep.end)

    top = float(bar[rep.bottom])                       # `bottom` = the peak of the bar here
    seg = bar[rep.start: rep.end + 1]
    at_top = np.flatnonzero(seg >= top - cfg.top_zone) + rep.start
    lock_start = int(at_top[0]) if len(at_top) else rep.bottom
    hold = slice(lock_start, int(at_top[-1]) + 1 if len(at_top) else rep.bottom + 1)

    # ---- RED: knees locked, standing erect at lockout -----------------------
    knee = float(sig["knee_angle"].iloc[hold].max())
    hip = float(sig["hip_angle"].iloc[hold].max())
    if view == "front":
        for code, what in (("deadlift_knees_locked", "Knee lockout"), ("deadlift_erect", "Standing erect")):
            v.checks.append(Check(code, Card.RED, Status.UNKNOWN,
                                  f"{what} not judged: filmed from the front. Film from the side", lock_start))
    else:
        ok = knee >= cfg.lockout_knee_angle
        v.checks.append(Check(
            "deadlift_knees_locked", Card.RED, Status.PASS if ok else Status.FAIL,
            f"Knees locked at the top ({knee:.0f}°)" if ok else
            f"Knees not locked at the top: {knee:.0f}° < {cfg.lockout_knee_angle:.0f}°",
            lock_start, knee, cfg.lockout_knee_angle))
        ok = hip >= cfg.erect_hip_angle
        v.checks.append(Check(
            "deadlift_erect", Card.RED, Status.PASS if ok else Status.FAIL,
            f"Standing erect, shoulders back ({hip:.0f}°)" if ok else
            f"Not standing erect with shoulders back: hips at {hip:.0f}° < {cfg.erect_hip_angle:.0f}°",
            lock_start, hip, cfg.erect_hip_angle))

    # ---- BLUE: downward movement of the bar before lockout ------------------
    pull = bar[rep.start: lock_start + 1]
    drops = np.maximum.accumulate(pull) - pull if len(pull) else np.zeros(1)
    drop = float(drops.max())
    moved_down = drop > cfg.downward_tol
    v.checks.append(Check(
        "deadlift_downward_movement", Card.BLUE, Status.FAIL if moved_down else Status.PASS,
        f"Downward movement of the bar before lockout ({drop:.2f} thighs)" if moved_down
        else "Bar went up without dropping",
        rep.start + int(drops.argmax()) if moved_down else None, drop, cfg.downward_tol))

    # ---- BLUE: hitching (knees re-bend under the bar on the thighs) ---------
    if view == "front":
        v.checks.append(Check("deadlift_hitching", Card.BLUE, Status.UNKNOWN,
                              "Hitching not judged: filmed from the front", None))
    else:
        ka = sig["knee_angle"].to_numpy()[rep.start: lock_start + 1]
        bar_above_knee = pull > ((sig["ankle_y"] - sig["knee_y"]) / sig["thigh_px"]).to_numpy()[rep.start: lock_start + 1]
        rebend = np.where(bar_above_knee, np.maximum.accumulate(ka) - ka, 0.0) if len(ka) else np.zeros(1)
        worst = float(rebend.max())
        hitched = worst > cfg.hitch_rebend
        v.checks.append(Check(
            "deadlift_hitching", Card.BLUE, Status.FAIL if hitched else Status.PASS,
            f"Hitching: knees re-bent {worst:.0f}° with the bar on the thighs" if hitched
            else "No hitching", rep.start + int(rebend.argmax()) if hitched else None, worst, cfg.hitch_rebend))

    # ---- YELLOW: feet moved -------------------------------------------------
    ank = sig[["ankle_x", "ankle_y"]].iloc[rep.start: rep.end + 1].to_numpy()
    thigh = float(sig["thigh_px"].iloc[0])
    shift = float(np.linalg.norm(ank - ank[0], axis=1).max() / thigh) if len(ank) else 0.0
    stepped = shift > cfg.feet_tol
    v.checks.append(Check(
        "deadlift_feet", Card.YELLOW, Status.FAIL if stepped else Status.PASS,
        f"Feet moved {shift:.2f} thighs (step)" if stepped else "Feet stayed in place",
        None, shift, cfg.feet_tol))

    # ---- coaching metrics (not judged) --------------------------------------
    speed = np.gradient(bar) * fps
    lo, hi = bar[rep.start], top
    mid = np.flatnonzero((pull > lo + 0.1 * (hi - lo)) & (pull < hi - 0.1 * (hi - lo))) + rep.start
    wrist_x = sig["wrist_x"].iloc[rep.start: lock_start + 1].to_numpy() / thigh
    v.metrics.update(
        pull_s=(lock_start - rep.start) / fps,
        hold_s=(hold.stop - hold.start) / fps,
        lockout_knee=knee,
        lockout_hip=hip,
        bar_drift=float(wrist_x.max() - wrist_x.min()) if len(wrist_x) else float("nan"),  # thighs
        sticking_speed=float(speed[mid].min()) if len(mid) else float("nan"),          # thighs/s
        sticking_frame=int(mid[speed[mid].argmin()]) if len(mid) else -1,
    )
    return v


def judge_deadlift(sig: pd.DataFrame, cfg: DeadliftConfig | None = None) -> list[RepVerdict]:
    """Judge every rep in a deadlift video's signals (from `signals.body_signals`)."""
    if cfg is not None and not isinstance(cfg, DeadliftConfig):
        cfg = None   # a squat config was passed: use deadlift defaults
    reps = segment_reps(bar_height(sig), sig.attrs.get("fps", 30.0))
    return [judge_rep(sig, r, cfg) for r in reps]
