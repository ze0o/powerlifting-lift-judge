"""IPF squat rules turned into measurable checks.

Card groups follow the IPF Technical Rulebook (2026):
  RED    - not lowering until the top surface of the legs at the hip joint is
           below the top of the knees (depth)
  BLUE   - not upright with knees locked at the start / finish; double bouncing
           or more than one recovery attempt; any downward movement during the ascent
  YELLOW - stepping backward/forward or moving the feet laterally
           (commands from the Chief Referee cannot be judged from video)

Needs a SIDE view, camera roughly at hip height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..reps import Rep, segment_reps
from .base import Card, Check, RepVerdict, Status


@dataclass
class SquatConfig:
    # A pose model finds the CENTRE of the hip and knee joints. The rule compares
    # the TOP SURFACE of the thigh at the hip crease with the TOP of the knee.
    # The thigh is thicker at the hip than at the knee, so the hip joint centre
    # has to sit a bit lower than the knee joint centre for the surfaces to pass.
    # This offset (in thigh lengths, ~0.07 = ~3 cm) must be calibrated on
    # labelled videos - see notebooks/02.
    crease_offset: float = 0.07
    # +/- band around the line where we refuse to call it (keypoint noise).
    borderline_band: float = 0.04
    lockout_knee_angle: float = 160.0   # degrees, 2-D side view
    downward_tol: float = 0.04          # thigh lengths of hip drop allowed as noise on the way up
    bounce_tol: float = 0.08            # rise-then-drop near the bottom that counts as a bounce
    bottom_zone: float = 0.35           # how far above the bottom (thighs) counts as "the hole"
    feet_tol: float = 0.15              # ankle displacement (thighs) that counts as a step
    lockout_window_s: float = 0.3
    min_leg_conf: float = 0.4


def _lockout_check(sig: pd.DataFrame, frames: slice, cfg: SquatConfig, where: str) -> Check:
    ka = sig["knee_angle"].iloc[frames]
    best = float(ka.max()) if len(ka) else float("nan")
    ok = np.isfinite(best) and best >= cfg.lockout_knee_angle
    return Check(
        code=f"squat_lockout_{where}", card=Card.BLUE,
        status=Status.PASS if ok else Status.FAIL,
        message=(f"Knees locked at the {where} ({best:.0f}°)" if ok else
                 f"Knees not locked at the {where}: {best:.0f}° < {cfg.lockout_knee_angle:.0f}°"),
        frame=int(sig["knee_angle"].iloc[frames].idxmax()) if len(ka) else None,
        value=best, threshold=cfg.lockout_knee_angle,
    )


def judge_rep(sig: pd.DataFrame, rep: Rep, cfg: SquatConfig | None = None) -> RepVerdict:
    cfg = cfg or SquatConfig()
    fps = sig.attrs.get("fps", 30.0)
    depth = sig["depth"].to_numpy()
    v = RepVerdict(lift="squat", rep=rep.index, start=rep.start, bottom=rep.bottom, end=rep.end)

    # ---- RED: depth -------------------------------------------------------
    max_depth = float(depth[rep.bottom])
    margin = max_depth - cfg.crease_offset
    conf = float(sig["leg_conf"].iloc[max(0, rep.bottom - 2): rep.bottom + 3].mean())
    if conf < cfg.min_leg_conf:
        st, msg = Status.UNKNOWN, f"Depth unclear: legs hard to see at the bottom (confidence {conf:.2f})"
    elif margin > cfg.borderline_band:
        st, msg = Status.PASS, f"Depth: hip crease below the knee (+{margin:.2f} thighs)"
    elif margin < -cfg.borderline_band:
        st, msg = Status.FAIL, f"High: hip crease did not go below the top of the knee ({margin:+.2f} thighs)"
    else:
        st, msg = Status.BORDERLINE, f"Depth borderline ({margin:+.2f} thighs): too close to call from video"
    v.checks.append(Check("squat_depth", Card.RED, st, msg, rep.bottom, margin, 0.0))

    # ---- BLUE: knees locked at start and finish ----------------------------
    w = max(1, int(cfg.lockout_window_s * fps))
    v.checks.append(_lockout_check(sig, slice(max(0, rep.start - w), rep.start + 1), cfg, "start"))
    if rep.complete:
        v.checks.append(_lockout_check(sig, slice(rep.end, min(len(sig), rep.end + w)), cfg, "finish"))
    else:
        v.checks.append(Check("squat_lockout_finish", Card.BLUE, Status.FAIL,
                              "Lift not completed (did not return to standing)", rep.end))

    # ---- BLUE: double bounce in the hole ----------------------------------
    descent = depth[rep.start: rep.bottom + 1]
    in_hole = descent > max_depth - cfg.bottom_zone
    drawback = 0.0
    if in_hole.any():
        hole = descent[in_hole.argmax():]
        drawback = float((np.maximum.accumulate(hole) - hole).max())
    bounced = drawback > cfg.bounce_tol
    v.checks.append(Check(
        "squat_bounce", Card.BLUE, Status.FAIL if bounced else Status.PASS,
        (f"Double bounce: hips rose {drawback:.2f} thighs then dropped again in the hole"
         if bounced else "Single, controlled bottom position"),
        rep.bottom, drawback, cfg.bounce_tol))

    # ---- BLUE: downward movement during the ascent -------------------------
    ascent = depth[rep.bottom: rep.end + 1]
    drop = float((ascent - np.minimum.accumulate(ascent)).max()) if len(ascent) else 0.0
    drop_frame = rep.bottom + int((ascent - np.minimum.accumulate(ascent)).argmax()) if len(ascent) else None
    moved_down = drop > cfg.downward_tol
    v.checks.append(Check(
        "squat_downward_movement", Card.BLUE, Status.FAIL if moved_down else Status.PASS,
        (f"Downward movement during the ascent ({drop:.2f} thighs)"
         if moved_down else "Continuous ascent, no downward movement"),
        drop_frame if moved_down else None, drop, cfg.downward_tol))

    # ---- YELLOW: feet moved -----------------------------------------------
    ank = sig[["ankle_x", "ankle_y"]].iloc[rep.start: rep.end + 1].to_numpy()
    thigh = float(sig["thigh_px"].iloc[0])
    shift = float(np.linalg.norm(ank - ank[0], axis=1).max() / thigh) if len(ank) else 0.0
    stepped = shift > cfg.feet_tol
    v.checks.append(Check(
        "squat_feet", Card.YELLOW, Status.FAIL if stepped else Status.PASS,
        f"Feet moved {shift:.2f} thighs (step)" if stepped else "Feet stayed in place",
        None, shift, cfg.feet_tol))

    # ---- coaching metrics (not judged) ------------------------------------
    shoulder_x = sig["shoulder_x"].iloc[rep.start: rep.end + 1].to_numpy() / thigh
    # slowest point of the ascent (the sticking point), measured between 10% and
    # 90% of the way up so the pause at the bottom and the lockout do not count
    ascent_vel = -sig["hip_vel"].to_numpy()[rep.bottom: rep.end + 1]
    lo, hi = depth[rep.bottom], depth[rep.end]
    mid = (ascent < lo - 0.1 * (lo - hi)) & (ascent > hi + 0.1 * (lo - hi))
    v.metrics.update(
        depth_margin=margin,
        descent_s=(rep.bottom - rep.start) / fps,
        ascent_s=(rep.end - rep.bottom) / fps,
        shoulder_drift=float(shoulder_x.max() - shoulder_x.min()),  # horizontal sway, thighs
        sticking_speed=float(ascent_vel[mid].min()) if mid.any() else float("nan"),  # thighs/s
        sticking_frame=int(rep.bottom + np.flatnonzero(mid)[ascent_vel[mid].argmin()]) if mid.any() else -1,
        leg_conf=conf,
    )
    return v


def judge_squat(sig: pd.DataFrame, cfg: SquatConfig | None = None) -> list[RepVerdict]:
    """Judge every rep in a squat video's signals (from `signals.body_signals`)."""
    reps = segment_reps(sig["depth"].to_numpy(), sig.attrs.get("fps", 30.0))
    return [judge_rep(sig, r, cfg) for r in reps]
