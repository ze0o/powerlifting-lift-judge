"""Split a set into individual reps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class Rep:
    index: int
    start: int      # frame where the descent begins
    bottom: int     # frame of the lowest point
    end: int        # frame where the lifter is back at the top
    complete: bool  # False if the video ends before the lifter finishes


def segment_reps(signal: np.ndarray, fps: float, min_range: float = 0.5,
                 min_gap_s: float = 0.8, top_frac: float = 0.15) -> list[Rep]:
    """Find reps in a 1-D signal where a rep is a rise to a peak and back down.

    For the squat pass `depth` (rises as the hips go down). For the bench pass
    the wrist y (rises as the bar goes down). For the deadlift pass the negative
    bar height.

    min_range   minimum peak prominence in thigh lengths, so small wobbles at the
                bottom (e.g. a double bounce) stay inside one rep.
    top_frac    how close to the top (as a fraction of the rep's range) counts as
                "back at the top".
    """
    x = pd.Series(signal).interpolate(limit_direction="both").to_numpy()
    peaks, _ = find_peaks(x, prominence=min_range, distance=max(1, int(min_gap_s * fps)))
    reps: list[Rep] = []
    bounds = [0, *peaks.tolist(), len(x) - 1]
    for k, p in enumerate(peaks):
        lo_before = bounds[k] if k == 0 else reps[-1].end
        before = x[lo_before:p + 1]
        after = x[p:bounds[k + 2] + 1]
        top_before, top_after = before.min(), after.min()
        thr_before = top_before + top_frac * (x[p] - top_before)
        thr_after = top_after + top_frac * (x[p] - top_after)

        idx_b = np.where(before <= thr_before)[0]
        start = lo_before + (idx_b[-1] if len(idx_b) else 0)
        idx_a = np.where(after <= thr_after)[0]
        complete = len(idx_a) > 0
        end = p + (idx_a[0] if complete else len(after) - 1)
        # extend the end to where the lifter actually settles at the top
        if complete:
            seg = x[end:bounds[k + 2] + 1]
            settle = np.where(np.diff(seg) >= 0)[0]
            end = end + (settle[0] if len(settle) else len(seg) - 1)
        reps.append(Rep(index=len(reps), start=int(start), bottom=int(p),
                        end=int(end), complete=bool(complete)))
    return reps
