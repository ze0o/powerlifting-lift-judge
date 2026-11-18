"""liftjudge: judge powerlifting technique from video against IPF rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .keypoints import DEFAULT_MODEL, PoseSequence, extract_pose
from .rules import JUDGES, RepVerdict, SquatConfig
from .signals import body_signals

__version__ = "0.1.0"


@dataclass
class Analysis:
    lift: str
    pose: PoseSequence
    signals: pd.DataFrame
    verdicts: list[RepVerdict]

    def summary(self) -> str:
        if not self.verdicts:
            return "No reps found. Check the camera angle (side view, whole body in frame)."
        return "\n".join(v.summary() for v in self.verdicts)


def analyze_pose(pose: PoseSequence, lift: str = "squat", side: str | None = None,
                 config: SquatConfig | None = None) -> Analysis:
    if lift not in JUDGES:
        raise NotImplementedError(f"'{lift}' is not supported yet (available: {', '.join(JUDGES)})")
    sig = body_signals(pose, side=side)
    return Analysis(lift, pose, sig, JUDGES[lift](sig, config))


def analyze(video: str | Path, lift: str = "squat", model: str = DEFAULT_MODEL,
            side: str | None = None, config: SquatConfig | None = None) -> Analysis:
    """Full pipeline: video -> keypoints -> signals -> per-rep verdicts."""
    return analyze_pose(extract_pose(video, model=model), lift, side, config)


__all__ = ["analyze", "analyze_pose", "Analysis", "PoseSequence", "extract_pose",
           "body_signals", "SquatConfig", "RepVerdict"]
