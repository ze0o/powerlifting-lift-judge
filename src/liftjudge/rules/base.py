"""Shared types for the lift rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Card(str, Enum):
    """IPF referee card colours. Each colour is a group of failures for a lift."""
    RED = "red"
    BLUE = "blue"
    YELLOW = "yellow"


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BORDERLINE = "borderline"   # too close to call from video: a human should look
    UNKNOWN = "unknown"         # not enough data (e.g. keypoints not visible)


@dataclass
class Check:
    """Result of one rule on one rep."""
    code: str               # e.g. "squat_depth"
    card: Card
    status: Status
    message: str            # human explanation, shown to the lifter
    frame: int | None = None
    value: float | None = None       # measured value
    threshold: float | None = None   # what it was compared against


@dataclass
class RepVerdict:
    lift: str
    rep: int
    start: int
    bottom: int
    end: int
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def status(self) -> Status:
        states = {c.status for c in self.checks}
        if Status.FAIL in states:
            return Status.FAIL
        if Status.BORDERLINE in states or Status.UNKNOWN in states:
            return Status.BORDERLINE
        return Status.PASS

    @property
    def cards(self) -> list[Card]:
        """Cards a referee would show: one per failed colour group."""
        failed = {c.card for c in self.checks if c.status == Status.FAIL}
        return [c for c in Card if c in failed]

    @property
    def label(self) -> str:
        return {Status.PASS: "GOOD LIFT", Status.FAIL: "NO LIFT",
                Status.BORDERLINE: "BORDERLINE"}.get(self.status, "UNKNOWN")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["label"] = self.label
        d["cards"] = [c.value for c in self.cards]
        for c in d["checks"]:
            c["card"], c["status"] = c["card"].value, c["status"].value
        return d

    def summary(self) -> str:
        lines = [f"Rep {self.rep + 1}: {self.label}"
                 + (f"  [{', '.join(c.value for c in self.cards)} card]" if self.cards else "")]
        for c in self.checks:
            mark = {"pass": "ok ", "fail": "X  ", "borderline": "?  ", "unknown": "-  "}[c.status.value]
            lines.append(f"   {mark}{c.message}")
        return "\n".join(lines)
