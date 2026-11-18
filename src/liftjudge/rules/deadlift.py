"""Deadlift rules - phase 3 (not built yet).

Planned checks (IPF 2026 card groups):
  RED    knees not locked at completion; front deltoid not behind the bar;
         not standing erect with shoulders back (shoulder x behind hip x)
  BLUE   downward movement of the bar before the final position (hitching
         detection on the bar height signal); supporting the bar on the thighs
  YELLOW lowering before the signal / dropping the bar (needs audio or the
         referee light, out of scope for video-only judging)
"""


def judge_deadlift(*_args, **_kwargs):
    raise NotImplementedError("Deadlift judging is planned for phase 3 - see docs/roadmap.md")
