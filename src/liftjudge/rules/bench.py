"""Bench press rules - phase 3 (not built yet).

Planned checks (IPF 2026 card groups):
  RED    bar not lowered to the chest/abdomen; elbows not at or below shoulder level
  BLUE   downward movement of the whole bar while pressing; elbows not locked
  YELLOW heaving/sinking after the pause; head/shoulders/buttocks lifting
         (hip keypoint leaves the bench line); feet touching the bench

Needs barbell tracking (fine-tuned plate detector) because the wrists hide
behind the plates from a side view. The pause is detected as bar speed ~0 for
a minimum number of frames at the lowest point.
"""


def judge_bench(*_args, **_kwargs):
    raise NotImplementedError("Bench press judging is planned for phase 3 - see docs/roadmap.md")
