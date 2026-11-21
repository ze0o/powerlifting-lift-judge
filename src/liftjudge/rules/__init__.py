from .base import Card, Check, RepVerdict, Status
from .deadlift import DeadliftConfig, judge_deadlift
from .squat import SquatConfig, judge_squat

JUDGES = {"squat": judge_squat, "deadlift": judge_deadlift}

__all__ = ["Card", "Check", "RepVerdict", "Status", "SquatConfig", "DeadliftConfig",
           "judge_squat", "judge_deadlift", "JUDGES"]
