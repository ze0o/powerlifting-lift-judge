from .base import Card, Check, RepVerdict, Status
from .squat import SquatConfig, judge_squat

JUDGES = {"squat": judge_squat}

__all__ = ["Card", "Check", "RepVerdict", "Status", "SquatConfig", "judge_squat", "JUDGES"]
