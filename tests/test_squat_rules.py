import numpy as np
import pytest

from liftjudge import analyze_pose
from liftjudge.keypoints import PoseSequence
from liftjudge.rules import Card, Status
from liftjudge.synthetic import make_squat


def judge(**kw):
    return analyze_pose(make_squat(**kw), "squat").verdicts


def check(verdict, code):
    return next(c for c in verdict.checks if c.code == code)


def test_deep_squat_is_good_lift():
    (v,) = judge(depth_deg=15)
    assert v.status == Status.PASS, v.summary()
    assert v.cards == []


def test_high_squat_gets_red_card():
    (v,) = judge(depth_deg=-10)
    assert v.status == Status.FAIL
    assert check(v, "squat_depth").status == Status.FAIL
    assert Card.RED in v.cards


def test_right_on_the_line_is_borderline():
    # depth ~= crease offset: too close to call from video
    (v,) = judge(depth_deg=4)
    assert check(v, "squat_depth").status == Status.BORDERLINE
    assert v.status == Status.BORDERLINE


def test_downward_movement_on_ascent_gets_blue_card():
    (v,) = judge(depth_deg=15, ascent_dip=0.12)
    assert check(v, "squat_downward_movement").status == Status.FAIL
    assert Card.BLUE in v.cards


def test_double_bounce_gets_blue_card():
    (v,) = judge(depth_deg=15, bounce=True)
    assert v.status == Status.FAIL, v.summary()
    assert Card.BLUE in v.cards


def test_not_locked_at_finish_gets_blue_card():
    (v,) = judge(depth_deg=15, finish_s=0.25)
    assert check(v, "squat_lockout_finish").status == Status.FAIL
    assert check(v, "squat_lockout_start").status == Status.PASS


def test_step_gets_yellow_card():
    (v,) = judge(depth_deg=15, step_px=45)
    assert check(v, "squat_feet").status == Status.FAIL
    assert Card.YELLOW in v.cards


def test_counts_every_rep_in_a_set():
    vs = judge(depth_deg=15, reps=3)
    assert len(vs) == 3
    assert all(v.status == Status.PASS for v in vs)


def test_works_at_60_fps_and_from_the_right_side():
    (v,) = judge(depth_deg=15, fps=60, side="right")
    assert v.status == Status.PASS


def test_hidden_legs_at_the_bottom_are_not_judged():
    seq = make_squat(depth_deg=15)
    res = analyze_pose(seq, "squat")
    bottom = res.verdicts[0].bottom
    seq.conf[bottom - 5: bottom + 5] = 0.1
    (v,) = analyze_pose(seq, "squat").verdicts
    assert check(v, "squat_depth").status == Status.UNKNOWN
    assert v.status == Status.BORDERLINE


def test_keypoints_csv_round_trip(tmp_path):
    seq = make_squat()
    seq.save_csv(tmp_path / "k.csv")
    back = PoseSequence.load_csv(tmp_path / "k.csv")
    assert back.fps == seq.fps and back.width == seq.width
    np.testing.assert_allclose(back.xy, seq.xy)


def test_unsupported_lift_is_explicit():
    with pytest.raises(NotImplementedError):
        analyze_pose(make_squat(), "bench")
