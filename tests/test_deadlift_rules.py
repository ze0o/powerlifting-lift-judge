from liftjudge import analyze_pose
from liftjudge.rules import Card, Status
from liftjudge.synthetic import make_deadlift


def judge(**kw):
    return analyze_pose(make_deadlift(**kw), "deadlift").verdicts


def check(v, code):
    return next(c for c in v.checks if c.code == code)


def test_clean_pull_is_good():
    (v,) = judge()
    assert v.status == Status.PASS, v.summary()


def test_soft_lockout_red_card():
    (v,) = judge(lean_top=20)
    assert check(v, "deadlift_erect").status == Status.FAIL
    assert v.cards == [Card.RED]


def test_knees_not_locked_red_card():
    (v,) = judge(knee_top=22)
    assert check(v, "deadlift_knees_locked").status == Status.FAIL
    assert Card.RED in v.cards


def test_downward_movement_blue_card():
    (v,) = judge(dip=0.08)
    assert check(v, "deadlift_downward_movement").status == Status.FAIL
    assert check(v, "deadlift_hitching").status == Status.PASS   # a dip is not a hitch
    assert v.cards == [Card.BLUE]


def test_hitching_blue_card():
    (v,) = judge(hitch=True)
    assert check(v, "deadlift_hitching").status == Status.FAIL
    assert Card.BLUE in v.cards


def test_step_yellow_card():
    (v,) = judge(step_px=45)
    assert v.cards == [Card.YELLOW]


def test_counts_reps_any_speed_and_side():
    assert len(judge(reps=3)) == 3
    (v,) = judge(fps=60, side="right", tempo=1.5)
    assert v.status == Status.PASS


def test_front_view_does_not_judge_lockout():
    from liftjudge.synthetic import KP
    seq = make_deadlift(lean_top=20)
    # spread left/right like a front camera
    for j, w in (("shoulder", 90), ("hip", 55), ("knee", 60), ("ankle", 65), ("elbow", 100), ("wrist", 95)):
        seq.xy[:, KP[f"left_{j}"], 0] -= w
        seq.xy[:, KP[f"right_{j}"], 0] = seq.xy[:, KP[f"left_{j}"], 0] + 2 * w
        seq.conf[:, KP[f"right_{j}"]] = 0.9
    res = analyze_pose(seq, "deadlift")
    assert res.view.kind == "front"
    (v,) = res.verdicts
    assert check(v, "deadlift_erect").status == Status.UNKNOWN
    assert v.status == Status.BORDERLINE
