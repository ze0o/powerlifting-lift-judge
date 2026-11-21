import numpy as np

from liftjudge import analyze_pose
from liftjudge.keypoints import select_lifter
from liftjudge.rules import Status
from liftjudge.synthetic import make_squat


def depth_check(v):
    return next(c for c in v.checks if c.code == "squat_depth")


def test_side_view_is_detected_and_judged():
    res = analyze_pose(make_squat(depth_deg=12))
    assert res.view.kind == "side"
    assert res.verdicts[0].status == Status.PASS


def test_front_view_does_not_judge_depth():
    res = analyze_pose(make_squat(depth_deg=-10, view="front"))   # high, but filmed from the front
    assert res.view.kind == "front"
    assert res.signals.attrs["side"] == "both"
    v = res.verdicts[0]
    assert depth_check(v).status == Status.UNKNOWN
    assert "front" in depth_check(v).message
    assert v.status == Status.BORDERLINE              # never a confident call
    assert "Front/back view" in res.summary()


def test_front_view_still_catches_a_bounce():
    v = analyze_pose(make_squat(depth_deg=12, bounce=True, view="front")).verdicts[0]
    assert v.status == Status.FAIL


def test_angled_view_widens_the_borderline_zone():
    # margin ~ +0.07: a pass from the side, borderline at 45°
    side = analyze_pose(make_squat(depth_deg=8)).verdicts[0]
    angled = analyze_pose(make_squat(depth_deg=8, view="angled"))
    assert angled.view.kind == "angled"
    assert depth_check(side).status == Status.PASS
    assert depth_check(angled.verdicts[0]).status == Status.BORDERLINE


def _det(id_, x1, x2, y1=100, y2=500):
    return {"id": id_, "box": np.array([x1, y1, x2, y2], float), "kxy": np.full((17, 2), float(id_)),
            "kconf": np.ones(17)}


def test_lifter_beats_a_bigger_spotter_at_the_edge():
    # spotter at the edge is slightly bigger, lifter is centred
    frames = [[_det(1, 0, 230), _det(2, 400, 600)] for _ in range(10)]
    picks = select_lifter(frames, width=1000)
    assert all(p["id"] == 2 for p in picks)


def test_follows_the_lifter_through_an_id_switch():
    # tracker loses id 2 at the bottom and calls the lifter id 7 from frame 5
    frames = [[_det(1, 0, 150), _det(2, 400, 600)] for _ in range(5)]
    frames += [[_det(1, 0, 150), _det(7, 410, 605, 140, 520)] for _ in range(5)]
    picks = select_lifter(frames, width=1000)
    assert [p["id"] for p in picks] == [2] * 5 + [7] * 5


def test_frames_before_a_late_id_are_stitched_backwards():
    # the new id (7) is on screen longer, so it wins; frames before the switch must still be found
    frames = [[_det(1, 0, 150), _det(2, 400, 600)] for _ in range(3)]
    frames += [[_det(1, 0, 150), _det(7, 410, 605, 140, 520)] for _ in range(9)]
    picks = select_lifter(frames, width=1000)
    assert all(p is not None for p in picks)
    assert [p["id"] for p in picks] == [2] * 3 + [7] * 9


def test_nobody_detected():
    assert select_lifter([[], []], width=100) == [None, None]
