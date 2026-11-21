from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from liftjudge.synthetic import make_squat  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LIFTJUDGE_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setenv("LIFTJUDGE_DATA", str(tmp_path / "data" / "raw"))
    import importlib

    import liftjudge.api as api
    importlib.reload(api)
    return TestClient(api.app), api


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_sample_matches_the_rules(client):
    c, _ = client
    body = c.get("/sample").json()
    assert body["summary"] == {"reps": 3, "good": 1, "no_lift": 2, "borderline": 0}
    assert [r["label"] for r in body["reps"]] == ["GOOD LIFT", "NO LIFT", "NO LIFT"]
    assert body["reps"][1]["cards"] == ["red"]
    assert body["series"]["metric"] == "depth"
    assert len(body["series"]["time_s"]) == len(body["series"]["value"]) > 50
    assert body["view"]["kind"] == "side"


def test_analyze_upload_returns_verdicts_and_playable_video(client, monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    c, api = client
    pose = make_squat(depth_deg=-8, reps=2, seed=5)
    video = tmp_path / "set.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), pose.fps, (pose.width, pose.height))
    for _ in range(len(pose)):
        w.write(np.full((pose.height, pose.width, 3), 60, np.uint8))
    w.release()
    # pose detection is tested elsewhere; here feed the known keypoints
    monkeypatch.setattr(api, "extract_pose", lambda path, model: pose)

    with open(video, "rb") as f:
        r = c.post("/analyze", files={"video": ("set.mp4", f, "video/mp4")},
                   data={"lift": "squat", "model_size": "n"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["no_lift"] == 2
    assert body["reps"][0]["start_s"] < body["reps"][0]["bottom_s"] < body["reps"][0]["end_s"]
    v = c.get(body["video_url"])
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4" and len(v.content) > 1000


def _upload(c, api, monkeypatch, tmp_path, pose, lift):
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / f"{lift}.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), pose.fps, (pose.width, pose.height))
    for _ in range(len(pose)):
        w.write(np.full((pose.height, pose.width, 3), 60, np.uint8))
    w.release()
    monkeypatch.setattr(api, "extract_pose", lambda path, model: pose)
    with open(video, "rb") as f:
        return c.post("/analyze", files={"video": (video.name, f, "video/mp4")}, data={"lift": lift})


def test_deadlift_sample_and_upload(client, monkeypatch, tmp_path):
    from liftjudge.synthetic import make_deadlift
    c, api = client
    body = c.get("/sample", params={"lift": "deadlift"}).json()
    assert [r["label"] for r in body["reps"]] == ["GOOD LIFT", "NO LIFT", "NO LIFT"]
    assert body["series"]["metric"] == "bar_height" and body["config"] == {}
    assert "dl" not in body["reps"][0]          # the network only knows squats

    r = _upload(c, api, monkeypatch, tmp_path, make_deadlift(hitch=True, seed=4), "deadlift")
    assert r.status_code == 200, r.text
    assert r.json()["reps"][0]["cards"] == ["blue"]


def test_save_for_training_copies_video_and_keypoints(client, monkeypatch, tmp_path):
    c, api = client
    body = _upload(c, api, monkeypatch, tmp_path, make_squat(seed=6), "squat").json()
    r = c.post(f"/jobs/{body['job_id']}/save")
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["lift"] == "squat" and saved["next"] == "liftjudge label data/raw/squat"
    folder = tmp_path / "data" / "raw" / "squat"
    videos = [p for p in folder.iterdir() if p.suffix == ".mp4"]
    assert len(videos) == 1 and (folder / f"{videos[0].stem}_keypoints.csv").exists()
    c.post(f"/jobs/{body['job_id']}/save")                     # saving twice keeps one copy
    assert len([p for p in folder.iterdir() if p.suffix == ".mp4"]) == 1
    assert c.post("/jobs/doesnotexist/save").status_code == 404


def test_front_view_upload_is_flagged(client, monkeypatch, tmp_path):
    c, api = client
    body = _upload(c, api, monkeypatch, tmp_path, make_squat(depth_deg=-10, view="front", seed=8), "squat").json()
    assert body["view"]["kind"] == "front"
    assert body["summary"]["borderline"] == 1 and body["summary"]["no_lift"] == 0


def test_rejects_unsupported_lift(client, tmp_path):
    c, _ = client
    r = c.post("/analyze", files={"video": ("a.mp4", b"x", "video/mp4")}, data={"lift": "bench"})
    assert r.status_code == 400


def test_video_path_is_not_traversable(client):
    c, _ = client
    assert c.get("/videos/..%2F..%2Fetc.mp4").status_code == 404


def test_unmeasurable_metrics_become_null_not_a_crash(client):
    import math
    _, api = client
    from liftjudge import analyze_pose
    from liftjudge.rules import SquatConfig
    res = analyze_pose(make_squat(seed=9), "squat")
    res.verdicts[0].metrics["sticking_speed"] = math.nan
    import json
    body = api.build_response(res, SquatConfig())
    json.dumps(body, allow_nan=False)
    assert body["reps"][0]["metrics"]["sticking_speed"] is None


def test_browser_app_is_allowed_by_cors(client):
    c, _ = client
    r = c.options("/analyze", headers={"Origin": "http://localhost:61943",
                                       "Access-Control-Request-Method": "POST"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] in ("*", "http://localhost:61943")
