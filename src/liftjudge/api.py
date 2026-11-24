"""HTTP API for the mobile app.

Run on your PC (phone on the same Wi-Fi):
    uvicorn liftjudge.api:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health                  status + whether the neural network is loaded
  GET  /sample?lift=squat       analysis of a synthetic 3-rep set (no video needed)
  POST /analyze                 multipart: video file + lift (squat|deadlift), model_size, crease_offset
  GET  /videos/{job_id}.mp4     annotated video (H.264) of an analysed upload
  POST /jobs/{job_id}/save      keep an analysed upload for training: data/raw/<lift>/
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from . import __version__, analyze_pose
from .keypoints import PoseSequence, extract_pose
from .rules import DeadliftConfig, SquatConfig

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
except ImportError as e:  # pragma: no cover
    raise ImportError("The API needs: pip install -e \".[api]\"") from e

ROOT = Path(__file__).resolve().parents[2]
JOBS = Path(os.environ.get("LIFTJUDGE_JOBS", Path(tempfile.gettempdir()) / "liftjudge_jobs"))
DL_PATHS = {
    "squat": Path(os.environ.get("LIFTJUDGE_DL_MODEL", ROOT / "models" / "rep_fault_net.keras")),
    "deadlift": Path(os.environ.get("LIFTJUDGE_DL_MODEL_DEADLIFT", ROOT / "models" / "rep_fault_net_deadlift.keras")),
}
DATA = Path(os.environ.get("LIFTJUDGE_DATA", ROOT / "data" / "raw"))
MAX_MB = 300
LIFTS = ("squat", "deadlift")

app = FastAPI(title="Powerlifting Lift Judge", version=__version__)
# The Flutter web app runs on another localhost port, so the browser needs CORS.
# Development server on your own network: any origin is fine. Lock this down if you ever host it.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
_dl: dict[str, object] = {}


def dl_model(lift: str = "squat"):
    """The neural network for this lift, loaded once (None if not trained or no Keras)."""
    if lift not in _dl:
        try:
            from .dl import load_model
            path = DL_PATHS.get(lift)
            _dl[lift] = load_model(path) if path is not None and path.exists() else None
        except Exception:
            _dl[lift] = None
    return _dl[lift]


def _series(sig, lift: str, n: int = 300) -> dict:
    """Downsampled curve for the app's chart: hip depth (squat) or bar height (deadlift)."""
    if not len(sig):
        return {"metric": "depth", "time_s": [], "value": []}
    idx = np.linspace(0, len(sig) - 1, min(n, len(sig))).astype(int)
    if lift == "deadlift":
        from .rules.deadlift import bar_height
        metric, values = "bar_height", bar_height(sig)
    else:
        metric, values = "depth", sig["depth"].to_numpy()
    return {"metric": metric,
            "time_s": [round(float(x), 3) for x in sig["time_s"].to_numpy()[idx]],
            "value": [round(float(x), 4) for x in values[idx]]}


def _clean(x):
    """JSON has no NaN/inf: turn them into null (e.g. a metric that couldn't be measured)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def build_response(res, cfg: SquatConfig, video_url: str | None = None) -> dict:
    fps = res.pose.fps
    reps = []
    dl = None
    model = dl_model(res.lift)               # one network per lift
    if model is not None and res.verdicts:
        from .dl import LIFT_FAULTS, fault_text, predict_faults
        faults = LIFT_FAULTS[res.lift]
        dl = predict_faults(model, res)
    for v in res.verdicts:
        d = v.to_dict()
        d["start_s"], d["bottom_s"], d["end_s"] = (round(x / fps, 2) for x in (v.start, v.bottom, v.end))
        if dl is not None:
            r = dl.iloc[v.rep]
            d["dl"] = {"label": r.dl_label,
                       "faults": {f: round(float(r[f]), 3) for f in faults},
                       "found": [fault_text(res.lift, f) for f in faults if r[f] >= 0.5]}
        reps.append(d)
    statuses = [r["status"] for r in reps]
    return _clean({
        "lift": res.lift,
        "fps": fps,
        "summary": {"reps": len(reps), "good": statuses.count("pass"),
                    "no_lift": statuses.count("fail"), "borderline": statuses.count("borderline")},
        "config": ({"crease_offset": cfg.crease_offset, "borderline_band": cfg.borderline_band}
                   if res.lift == "squat" else {}),
        "view": res.view.to_dict() if res.view else None,
        "reps": reps,
        "series": _series(res.signals, res.lift),
        "video_url": video_url,
        "dl_available": model is not None,
    })


@app.get("/health")
def health():
    models = {lift: dl_model(lift) is not None for lift in LIFTS}
    return {"status": "ok", "version": __version__, "dl_model": models["squat"], "dl_models": models}


@app.get("/sample")
def sample(lift: str = "squat", crease_offset: float = SquatConfig.crease_offset):
    if lift not in LIFTS:
        raise HTTPException(400, f"lift must be one of {', '.join(LIFTS)}")
    cfg = SquatConfig(crease_offset=crease_offset)
    from .synthetic import sample_set
    return build_response(analyze_pose(sample_set(lift), lift, config=cfg), cfg)


@app.post("/analyze")
def analyze(video: UploadFile = File(...), lift: str = Form("squat"), model_size: str = Form("n"),
            crease_offset: float = Form(SquatConfig.crease_offset), render: bool = Form(True)):
    if lift not in LIFTS:
        raise HTTPException(400, f"'{lift}' is not supported yet: squat and deadlift for now")
    if model_size not in {"n", "s", "m", "l", "x"}:
        raise HTTPException(400, "model_size must be one of n, s, m, l, x")
    job = uuid.uuid4().hex[:12]
    folder = JOBS / job
    folder.mkdir(parents=True, exist_ok=True)
    src = folder / f"input{Path(video.filename or 'v.mp4').suffix or '.mp4'}"
    with open(src, "wb") as f:
        shutil.copyfileobj(video.file, f)
    if src.stat().st_size > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Video larger than {MAX_MB} MB: trim it to the set")

    try:
        pose = extract_pose(src, model=f"yolo26{model_size}-pose.pt")
    except FileNotFoundError:
        raise HTTPException(400, "Could not read the video file")
    pose.save_csv(folder / "keypoints.csv")
    (folder / "job.json").write_text(json.dumps({"lift": lift, "file": src.name,
                                                 "original_name": video.filename, "model_size": model_size}))
    cfg = SquatConfig(crease_offset=crease_offset)
    try:
        res = analyze_pose(pose, lift, config=cfg)
    except ValueError:
        raise HTTPException(422, "Couldn't see the lifter's legs. Film from the side with the whole body in frame.")

    url = None
    if render and res.verdicts:
        from .render import render_video
        render_video(src, folder / "judged.mp4", pose, res.signals, res.verdicts, cfg.crease_offset, lift=lift)
        url = f"/videos/{job}.mp4"
    out = build_response(res, cfg, url)
    out["job_id"] = job
    return out


@app.get("/videos/{job_id}.mp4")
def video(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(404)
    path = JOBS / job_id / "judged.mp4"
    if not path.exists():
        raise HTTPException(404, "No video for this job")
    return FileResponse(path, media_type="video/mp4")


@app.post("/jobs/{job_id}/save")
def save_for_training(job_id: str):
    """Copy an analysed upload into data/raw/<lift>/ (with its keypoints) for labelling.

    Next: `liftjudge label data/raw/<lift>` drafts the labels for you to check.
    """
    folder = JOBS / job_id
    if not job_id.isalnum() or not (folder / "job.json").exists():
        raise HTTPException(404, "Unknown job: analyse the video first")
    meta = json.loads((folder / "job.json").read_text())
    dest = DATA / meta["lift"]
    dest.mkdir(parents=True, exist_ok=True)
    stem = f"{meta['lift']}_{datetime.now():%Y-%m-%d_%H%M%S}_{job_id[:6]}"
    video = dest / f"{stem}{Path(meta['file']).suffix}"
    if not any(dest.glob(f"*_{job_id[:6]}.*")):          # saving twice keeps one copy
        shutil.copy2(folder / meta["file"], video)
        shutil.copy2(folder / "keypoints.csv", dest / f"{stem}_keypoints.csv")
    else:
        video = next(p for p in dest.glob(f"*_{job_id[:6]}.*") if not p.stem.endswith("_keypoints"))
    rel = video.relative_to(ROOT) if video.is_relative_to(ROOT) else video
    return {"saved": True, "path": str(rel).replace("\\", "/"), "lift": meta["lift"],
            "next": f"liftjudge label data/raw/{meta['lift']}"}
