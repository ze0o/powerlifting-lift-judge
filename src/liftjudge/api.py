"""HTTP API for the mobile app.

Run on your PC (phone on the same Wi-Fi):
    uvicorn liftjudge.api:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health                  status + whether the neural network is loaded
  GET  /sample                  analysis of a synthetic 3-rep set (no video needed)
  POST /analyze                 multipart: video file + lift, model_size, crease_offset
  GET  /videos/{job_id}.mp4     annotated video (H.264) of an analysed upload
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

from . import __version__, analyze_pose
from .keypoints import PoseSequence, extract_pose
from .rules import SquatConfig

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
except ImportError as e:  # pragma: no cover
    raise ImportError("The API needs: pip install -e \".[api]\"") from e

ROOT = Path(__file__).resolve().parents[2]
JOBS = Path(os.environ.get("LIFTJUDGE_JOBS", Path(tempfile.gettempdir()) / "liftjudge_jobs"))
DL_PATH = Path(os.environ.get("LIFTJUDGE_DL_MODEL", ROOT / "models" / "rep_fault_net.keras"))
MAX_MB = 300

app = FastAPI(title="Powerlifting Lift Judge", version=__version__)
# The Flutter web app runs on another localhost port, so the browser needs CORS.
# Development server on your own network: any origin is fine. Lock this down if you ever host it.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
_dl = {"model": None, "tried": False}


def dl_model():
    if not _dl["tried"]:
        _dl["tried"] = True
        try:
            from .dl import load_model
            _dl["model"] = load_model(DL_PATH) if DL_PATH.exists() else None
        except Exception:
            _dl["model"] = None
    return _dl["model"]


def _series(sig, n: int = 300) -> dict:
    """Downsampled depth curve for the app's chart."""
    idx = np.linspace(0, len(sig) - 1, min(n, len(sig))).astype(int)
    return {"time_s": [round(float(x), 3) for x in sig["time_s"].to_numpy()[idx]],
            "depth": [round(float(x), 4) for x in sig["depth"].to_numpy()[idx]]}


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
    model = dl_model()
    if model is not None and res.verdicts:
        from .dl import FAULT_TEXT, FAULTS, predict_faults
        dl = predict_faults(model, res)
    for v in res.verdicts:
        d = v.to_dict()
        d["start_s"], d["bottom_s"], d["end_s"] = (round(x / fps, 2) for x in (v.start, v.bottom, v.end))
        if dl is not None:
            r = dl.iloc[v.rep]
            d["dl"] = {"label": r.dl_label,
                       "faults": {f: round(float(r[f]), 3) for f in FAULTS},
                       "found": [FAULT_TEXT[f] for f in FAULTS if r[f] >= 0.5]}
        reps.append(d)
    statuses = [r["status"] for r in reps]
    return _clean({
        "lift": res.lift,
        "fps": fps,
        "summary": {"reps": len(reps), "good": statuses.count("pass"),
                    "no_lift": statuses.count("fail"), "borderline": statuses.count("borderline")},
        "config": {"crease_offset": cfg.crease_offset, "borderline_band": cfg.borderline_band},
        "reps": reps,
        "series": _series(res.signals) if len(res.signals) else {"time_s": [], "depth": []},
        "video_url": video_url,
        "dl_available": model is not None,
    })


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "dl_model": dl_model() is not None}


@app.get("/sample")
def sample(crease_offset: float = SquatConfig.crease_offset):
    from .synthetic import make_squat
    parts = [make_squat(depth_deg=d, bounce=b, seed=s) for d, b, s in ((14, False, 1), (-6, False, 2), (12, True, 3))]
    pose = PoseSequence(xy=np.concatenate([p.xy for p in parts]), conf=np.concatenate([p.conf for p in parts]),
                        fps=parts[0].fps, width=parts[0].width, height=parts[0].height)
    cfg = SquatConfig(crease_offset=crease_offset)
    return build_response(analyze_pose(pose, "squat", config=cfg), cfg)


@app.post("/analyze")
def analyze(video: UploadFile = File(...), lift: str = Form("squat"), model_size: str = Form("n"),
            crease_offset: float = Form(SquatConfig.crease_offset), render: bool = Form(True)):
    if lift != "squat":
        raise HTTPException(400, f"'{lift}' is not supported yet: only squat for now")
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
    cfg = SquatConfig(crease_offset=crease_offset)
    try:
        res = analyze_pose(pose, lift, config=cfg)
    except ValueError:
        raise HTTPException(422, "Couldn't see the lifter's legs. Film from the side with the whole body in frame.")

    url = None
    if render and res.verdicts:
        from .render import render_video
        render_video(src, folder / "judged.mp4", pose, res.signals, res.verdicts, cfg.crease_offset)
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
