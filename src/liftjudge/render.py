"""Draw the skeleton, a depth line and the verdict on top of the video."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .keypoints import KP, SKELETON, PoseSequence
from .rules.base import RepVerdict, Status

# BGR colours
GREEN, RED, AMBER, WHITE, BLACK = (80, 200, 80), (60, 60, 230), (0, 190, 255), (255, 255, 255), (0, 0, 0)
STATUS_COLOR = {Status.PASS: GREEN, Status.FAIL: RED, Status.BORDERLINE: AMBER, Status.UNKNOWN: AMBER}
CARD_BGR = {"red": (40, 40, 220), "blue": (220, 120, 30), "yellow": (0, 215, 255)}


def _banner(cv2, img, text, color, y=10):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(img, (10, y), (30 + tw, y + th + 20), color, -1)
    cv2.putText(img, text, (20, y + th + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
    return y + th + 30


def ffmpeg_exe() -> str | None:
    """ffmpeg from imageio-ffmpeg (bundled binary on Windows/macOS/Linux), else PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg")


def to_h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 so the video plays in browsers, Streamlit and phones."""
    import subprocess
    exe = ffmpeg_exe()
    if not exe:
        return False
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    return subprocess.run(cmd).returncode == 0 and dst.exists()


def render_video(video_path: str | Path, out_path: str | Path, seq: PoseSequence,
                 sig: pd.DataFrame, verdicts: list[RepVerdict], crease_offset: float = 0.07,
                 hold_s: float = 1.5, h264: bool = True, lift: str = "squat") -> Path:
    """Write the annotated video. With h264=True (default) it is re-encoded so it
    plays in a browser; without ffmpeg it falls back to OpenCV's mp4v."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or seq.fps
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_name(out_path.stem + "_raw.mp4") if h264 else out_path
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    side = sig.attrs.get("side", "left")
    hold = int(hold_s * fps)
    thigh = float(sig["thigh_px"].iloc[0])

    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if f < len(seq):
            pts = seq.xy[f]
            for a, b in SKELETON:
                pa, pb = pts[KP[a]], pts[KP[b]]
                if np.isfinite(pa).all() and np.isfinite(pb).all():
                    cv2.line(img, tuple(int(v) for v in pa), tuple(int(v) for v in pb), WHITE, 2, cv2.LINE_AA)
            sides = ("left", "right") if side == "both" else (side,)
            for name in (f"{s}_{j}" for s in sides for j in ("hip", "knee", "ankle", "shoulder")):
                p = pts[KP[name]]
                if np.isfinite(p).all():
                    cv2.circle(img, (int(p[0]), int(p[1])), 6, AMBER, -1, cv2.LINE_AA)

        active = next((v for v in verdicts if v.start <= f <= v.end + hold), None)
        if active is not None and f < len(sig) and lift == "squat" and sig.attrs.get("view") != "front":
            # depth line: where the hip joint has to reach, drawn at the knee
            ky = int(sig["knee_y"].iloc[f] + crease_offset * thigh)
            kx = int(sig["knee_x"].iloc[f])
            cv2.line(img, (kx - int(thigh), ky), (kx + int(thigh), ky), AMBER, 2, cv2.LINE_AA)
        if active is not None and f < len(sig):
            if f >= active.bottom:
                y = _banner(cv2, img, f"Rep {active.rep + 1}: {active.label}", STATUS_COLOR[active.status])
                for c in active.cards:
                    cv2.rectangle(img, (12, y), (52, y + 55), CARD_BGR[c.value], -1)
                    cv2.rectangle(img, (12, y), (52, y + 55), BLACK, 2)
                    cv2.putText(img, c.value, (60, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
                    y += 65
                for chk in active.checks:
                    if chk.status in (Status.FAIL, Status.BORDERLINE, Status.UNKNOWN):
                        cv2.putText(img, chk.message[:70], (14, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, WHITE, 2, cv2.LINE_AA)
                        y += 28
            else:
                _banner(cv2, img, f"Rep {active.rep + 1}", (90, 90, 90))
        writer.write(img)
        f += 1
    cap.release()
    writer.release()
    if h264:
        if to_h264(raw_path, out_path):
            raw_path.unlink(missing_ok=True)
        else:
            raw_path.replace(out_path)
    return out_path
