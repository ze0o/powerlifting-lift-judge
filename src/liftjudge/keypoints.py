"""Pose extraction: video -> per-frame body keypoints for the lifter.

Uses an Ultralytics YOLO pose model (COCO 17 keypoints). The heavy imports
(ultralytics / torch) happen inside `extract_pose`, so the rest of the package
(signals, rules, tests) works without them installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# COCO-17 keypoint order used by YOLO pose models.
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINTS)}

# Skeleton edges for drawing.
SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

DEFAULT_MODEL = "yolo26n-pose.pt"


@dataclass
class PoseSequence:
    """Keypoints of one person across a video.

    xy:   (T, 17, 2) pixel coordinates, NaN where the person was not detected.
          Image coordinates: x to the right, y DOWN.
    conf: (T, 17) keypoint confidence in [0, 1], 0 where missing.
    """

    xy: np.ndarray
    conf: np.ndarray
    fps: float
    width: int
    height: int

    def __len__(self) -> int:
        return self.xy.shape[0]

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self)) / self.fps

    def point(self, name: str) -> np.ndarray:
        """(T, 2) trajectory of one keypoint."""
        return self.xy[:, KP[name], :]

    def to_frame(self) -> pd.DataFrame:
        """Wide table: one row per frame, columns <kp>_x, <kp>_y, <kp>_conf."""
        data = {"frame": np.arange(len(self)), "time_s": self.t}
        for name, i in KP.items():
            data[f"{name}_x"] = self.xy[:, i, 0]
            data[f"{name}_y"] = self.xy[:, i, 1]
            data[f"{name}_conf"] = self.conf[:, i]
        df = pd.DataFrame(data)
        df.attrs.update(fps=self.fps, width=self.width, height=self.height)
        return df

    @classmethod
    def from_frame(cls, df: pd.DataFrame, fps: float | None = None,
                   width: int | None = None, height: int | None = None) -> "PoseSequence":
        n = len(df)
        xy = np.full((n, len(KEYPOINTS), 2), np.nan)
        conf = np.zeros((n, len(KEYPOINTS)))
        for name, i in KP.items():
            xy[:, i, 0] = df[f"{name}_x"].to_numpy()
            xy[:, i, 1] = df[f"{name}_y"].to_numpy()
            conf[:, i] = df[f"{name}_conf"].fillna(0).to_numpy()
        return cls(
            xy=xy, conf=conf,
            fps=fps or df.attrs.get("fps", 30.0),
            width=width or df.attrs.get("width", 0),
            height=height or df.attrs.get("height", 0),
        )

    def save_csv(self, path: str | Path) -> None:
        df = self.to_frame()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Store video metadata in a one-line header comment.
        with open(path, "w", newline="") as f:
            f.write(f"# fps={self.fps},width={self.width},height={self.height}\n")
            df.to_csv(f, index=False)

    @classmethod
    def load_csv(cls, path: str | Path) -> "PoseSequence":
        with open(path) as f:
            header = f.readline().lstrip("#").strip()
        meta = dict(kv.split("=") for kv in header.split(","))
        df = pd.read_csv(path, comment="#")
        return cls.from_frame(df, fps=float(meta["fps"]),
                             width=int(meta["width"]), height=int(meta["height"]))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap of two boxes given as (x1, y1, x2, y2)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def select_lifter(frames: list[list[dict]], width: int, min_iou: float = 0.3) -> list[dict | None]:
    """Pick the lifter in every frame from all detected people.

    frames: per frame, a list of detections {"id", "box" (x1,y1,x2,y2), "kxy", "kconf"}.

    1. The lifter is the track with the highest  sum over frames of
       box area x centrality  - the biggest person, most often near the middle of
       the frame. Spotters stand at the sides, referees and the crowd are smaller.
    2. Spotters crowd in at the bottom of a squat and the tracker can lose the
       lifter or give them a new id. Whenever the chosen track is missing, take the
       detection that overlaps most with the lifter's last box (stitching).
    """
    score: dict[int, float] = {}
    for dets in frames:
        for d in dets:
            x1, _, x2, _ = d["box"]
            centre = 1.0 - abs((x1 + x2) / 2 - width / 2) / (width / 2)
            area = (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
            score[d["id"]] = score.get(d["id"], 0.0) + area * max(centre, 0.2)
    if not score:
        return [None] * len(frames)
    best = max(score, key=score.get)

    def follow(order: list[int], track: int) -> dict[int, dict]:
        picks, last = {}, None
        for i in order:
            dets = frames[i]
            pick = next((d for d in dets if d["id"] == track), None)
            if pick is None and last is not None and dets:
                cand = max(dets, key=lambda d: _iou(d["box"], last["box"]))
                if _iou(cand["box"], last["box"]) >= min_iou:
                    pick, track = cand, cand["id"]   # follow the new id from here on
            if pick is not None:
                picks[i], last = pick, pick
        return picks

    # start where the chosen track first appears, follow it forwards and backwards
    first = next(i for i, dets in enumerate(frames) if any(d["id"] == best for d in dets))
    picks = follow(list(range(first, len(frames))), best)
    picks.update(follow(list(range(first, -1, -1)), best))
    return [picks.get(i) for i in range(len(frames))]


def extract_pose(video_path: str | Path, model: str = DEFAULT_MODEL,
                 min_conf: float = 0.25, device: str | None = None,
                 verbose: bool = False) -> PoseSequence:
    """Run a YOLO pose model over a video and return the lifter's keypoints.

    All people are tracked; `select_lifter` picks the lifter and follows them
    through id switches (see there).
    """
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    net = YOLO(model)
    frames: list[list[dict]] = []
    for res in net.track(source=str(video_path), stream=True, persist=True, conf=min_conf,
                         device=device, verbose=verbose):
        dets: list[dict] = []
        if res.keypoints is not None and res.boxes is not None and len(res.boxes):
            kxy = res.keypoints.xy.cpu().numpy()          # (N, 17, 2)
            kconf = (res.keypoints.conf.cpu().numpy()
                     if res.keypoints.conf is not None else np.ones(kxy.shape[:2]))
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = (res.boxes.id.cpu().numpy().astype(int)
                   if res.boxes.id is not None else -1_000_000 - np.arange(len(boxes)) - 1000 * len(frames))
            for j, tid in enumerate(ids):
                dets.append({"id": int(tid), "box": boxes[j], "kxy": kxy[j], "kconf": kconf[j]})
        frames.append(dets)

    n_frames = len(frames)
    xy = np.full((n_frames, len(KEYPOINTS), 2), np.nan)
    conf = np.zeros((n_frames, len(KEYPOINTS)))
    for f_idx, d in enumerate(select_lifter(frames, width)):
        if d is not None:
            xy[f_idx], conf[f_idx] = d["kxy"], d["kconf"]
    # YOLO returns (0, 0) for keypoints it could not place.
    missing = (xy[..., 0] == 0) & (xy[..., 1] == 0)
    xy[missing] = np.nan
    conf[missing] = 0.0
    return PoseSequence(xy=xy, conf=conf, fps=fps, width=width, height=height)
