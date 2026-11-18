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


def extract_pose(video_path: str | Path, model: str = DEFAULT_MODEL,
                 min_conf: float = 0.25, device: str | None = None,
                 verbose: bool = False) -> PoseSequence:
    """Run a YOLO pose model over a video and return the lifter's keypoints.

    The lifter is the tracked person with the largest total bounding-box area
    over the video (biggest and most present person in frame). This filters out
    spotters, referees and people walking past in the background.
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
    tracks: dict[int, dict[int, tuple[np.ndarray, np.ndarray, float]]] = {}
    n_frames = 0
    for f_idx, res in enumerate(net.track(source=str(video_path), stream=True,
                                          persist=True, conf=min_conf,
                                          device=device, verbose=verbose)):
        n_frames = f_idx + 1
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            continue
        kxy = res.keypoints.xy.cpu().numpy()          # (N, 17, 2)
        kconf = (res.keypoints.conf.cpu().numpy()
                 if res.keypoints.conf is not None else np.ones(kxy.shape[:2]))
        boxes = res.boxes.xywh.cpu().numpy()
        ids = (res.boxes.id.cpu().numpy().astype(int)
               if res.boxes.id is not None else np.arange(len(boxes)) - 1_000_000)
        for j, tid in enumerate(ids):
            area = float(boxes[j, 2] * boxes[j, 3])
            tracks.setdefault(int(tid), {})[f_idx] = (kxy[j], kconf[j], area)

    xy = np.full((n_frames, len(KEYPOINTS), 2), np.nan)
    conf = np.zeros((n_frames, len(KEYPOINTS)))
    if tracks:
        best = max(tracks, key=lambda k: sum(v[2] for v in tracks[k].values()))
        for f_idx, (k, c, _) in tracks[best].items():
            xy[f_idx], conf[f_idx] = k, c
        # YOLO returns (0, 0) for keypoints it could not place.
        missing = (xy[..., 0] == 0) & (xy[..., 1] == 0)
        xy[missing] = np.nan
        conf[missing] = 0.0
    return PoseSequence(xy=xy, conf=conf, fps=fps, width=width, height=height)
