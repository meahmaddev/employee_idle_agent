"""Video ingest: local file (cam-01 recordings) and live RTSP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray
    index: int
    timestamp_seconds: float
    source_fps: float


class VideoSource:
    """OpenCV-backed frame iterator for file or RTSP URLs."""

    def __init__(self, uri: str, source_type: str = "file", max_frames: Optional[int] = None):
        self.uri = uri
        self.source_type = source_type
        self.max_frames = max_frames
        self._cap: Optional[cv2.VideoCapture] = None
        self.source_fps: float = 25.0
        self.frame_width: int = 0
        self.frame_height: int = 0

    def open(self) -> None:
        if self.source_type == "file":
            path = Path(self.uri)
            if not path.is_file():
                raise FileNotFoundError(f"Video file not found: {path.resolve()}")
            self._cap = cv2.VideoCapture(str(path))
        elif self.source_type == "rtsp":
            self._cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)
        else:
            raise ValueError(f"Unsupported source_type: {self.source_type}")

        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.uri}")

        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.source_fps = fps if fps > 1.0 else 25.0
        self.frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def frames(self) -> Iterator[FramePacket]:
        if self._cap is None:
            self.open()
        assert self._cap is not None

        index = 0
        while True:
            if self.max_frames is not None and index >= self.max_frames:
                break
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            ts = index / self.source_fps
            yield FramePacket(
                frame=frame,
                index=index,
                timestamp_seconds=ts,
                source_fps=self.source_fps,
            )
            index += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
