"""YOLOv8 pose + ByteTrack person detection and motion scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PersonDetection:
    track_id: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    center_norm: Tuple[float, float]
    # Lower point of the box (seat/feet) — more stable for zone tests when leaning
    seat_norm: Tuple[float, float] = (0.0, 0.0)
    keypoints: Optional[np.ndarray] = None  # (K, 3) x, y, confidence
    motion_score: float = 0.0
    phone_score: float = 0.0


@dataclass
class Detector:
    model_name: str = "yolov8s-pose.pt"
    conf_min: float = 0.45
    device: Optional[str] = None
    imgsz: int = 640

    _model: object = field(default=None, init=False, repr=False)
    _prev_keypoints: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.model_name)
        # Warm device selection: ultralytics picks CUDA if available
        if self.device:
            self._model.to(self.device)

    def infer(self, frame: np.ndarray) -> List[PersonDetection]:
        if self._model is None:
            self.load()
        assert self._model is not None

        h, w = frame.shape[:2]
        results = self._model.track(
            frame,
            persist=True,
            conf=self.conf_min,
            imgsz=self.imgsz,
            classes=[0],  # person
            verbose=False,
            tracker="bytetrack.yaml",
        )

        detections: List[PersonDetection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        keypoints_obj = getattr(result, "keypoints", None)
        ids = boxes.id
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            if conf < self.conf_min:
                continue
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            x1, y1, x2, y2 = xyxy
            cx = ((x1 + x2) / 2.0) / max(w, 1)
            cy = ((y1 + y2) / 2.0) / max(h, 1)
            # Seat / lower-torso point (better for seated desk zones)
            sx = ((x1 + x2) / 2.0) / max(w, 1)
            sy = (y1 * 0.25 + y2 * 0.75) / max(h, 1)

            track_raw = None
            if ids is not None:
                track_raw = ids[i]
            track_id = f"t-{int(track_raw.item())}" if track_raw is not None else f"d-{i}"

            kps = None
            if keypoints_obj is not None and keypoints_obj.xy is not None:
                xy = keypoints_obj.xy[i].cpu().numpy()
                kp_conf_tensor = getattr(keypoints_obj, "conf", None)
                if kp_conf_tensor is not None:
                    kp_conf = kp_conf_tensor[i].cpu().numpy().reshape(-1, 1)
                    kps = np.concatenate([xy, kp_conf], axis=1)
                else:
                    kps = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float32)], axis=1)

            motion = self._motion_score(track_id, kps, (x1, y1, x2, y2), w, h)
            detections.append(
                PersonDetection(
                    track_id=track_id,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_norm=(cx, cy),
                    seat_norm=(sx, sy),
                    keypoints=kps,
                    motion_score=motion,
                )
            )
        return detections

    def _motion_score(
        self,
        track_id: str,
        keypoints: Optional[np.ndarray],
        bbox: Tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> float:
        """Average normalized joint displacement; bbox-center fallback."""
        if keypoints is not None and keypoints.size >= 4:
            curr = keypoints[:, :2].astype(np.float32)
            prev = self._prev_keypoints.get(track_id)
            self._prev_keypoints[track_id] = curr.copy()
            if prev is None or prev.shape != curr.shape:
                return 0.0
            # Normalize by frame diagonal
            diag = float(np.hypot(width, height)) or 1.0
            deltas = np.linalg.norm(curr - prev, axis=1)
            # Ignore near-zero keypoints (often missing joints at 0,0)
            valid = (np.linalg.norm(curr, axis=1) > 1.0) & (np.linalg.norm(prev, axis=1) > 1.0)
            if not np.any(valid):
                return 0.0
            return float(np.mean(deltas[valid]) / diag)

        # Fallback: store bbox center as pseudo-keypoint
        x1, y1, x2, y2 = bbox
        curr = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], dtype=np.float32)
        prev = self._prev_keypoints.get(track_id)
        self._prev_keypoints[track_id] = curr.copy()
        if prev is None:
            return 0.0
        diag = float(np.hypot(width, height)) or 1.0
        return float(np.linalg.norm(curr - prev) / diag)
