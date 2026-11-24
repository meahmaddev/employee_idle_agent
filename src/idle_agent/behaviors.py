"""Per-frame behavior scores from pose keypoints and zone polygons.

These sit beside the motion score. Motion alone treats a phone, a conversation,
and a person standing in the workstation as ordinary activity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from idle_agent.detect import PersonDetection
from idle_agent.zones import Zone, point_in_polygon

# COCO-17 indices used by YOLOv8-pose
NOSE = 0
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12


@dataclass
class ZoneSignals:
    """What the primary occupant is doing this tick. All flags are instantaneous."""

    seat_occupied: bool = False
    phone_score: float = 0.0
    phone: bool = False
    hands_known: bool = False
    working: bool = False
    interacting: bool = False
    interaction_track_id: Optional[str] = None
    interaction_dist: float = 0.0


def _cfg(block: dict, key: str, default):
    return block.get(key, default) if isinstance(block, dict) else default


class BehaviorScorer:
    def __init__(self, behaviors: Optional[dict] = None):
        cfg = behaviors or {}
        self.kp_conf = float(cfg.get("keypoint_confidence_min", 0.35))
        phone = cfg.get("phone") or {}
        talk = cfg.get("interacting") or {}
        self.phone_enabled = bool(phone.get("enabled", True))
        self.wrist_ear_ratio = float(phone.get("wrist_ear_ratio", 0.65))
        self.face_ratio = float(phone.get("face_ratio", 0.75))
        self.phone_min_score = float(phone.get("min_score", 0.40))
        self.talk_enabled = bool(talk.get("enabled", True))
        self.max_center_dist = float(talk.get("max_center_dist", 0.16))
        self.close_dist = float(talk.get("close_dist", 0.07))
        self.facing_dot = float(talk.get("facing_dot", 0.15))

    def score(
        self,
        person: Optional[PersonDetection],
        people: Sequence[PersonDetection],
        zone: Zone,
        frame_w: int,
        frame_h: int,
        occupied: bool,
    ) -> ZoneSignals:
        if person is None or not occupied:
            return ZoneSignals()

        phone_score = (
            phone_score_of(person, self.kp_conf, self.wrist_ear_ratio, self.face_ratio)
            if self.phone_enabled
            else 0.0
        )
        hands_known, working = wrists_in_work(
            person, zone, frame_w, frame_h, self.kp_conf
        )
        seat = hips_in_seat(person, zone, frame_w, frame_h, self.kp_conf)
        interacting = False
        other_id = None
        dist = 0.0
        if self.talk_enabled:
            interacting, other_id, dist = nearest_interaction(
                person,
                people,
                self.kp_conf,
                self.max_center_dist,
                self.close_dist,
                self.facing_dot,
            )
        return ZoneSignals(
            seat_occupied=seat,
            phone_score=phone_score,
            phone=phone_score >= self.phone_min_score,
            hands_known=hands_known,
            working=working,
            interacting=interacting,
            interaction_track_id=other_id,
            interaction_dist=dist,
        )


def _kp(
    keypoints: Optional[np.ndarray],
    index: int,
    conf_min: float,
) -> Optional[Tuple[float, float]]:
    if keypoints is None or index >= len(keypoints):
        return None
    x = float(keypoints[index, 0])
    y = float(keypoints[index, 1])
    conf = float(keypoints[index, 2]) if keypoints.shape[1] >= 3 else 1.0
    if conf < conf_min:
        return None
    if x <= 1.0 and y <= 1.0:
        # Missing joints are often left at the origin.
        return None
    return x, y


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _shoulder_scale(keypoints: Optional[np.ndarray], conf_min: float) -> float:
    left = _kp(keypoints, LEFT_SHOULDER, conf_min)
    right = _kp(keypoints, RIGHT_SHOULDER, conf_min)
    if left is None or right is None:
        return 0.0
    return _dist(left, right)


def phone_score_of(
    person: PersonDetection,
    conf_min: float,
    wrist_ear_ratio: float,
    face_ratio: float,
) -> float:
    """1 when a wrist is at the ear or held in front of the face. 0 otherwise."""
    kps = person.keypoints
    scale = _shoulder_scale(kps, conf_min)
    if scale < 8.0:
        return 0.0
    ears = [p for p in (_kp(kps, LEFT_EAR, conf_min), _kp(kps, RIGHT_EAR, conf_min), _kp(kps, NOSE, conf_min)) if p]
    wrists = [p for p in (_kp(kps, LEFT_WRIST, conf_min), _kp(kps, RIGHT_WRIST, conf_min)) if p]
    if not wrists:
        return 0.0

    ear_score = 0.0
    if ears and wrist_ear_ratio > 0:
        best = min(_dist(w, e) / scale for w in wrists for e in ears)
        if best <= wrist_ear_ratio:
            ear_score = (wrist_ear_ratio - best) / wrist_ear_ratio

    face_score = 0.0
    nose = _kp(kps, NOSE, conf_min)
    left_s = _kp(kps, LEFT_SHOULDER, conf_min)
    right_s = _kp(kps, RIGHT_SHOULDER, conf_min)
    if nose is not None and left_s is not None and right_s is not None and face_ratio > 0:
        mid = ((left_s[0] + right_s[0]) / 2.0, (left_s[1] + right_s[1]) / 2.0)
        for wrist in wrists:
            ratio = _dist(wrist, nose) / scale
            # In front of the face or upper chest, not down at the keyboard.
            in_front = wrist[1] >= nose[1] - 0.25 * scale and wrist[1] <= mid[1] + 0.35 * scale
            near_column = abs(wrist[0] - mid[0]) <= 1.1 * scale
            if in_front and near_column and ratio <= face_ratio:
                face_score = max(face_score, (face_ratio - ratio) / face_ratio)
    return float(max(0.0, min(1.0, max(ear_score, face_score))))


def _norm(px: float, py: float, frame_w: int, frame_h: int) -> Tuple[float, float]:
    return px / max(frame_w, 1), py / max(frame_h, 1)


def hips_in_seat(
    person: PersonDetection,
    zone: Zone,
    frame_w: int,
    frame_h: int,
    conf_min: float,
) -> bool:
    polygon = zone.seat_polygon or zone.polygon
    hips = [
        p
        for p in (
            _kp(person.keypoints, LEFT_HIP, conf_min),
            _kp(person.keypoints, RIGHT_HIP, conf_min),
        )
        if p
    ]
    if hips:
        mx = sum(p[0] for p in hips) / len(hips)
        my = sum(p[1] for p in hips) / len(hips)
        x, y = _norm(mx, my, frame_w, frame_h)
        return point_in_polygon(x, y, polygon)
    return point_in_polygon(person.seat_norm[0], person.seat_norm[1], polygon)


def wrists_in_work(
    person: PersonDetection,
    zone: Zone,
    frame_w: int,
    frame_h: int,
    conf_min: float,
) -> Tuple[bool, bool]:
    """Returns (hands_known, working). Unknown hands are not treated as idle."""
    polygon = zone.work_polygon or zone.polygon
    wrists = [
        p
        for p in (
            _kp(person.keypoints, LEFT_WRIST, conf_min),
            _kp(person.keypoints, RIGHT_WRIST, conf_min),
        )
        if p
    ]
    if not wrists:
        return False, False
    working = False
    for px, py in wrists:
        x, y = _norm(px, py, frame_w, frame_h)
        if point_in_polygon(x, y, polygon):
            working = True
            break
    return True, working


def _facing(a: PersonDetection, b: PersonDetection, conf_min: float, min_dot: float) -> bool:
    def aim(person: PersonDetection):
        kps = person.keypoints
        left = _kp(kps, LEFT_SHOULDER, conf_min)
        right = _kp(kps, RIGHT_SHOULDER, conf_min)
        nose = _kp(kps, NOSE, conf_min)
        if left is None or right is None or nose is None:
            return None
        mid = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
        return mid, nose

    def toward(src: PersonDetection, dst_mid: Tuple[float, float]) -> Optional[float]:
        aimed = aim(src)
        if aimed is None:
            return None
        mid, nose = aimed
        fx, fy = nose[0] - mid[0], nose[1] - mid[1]
        tx, ty = dst_mid[0] - mid[0], dst_mid[1] - mid[1]
        fn = float(np.hypot(fx, fy)) or 1.0
        tn = float(np.hypot(tx, ty)) or 1.0
        return (fx / fn) * (tx / tn) + (fy / fn) * (ty / tn)

    b_aim = aim(b)
    a_aim = aim(a)
    if a_aim is None or b_aim is None:
        return False
    a_dot = toward(a, b_aim[0])
    b_dot = toward(b, a_aim[0])
    return (a_dot is not None and a_dot >= min_dot) or (b_dot is not None and b_dot >= min_dot)


def nearest_interaction(
    person: PersonDetection,
    people: Sequence[PersonDetection],
    conf_min: float,
    max_dist: float,
    close_dist: float,
    facing_dot: float,
) -> Tuple[bool, Optional[str], float]:
    best: Optional[PersonDetection] = None
    best_d = 1e9
    for other in people:
        if other.track_id == person.track_id:
            continue
        dx = other.center_norm[0] - person.center_norm[0]
        dy = other.center_norm[1] - person.center_norm[1]
        dist = float(np.hypot(dx, dy))
        if dist < best_d:
            best_d = dist
            best = other
    if best is None:
        return False, None, 0.0
    if best_d <= close_dist:
        return True, best.track_id, best_d
    if best_d <= max_dist and _facing(person, best, conf_min, facing_dot):
        return True, best.track_id, best_d
    return False, None, best_d
