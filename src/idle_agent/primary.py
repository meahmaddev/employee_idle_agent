"""Sticky primary occupant per zone — ignore other people in the polygon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from idle_agent.detect import PersonDetection
from idle_agent.zones import Zone, person_in_zone


def _center_dist(a: PersonDetection, b: PersonDetection) -> float:
    return float(
        ((a.center_norm[0] - b.center_norm[0]) ** 2 + (a.center_norm[1] - b.center_norm[1]) ** 2)
        ** 0.5
    )


def _bbox_iou(a: PersonDetection, b: PersonDetection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class _PrimaryState:
    track_id: Optional[str] = None
    last_seen_t: Optional[float] = None
    last_person: Optional[PersonDetection] = None


@dataclass
class ZonePrimaryTracker:
    """
    Lock onto one logical occupant per zone.

    - Idle/away use only the primary person; others in the polygon are ignored.
    - If ByteTrack changes ID, re-attach to a nearby/overlapping detection.
    - Clear primary only after lost_grace with nobody matching the last seat position.
    """

    lost_grace_seconds: float = 3.0
    assign_dwell_seconds: float = 0.5
    match_center_dist: float = 0.08
    match_iou: float = 0.25
    enabled: bool = True
    _states: Dict[str, _PrimaryState] = field(default_factory=dict)
    _candidate_id: Dict[str, Optional[str]] = field(default_factory=dict)
    _candidate_since: Dict[str, Optional[float]] = field(default_factory=dict)

    def _state(self, zone_id: str) -> _PrimaryState:
        if zone_id not in self._states:
            self._states[zone_id] = _PrimaryState()
        return self._states[zone_id]

    def primary_ids(self) -> Dict[str, Optional[str]]:
        return {zid: st.track_id for zid, st in self._states.items()}

    def update(
        self,
        zone: Zone,
        t: float,
        people: List[PersonDetection],
    ) -> Tuple[bool, Optional[PersonDetection], Optional[str]]:
        # Seat or body center inside polygon (leaning / seated desks)
        in_zone = [
            p
            for p in people
            if person_in_zone(p.seat_norm[0], p.seat_norm[1], zone)
            or person_in_zone(p.center_norm[0], p.center_norm[1], zone)
        ]

        if not self.enabled:
            if not in_zone:
                return False, None, None
            best = max(in_zone, key=lambda p: p.confidence)
            return True, best, best.track_id

        st = self._state(zone.id)

        if st.track_id is not None:
            matched = self._match_primary(st, in_zone)
            if matched is not None:
                # Keep logical primary even if track_id string changed
                st.track_id = matched.track_id
                st.last_seen_t = t
                st.last_person = matched
                self._candidate_id[zone.id] = None
                self._candidate_since[zone.id] = None
                return True, matched, st.track_id

            if st.last_seen_t is not None and (t - st.last_seen_t) <= self.lost_grace_seconds:
                return True, st.last_person, st.track_id

            # Truly gone
            st.track_id = None
            st.last_person = None
            st.last_seen_t = None

        if not in_zone:
            self._candidate_id[zone.id] = None
            self._candidate_since[zone.id] = None
            return False, None, None

        best = self._pick_candidate(zone, in_zone)
        cand_id = self._candidate_id.get(zone.id)
        cand_since = self._candidate_since.get(zone.id)
        if cand_id != best.track_id:
            self._candidate_id[zone.id] = best.track_id
            self._candidate_since[zone.id] = t
            # Soft-occupy during short dwell so away timer does not thrash
            if self.assign_dwell_seconds <= 0:
                st.track_id = best.track_id
                st.last_seen_t = t
                st.last_person = best
                return True, best, st.track_id
            return True, best, best.track_id

        if cand_since is not None and (t - cand_since) >= self.assign_dwell_seconds:
            st.track_id = best.track_id
            st.last_seen_t = t
            st.last_person = best
            self._candidate_id[zone.id] = None
            self._candidate_since[zone.id] = None
            return True, best, st.track_id

        # Dwelling: already treating as occupied with candidate above on first frame;
        # continue occupying with candidate so idle can accumulate on the seated person.
        return True, best, best.track_id

    def _match_primary(
        self,
        st: _PrimaryState,
        in_zone: List[PersonDetection],
    ) -> Optional[PersonDetection]:
        if not in_zone or st.last_person is None:
            # Fall back to exact track id
            for p in in_zone:
                if p.track_id == st.track_id:
                    return p
            return None

        # Exact ID first
        for p in in_zone:
            if p.track_id == st.track_id:
                return p

        # Re-attach by position / IoU (track ID flicker)
        best: Optional[PersonDetection] = None
        best_score = 1e9
        for p in in_zone:
            dist = _center_dist(p, st.last_person)
            iou = _bbox_iou(p, st.last_person)
            if dist <= self.match_center_dist or iou >= self.match_iou:
                score = dist - iou
                if score < best_score:
                    best_score = score
                    best = p
        return best

    def _pick_candidate(self, zone: Zone, in_zone: List[PersonDetection]) -> PersonDetection:
        xs = [p[0] for p in zone.polygon]
        ys = [p[1] for p in zone.polygon]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        def score(p: PersonDetection) -> tuple:
            dx = p.center_norm[0] - cx
            dy = p.center_norm[1] - cy
            dist2 = dx * dx + dy * dy
            x1, y1, x2, y2 = p.bbox_xyxy
            area = max(0.0, (x2 - x1) * (y2 - y1))
            return (dist2, -p.confidence, -area)

        return min(in_zone, key=score)
