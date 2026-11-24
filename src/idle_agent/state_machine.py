"""Per-zone idle / away state machine."""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

import yaml


def _merge_dict(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_rules(idle_rules_path: str | Path, demo_overrides_path: Optional[str | Path] = None) -> dict:
    rules = yaml.safe_load(Path(idle_rules_path).read_text(encoding="utf-8")) or {}
    if demo_overrides_path and Path(demo_overrides_path).is_file():
        overrides = yaml.safe_load(Path(demo_overrides_path).read_text(encoding="utf-8")) or {}
        ignore_shifts = bool(overrides.pop("ignore_shifts", False))
        for drop in ("version", "demo"):
            overrides.pop(drop, None)
        rules = _merge_dict(rules, overrides)
        rules["ignore_shifts"] = ignore_shifts
    return rules


@dataclass
class IdleEvent:
    event_type: str
    camera_id: str
    zone_id: str
    track_id: Optional[str]
    started_at: str
    ended_at: Optional[str]
    duration_seconds: float
    confidence: float
    motion_score: float
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    snapshot_url: Optional[str] = None
    site_timezone: str = "Asia/Karachi"
    reason: Optional[str] = None
    phone_score: Optional[float] = None
    seat_occupied: Optional[bool] = None
    working: Optional[bool] = None
    interacting: Optional[bool] = None

    def to_dict(self) -> dict:
        idem = f"{self.camera_id}:{self.zone_id}:{self.event_type}:{self.started_at}"
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "track_id": self.track_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "confidence": round(self.confidence, 3),
            "snapshot_url": self.snapshot_url,
            "idempotency_key": idem,
            "site_timezone": self.site_timezone,
            "motion_score": round(self.motion_score, 4),
            "reason": self.reason,
            "phone_score": None if self.phone_score is None else round(self.phone_score, 3),
            "seat_occupied": self.seat_occupied,
            "working": self.working,
            "interacting": self.interacting,
        }


@dataclass
class _FlagRuntime:
    """Confirm / grace timer for one behavior (phone, talking, off-seat, not working)."""

    active: bool = False
    candidate_since: Optional[float] = None
    gap_since: Optional[float] = None
    episode_started_at: Optional[float] = None
    episode_track_id: Optional[str] = None
    episode_confidence: float = 0.0
    resume_since: Optional[float] = None
    escalate_emitted: bool = False


@dataclass
class _ZoneRuntime:
    state: str = "UNKNOWN"
    motion_window: Deque[float] = field(default_factory=deque)
    candidate_since: Optional[float] = None
    episode_started_at: Optional[float] = None
    episode_track_id: Optional[str] = None
    episode_confidence: float = 0.0
    last_motion: float = 0.0
    empty_since: Optional[float] = None
    resume_since: Optional[float] = None
    escalate_emitted: bool = False
    away_started_emitted: bool = False
    phone: _FlagRuntime = field(default_factory=_FlagRuntime)
    interacting: _FlagRuntime = field(default_factory=_FlagRuntime)
    off_seat: _FlagRuntime = field(default_factory=_FlagRuntime)
    not_working: _FlagRuntime = field(default_factory=_FlagRuntime)
    last_phone_score: float = 0.0
    last_seat_occupied: bool = False
    last_working: bool = False
    last_interacting: bool = False


class ZoneStateMachine:
    def __init__(
        self,
        camera_id: str,
        zone_ids: List[str],
        rules: dict,
        clip_start: Optional[datetime] = None,
    ):
        self.camera_id = camera_id
        self.rules = rules
        self.tz_name = rules.get("timezone", "Asia/Karachi")
        # Wall-clock start of the recording; video t=0 maps to this instant.
        self.clip_start = clip_start or datetime(1970, 1, 1, tzinfo=timezone.utc)
        eval_cfg = rules.get("evaluation", {})
        motion_cfg = rules.get("motion", {})
        thr = rules.get("thresholds", {})

        self.motion_window_seconds = float(eval_cfg.get("motion_window_seconds", 3))
        self.idle_threshold = float(motion_cfg.get("idle_threshold", 0.08))
        self.active_threshold = float(motion_cfg.get("active_threshold", 0.12))
        self.idle_confirm = float(thr.get("idle_confirm_seconds", 180))
        self.away_confirm = float(thr.get("away_confirm_seconds", 300))
        self.escalate_seconds = float(thr.get("escalate_seconds", 900))
        self.end_grace = float(thr.get("end_grace_seconds", 10))
        self.debounce_leave = float(eval_cfg.get("debounce_leave_seconds", 5))
        beh = rules.get("behaviors") or {}
        self.behavior_gap = float(beh.get("gap_grace_seconds", 2.0))
        phone = beh.get("phone") or {}
        talk = beh.get("interacting") or {}
        off = beh.get("off_seat") or {}
        idle_hands = beh.get("not_working") or {}
        self.phone_enabled = bool(phone.get("enabled", True))
        self.phone_min_score = float(phone.get("min_score", 0.40))
        self.phone_clear_score = float(phone.get("clear_score", 0.20))
        self.phone_confirm = float(phone.get("confirm_seconds", 20))
        self.phone_grace = float(phone.get("end_grace_seconds", 5))
        self.talk_enabled = bool(talk.get("enabled", True))
        self.talk_confirm = float(talk.get("confirm_seconds", 15))
        self.talk_grace = float(talk.get("end_grace_seconds", 5))
        self.off_seat_enabled = bool(off.get("enabled", True))
        self.off_seat_confirm = float(off.get("confirm_seconds", 45))
        self.off_seat_grace = float(off.get("end_grace_seconds", 5))
        self.not_working_enabled = bool(idle_hands.get("enabled", True))
        self.not_working_confirm = float(idle_hands.get("confirm_seconds", 60))
        self.not_working_grace = float(idle_hands.get("end_grace_seconds", 8))
        self._zones: Dict[str, _ZoneRuntime] = {zid: _ZoneRuntime() for zid in zone_ids}

    def _iso(self, t_seconds: float) -> str:
        wall = self.clip_start + timedelta(seconds=float(t_seconds))
        return wall.isoformat()

    def update(
        self,
        zone_id: str,
        t: float,
        occupied: bool,
        motion_score: float,
        track_id: Optional[str],
        confidence: float,
        signals: object = None,
    ) -> List[IdleEvent]:
        rt = self._zones[zone_id]
        events: List[IdleEvent] = []

        rt.motion_window.append(motion_score if occupied else 0.0)
        max_len = max(3, int(self.motion_window_seconds))
        while len(rt.motion_window) > max_len:
            rt.motion_window.popleft()
        avg_motion = float(sum(rt.motion_window) / max(len(rt.motion_window), 1))
        rt.last_motion = avg_motion
        low_motion = avg_motion < self.idle_threshold
        high_motion = avg_motion >= self.active_threshold
        self._remember_signals(rt, signals)

        # Hands on the keyboard/counter are work, even when the joints barely move.
        if signals is not None and getattr(signals, "hands_known", False) and getattr(signals, "working", False):
            if not getattr(signals, "phone", False) and not getattr(signals, "interacting", False):
                low_motion = False
                high_motion = True

        if occupied:
            rt.empty_since = None

            if rt.state == "AWAY":
                end_type = "AWAY_ENDED"
                events.extend(self._close_episode(rt, zone_id, t, end_type))
                rt.state = "ACTIVE" if high_motion or not low_motion else "IDLE"
                rt.away_started_emitted = False
                rt.escalate_emitted = False
                if rt.state == "IDLE":
                    rt.episode_started_at = t
                    rt.episode_track_id = track_id
                    rt.episode_confidence = confidence
                    events.append(
                        self._emit("IDLE_STARTED", zone_id, t, track_id, confidence, avg_motion)
                    )
                rt.candidate_since = None
                rt.resume_since = None
                events.extend(self._update_behaviors(rt, zone_id, t, True, track_id, confidence, avg_motion, signals))
                return events

            if rt.state in ("UNKNOWN", "ACTIVE"):
                if low_motion:
                    if rt.candidate_since is None:
                        rt.candidate_since = t
                    if t - rt.candidate_since >= self.idle_confirm:
                        rt.state = "IDLE"
                        rt.episode_started_at = rt.candidate_since
                        rt.episode_track_id = track_id
                        rt.episode_confidence = confidence
                        rt.escalate_emitted = False
                        events.append(
                            self._emit(
                                "IDLE_STARTED",
                                zone_id,
                                rt.candidate_since,
                                track_id,
                                confidence,
                                avg_motion,
                                duration=t - rt.candidate_since,
                            )
                        )
                        rt.candidate_since = None
                elif high_motion:
                    # Only clear idle countdown on clear activity (hysteresis band keeps candidate)
                    rt.candidate_since = None
                    if rt.state == "UNKNOWN":
                        rt.state = "ACTIVE"
                else:
                    # Between idle and active thresholds: keep counting toward idle
                    if rt.candidate_since is None:
                        rt.candidate_since = t
                    if rt.state == "UNKNOWN":
                        rt.state = "ACTIVE"

            elif rt.state == "IDLE":
                if high_motion:
                    if rt.resume_since is None:
                        rt.resume_since = t
                    if t - rt.resume_since >= self.end_grace:
                        events.extend(self._close_episode(rt, zone_id, t, "IDLE_ENDED"))
                        rt.state = "ACTIVE"
                        rt.resume_since = None
                        rt.escalate_emitted = False
                else:
                    rt.resume_since = None
                    events.extend(self._maybe_escalate(rt, zone_id, t, "IDLE_ESCALATED", avg_motion))

        else:
            if rt.empty_since is None:
                rt.empty_since = t
            empty_for = t - rt.empty_since

            if empty_for < self.debounce_leave:
                # Brief dropout — do not reset idle candidate timer
                return events

            if rt.state == "IDLE":
                events.extend(self._close_episode(rt, zone_id, t, "IDLE_ENDED"))
                rt.state = "AWAY"
                rt.episode_started_at = rt.empty_since
                rt.episode_track_id = None
                rt.episode_confidence = 0.0
                rt.away_started_emitted = False
                rt.escalate_emitted = False

            if rt.state in ("UNKNOWN", "ACTIVE") and empty_for >= self.away_confirm:
                rt.state = "AWAY"
                rt.episode_started_at = rt.empty_since
                rt.episode_track_id = None
                rt.away_started_emitted = False
                rt.escalate_emitted = False

            if rt.state == "AWAY" and empty_for >= self.away_confirm:
                if not rt.away_started_emitted:
                    events.append(
                        self._emit(
                            "AWAY_STARTED",
                            zone_id,
                            rt.episode_started_at or rt.empty_since or t,
                            None,
                            0.0,
                            0.0,
                            duration=empty_for,
                        )
                    )
                    rt.away_started_emitted = True
                events.extend(self._maybe_escalate(rt, zone_id, t, "AWAY_ESCALATED", 0.0))

            # Only clear idle candidate after a real leave (already past debounce)
            rt.candidate_since = None
            rt.resume_since = None

        events.extend(
            self._update_behaviors(
                rt, zone_id, t, occupied, track_id, confidence, avg_motion, signals
            )
        )
        return events

    def _maybe_escalate(
        self,
        rt: _ZoneRuntime,
        zone_id: str,
        t: float,
        event_type: str,
        motion_score: float,
    ) -> List[IdleEvent]:
        if rt.escalate_emitted or rt.episode_started_at is None:
            return []
        if t - rt.episode_started_at < self.escalate_seconds:
            return []
        rt.escalate_emitted = True
        return [
            self._emit(
                event_type,
                zone_id,
                rt.episode_started_at,
                rt.episode_track_id,
                rt.episode_confidence,
                motion_score,
                duration=t - rt.episode_started_at,
            )
        ]

    def _remember_signals(self, rt: _ZoneRuntime, signals: object) -> None:
        if signals is None:
            rt.last_phone_score = 0.0
            rt.last_seat_occupied = False
            rt.last_working = False
            rt.last_interacting = False
            return
        rt.last_phone_score = float(getattr(signals, "phone_score", 0.0) or 0.0)
        rt.last_seat_occupied = bool(getattr(signals, "seat_occupied", False))
        rt.last_working = bool(getattr(signals, "working", False))
        rt.last_interacting = bool(getattr(signals, "interacting", False))

    def _update_behaviors(
        self,
        rt: _ZoneRuntime,
        zone_id: str,
        t: float,
        occupied: bool,
        track_id: Optional[str],
        confidence: float,
        motion_score: float,
        signals: object,
    ) -> List[IdleEvent]:
        if signals is None:
            return []
        events: List[IdleEvent] = []
        phone_score = float(getattr(signals, "phone_score", 0.0) or 0.0)
        phone_hot = rt.phone.active or rt.phone.candidate_since is not None
        phone_on = self.phone_enabled and phone_score >= (
            self.phone_clear_score if phone_hot else self.phone_min_score
        )
        talk_on = self.talk_enabled and bool(getattr(signals, "interacting", False))
        off_seat = (
            self.off_seat_enabled
            and occupied
            and not bool(getattr(signals, "seat_occupied", False))
        )
        not_working = (
            self.not_working_enabled
            and occupied
            and bool(getattr(signals, "seat_occupied", False))
            and bool(getattr(signals, "hands_known", False))
            and not bool(getattr(signals, "working", False))
            and not phone_on
            and not talk_on
        )
        steps = (
            (rt.phone, phone_on, self.phone_confirm, self.phone_grace, "PHONE", "phone"),
            (rt.interacting, talk_on, self.talk_confirm, self.talk_grace, "INTERACTING", "talking"),
            (rt.off_seat, off_seat, self.off_seat_confirm, self.off_seat_grace, "OFF_SEAT", "off_seat"),
            (
                rt.not_working,
                not_working,
                self.not_working_confirm,
                self.not_working_grace,
                "NOT_WORKING",
                "not_working",
            ),
        )
        for flag, condition, confirm, grace, prefix, reason in steps:
            events.extend(
                self._step_flag(
                    flag,
                    condition,
                    confirm,
                    grace,
                    zone_id,
                    t,
                    track_id,
                    confidence,
                    motion_score,
                    prefix,
                    reason,
                )
            )
        return events

    def _step_flag(
        self,
        flag: _FlagRuntime,
        condition: bool,
        confirm: float,
        grace: float,
        zone_id: str,
        t: float,
        track_id: Optional[str],
        confidence: float,
        motion_score: float,
        prefix: str,
        reason: str,
    ) -> List[IdleEvent]:
        events: List[IdleEvent] = []
        if condition:
            flag.gap_since = None
            flag.resume_since = None
            if not flag.active:
                if flag.candidate_since is None:
                    flag.candidate_since = t
                if t - flag.candidate_since >= confirm:
                    flag.active = True
                    flag.episode_started_at = flag.candidate_since
                    flag.episode_track_id = track_id
                    flag.episode_confidence = confidence
                    flag.escalate_emitted = False
                    events.append(
                        self._emit(
                            f"{prefix}_STARTED",
                            zone_id,
                            flag.candidate_since,
                            track_id,
                            confidence,
                            motion_score,
                            duration=t - flag.candidate_since,
                            reason=reason,
                        )
                    )
                    flag.candidate_since = None
            elif flag.episode_started_at is not None and not flag.escalate_emitted:
                if t - flag.episode_started_at >= self.escalate_seconds:
                    flag.escalate_emitted = True
                    events.append(
                        self._emit(
                            f"{prefix}_ESCALATED",
                            zone_id,
                            flag.episode_started_at,
                            flag.episode_track_id,
                            flag.episode_confidence,
                            motion_score,
                            duration=t - flag.episode_started_at,
                            reason=reason,
                        )
                    )
            return events

        if flag.candidate_since is not None and not flag.active:
            if flag.gap_since is None:
                flag.gap_since = t
            if t - flag.gap_since >= self.behavior_gap:
                flag.candidate_since = None
                flag.gap_since = None

        if flag.active:
            if flag.resume_since is None:
                flag.resume_since = t
            if t - flag.resume_since >= grace and flag.episode_started_at is not None:
                events.append(
                    self._close_flag(flag, zone_id, t, f"{prefix}_ENDED", reason)
                )
        return events

    def _close_flag(
        self,
        flag: _FlagRuntime,
        zone_id: str,
        t: float,
        event_type: str,
        reason: str,
    ) -> IdleEvent:
        started = flag.episode_started_at if flag.episode_started_at is not None else t
        ev = self._emit(
            event_type,
            zone_id,
            started,
            flag.episode_track_id,
            flag.episode_confidence,
            self._zones[zone_id].last_motion,
            duration=max(0.0, t - started),
            reason=reason,
        )
        ev.ended_at = self._iso(t)
        flag.active = False
        flag.episode_started_at = None
        flag.episode_track_id = None
        flag.resume_since = None
        flag.gap_since = None
        flag.candidate_since = None
        flag.escalate_emitted = False
        return ev

    def _emit(
        self,
        event_type: str,
        zone_id: str,
        started_t: float,
        track_id: Optional[str],
        confidence: float,
        motion_score: float,
        duration: float = 0.0,
        reason: Optional[str] = None,
    ) -> IdleEvent:
        rt = self._zones[zone_id]
        if reason is None:
            if event_type.startswith("AWAY"):
                reason = "empty"
            else:
                reason = "low_motion"
        return IdleEvent(
            event_type=event_type,
            camera_id=self.camera_id,
            zone_id=zone_id,
            track_id=track_id,
            started_at=self._iso(started_t),
            ended_at=None,
            duration_seconds=duration,
            confidence=float(confidence),
            motion_score=motion_score,
            site_timezone=self.tz_name,
            reason=reason,
            phone_score=rt.last_phone_score,
            seat_occupied=rt.last_seat_occupied,
            working=rt.last_working,
            interacting=rt.last_interacting,
        )

    def _close_episode(
        self,
        rt: _ZoneRuntime,
        zone_id: str,
        t: float,
        event_type: str,
    ) -> List[IdleEvent]:
        if rt.episode_started_at is None:
            return []
        ev = self._emit(
            event_type,
            zone_id,
            rt.episode_started_at,
            rt.episode_track_id,
            rt.episode_confidence,
            rt.last_motion,
            duration=max(0.0, t - rt.episode_started_at),
        )
        ev.ended_at = self._iso(t)
        rt.episode_started_at = None
        rt.episode_track_id = None
        return [ev]

    def states_snapshot(self) -> Dict[str, str]:
        return {zid: rt.state for zid, rt in self._zones.items()}

    def behavior_labels(self) -> Dict[str, List[str]]:
        labels: Dict[str, List[str]] = {}
        for zid, rt in self._zones.items():
            active: List[str] = []
            if rt.phone.active:
                active.append("phone")
            if rt.interacting.active:
                active.append("talking")
            if rt.off_seat.active:
                active.append("off_seat")
            if rt.not_working.active:
                active.append("not_working")
            labels[zid] = active
        return labels
