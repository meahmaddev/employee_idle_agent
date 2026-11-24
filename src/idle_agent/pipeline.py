"""End-to-end offline / live inference pipeline."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from idle_agent.behaviors import BehaviorScorer
from idle_agent.detect import Detector, PersonDetection
from idle_agent.primary import ZonePrimaryTracker
from idle_agent.sources import VideoSource
from idle_agent.state_machine import ZoneStateMachine, load_rules
from idle_agent.zones import Zone, load_zones


def _draw_zones(
    frame: np.ndarray,
    zones: List[Zone],
    states: dict,
    primary_ids: Optional[Dict[str, Optional[str]]] = None,
    behaviors: Optional[Dict[str, List[str]]] = None,
) -> None:
    h, w = frame.shape[:2]
    primary_ids = primary_ids or {}
    behaviors = behaviors or {}
    for zone in zones:
        pts = np.array(zone.to_pixel_polygon(w, h), dtype=np.int32)
        color = (0, 200, 255)
        state = states.get(zone.id, "UNKNOWN")
        if state == "IDLE":
            color = (0, 165, 255)
        elif state == "AWAY":
            color = (0, 0, 255)
        elif state == "ACTIVE":
            color = (0, 220, 0)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        if zone.work_polygon:
            work = np.array(zone.polygon_pixels(zone.work_polygon, w, h), dtype=np.int32)
            cv2.polylines(frame, [work], isClosed=True, color=(255, 0, 255), thickness=1)
        if zone.seat_polygon:
            seat = np.array(zone.polygon_pixels(zone.seat_polygon, w, h), dtype=np.int32)
            cv2.polylines(frame, [seat], isClosed=True, color=(255, 180, 0), thickness=1)
        x, y = pts[0]
        prim = primary_ids.get(zone.id)
        label = f"{zone.id}:{state}"
        if prim:
            label += f" prim={prim}"
        flags = behaviors.get(zone.id) or []
        if flags:
            label += " " + ",".join(flags)
        cv2.putText(
            frame,
            label,
            (int(x), max(20, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_people(
    frame: np.ndarray,
    people: List[PersonDetection],
    primary_track_ids: Optional[Set[str]] = None,
) -> None:
    primary_track_ids = primary_track_ids or set()
    for p in people:
        x1, y1, x2, y2 = map(int, p.bbox_xyxy)
        is_primary = p.track_id in primary_track_ids
        color = (0, 255, 0) if is_primary else (255, 180, 0)
        thickness = 3 if is_primary else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        tag = "PRIMARY " if is_primary else ""
        phone = getattr(p, "phone_score", 0.0) or 0.0
        label = f"{tag}{p.track_id} m={p.motion_score:.3f}"
        if phone >= 0.2:
            label += f" ph={phone:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


# Events that should keep a JPEG for supervisor review / ERP evidence
_EVIDENCE_EVENT_TYPES = {
    "IDLE_STARTED",
    "IDLE_ESCALATED",
    "AWAY_STARTED",
    "AWAY_ESCALATED",
    "PHONE_STARTED",
    "PHONE_ESCALATED",
    "INTERACTING_STARTED",
    "INTERACTING_ESCALATED",
    "OFF_SEAT_STARTED",
    "OFF_SEAT_ESCALATED",
    "NOT_WORKING_STARTED",
    "NOT_WORKING_ESCALATED",
}


def _safe_stamp(iso_text: str) -> str:
    return (
        iso_text.replace(":", "")
        .replace("+", "_")
        .replace("-", "")
        .replace(".", "_")
    )


def _save_evidence_snapshot(
    frame: np.ndarray,
    people: List[PersonDetection],
    zones: List[Zone],
    states: dict,
    event_type: str,
    zone_id: str,
    event_id: str,
    started_at: str,
    evidence_dir: Path,
    camera_id: str,
    video_t: float,
    primary_ids: Optional[Dict[str, Optional[str]]] = None,
    behaviors: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Write annotated JPEG; return path relative to output dir (evidence/...)."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    primary_ids = primary_ids or {}
    prim_set = {v for v in primary_ids.values() if v}
    vis = frame.copy()
    _draw_zones(vis, zones, states, primary_ids, behaviors)
    _draw_people(vis, people, prim_set)
    cv2.putText(
        vis,
        f"{camera_id} {event_type} {zone_id} t={video_t:.1f}s",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        started_at,
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    fname = f"{_safe_stamp(started_at)}_{zone_id}_{event_type}_{event_id[:8]}.jpg"
    path = evidence_dir / fname
    cv2.imwrite(str(path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return f"evidence/{fname}"


def run_pipeline(
    source_type: str,
    source_uri: str,
    camera_id: str,
    zones_path: str | Path,
    idle_rules_path: str | Path,
    demo_overrides_path: Optional[str | Path],
    output_dir: str | Path,
    model_name: str = "yolov8s-pose.pt",
    target_fps: float = 8.0,
    max_frames: Optional[int] = None,
    device: Optional[str] = None,
    write_preview: bool = True,
    clip_start: Optional[datetime] = None,
    preview_scale: float = 1.0,
) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    preview_path = output_dir / "preview.mp4"
    evidence_dir = output_dir / "evidence"
    # Fresh evidence folder for this run (do not delete other output files here)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    zones = load_zones(zones_path, camera_id=camera_id)
    if not zones:
        raise RuntimeError(f"No enabled zones for camera_id={camera_id} in {zones_path}")

    rules = load_rules(idle_rules_path, demo_overrides_path)
    eval_cfg = rules.get("evaluation", {})
    primary_cfg = rules.get("primary_track", {})
    conf_min = float(eval_cfg.get("person_confidence_min", 0.45))
    include_snapshot = bool(rules.get("events", {}).get("include_snapshot", True))
    primary = ZonePrimaryTracker(
        enabled=bool(primary_cfg.get("enabled", True)),
        lost_grace_seconds=float(
            primary_cfg.get("lost_grace_seconds", eval_cfg.get("debounce_leave_seconds", 3))
        ),
        assign_dwell_seconds=float(primary_cfg.get("assign_dwell_seconds", 0.5)),
        match_center_dist=float(primary_cfg.get("match_center_dist", 0.08)),
        match_iou=float(primary_cfg.get("match_iou", 0.25)),
    )
    sm = ZoneStateMachine(camera_id, [z.id for z in zones], rules, clip_start=clip_start)
    scorer = BehaviorScorer(rules.get("behaviors") or {})
    if clip_start is not None:
        print(f"Clip wall-clock start: {clip_start.isoformat()}")
    print(
        f"Thresholds: idle={sm.idle_confirm}s away={sm.away_confirm}s "
        f"escalate={sm.escalate_seconds}s  evidence_snapshots={include_snapshot}"
    )
    print(
        f"Behaviors: phone={sm.phone_confirm}s talking={sm.talk_confirm}s "
        f"off_seat={sm.off_seat_confirm}s not_working={sm.not_working_confirm}s"
    )
    print(
        f"Primary track: enabled={primary.enabled} "
        f"lost_grace={primary.lost_grace_seconds}s "
        f"assign_dwell={primary.assign_dwell_seconds}s"
    )
    detector = Detector(model_name=model_name, conf_min=conf_min, device=device)
    detector.load()

    writer = None
    event_count = 0
    snapshot_count = 0
    wall_start = time.time()

    with VideoSource(source_uri, source_type=source_type, max_frames=max_frames) as source:
        sample_every = max(1, int(round(source.source_fps / max(target_fps, 0.1))))
        print(
            f"Opened {source_type}: {source_uri} "
            f"({source.frame_width}x{source.frame_height} @ {source.source_fps:.1f} fps, "
            f"sample every {sample_every} frame(s) → ~{target_fps} FPS)"
        )

        with events_path.open("w", encoding="utf-8") as event_file:
            for packet in source.frames():
                if packet.index % sample_every != 0:
                    continue

                people = detector.infer(packet.frame)
                states = sm.states_snapshot()

                for zone in zones:
                    occupied, person, prim_id = primary.update(
                        zone, packet.timestamp_seconds, people
                    )
                    signals = scorer.score(
                        person,
                        people,
                        zone,
                        packet.frame.shape[1],
                        packet.frame.shape[0],
                        occupied,
                    )
                    if person is not None:
                        person.phone_score = signals.phone_score
                    motion = person.motion_score if person else 0.0
                    track_id = prim_id or (person.track_id if person else None)
                    confidence = person.confidence if person else 0.0
                    new_events = sm.update(
                        zone.id,
                        packet.timestamp_seconds,
                        occupied,
                        motion,
                        track_id,
                        confidence,
                        signals,
                    )
                    states = sm.states_snapshot()
                    prim_map = primary.primary_ids()
                    prim_set = {v for v in prim_map.values() if v}
                    for ev in new_events:
                        if (
                            include_snapshot
                            and ev.event_type in _EVIDENCE_EVENT_TYPES
                        ):
                            rel = _save_evidence_snapshot(
                                packet.frame,
                                people,
                                zones,
                                states,
                                ev.event_type,
                                ev.zone_id,
                                ev.event_id,
                                ev.started_at,
                                evidence_dir,
                                camera_id,
                                packet.timestamp_seconds,
                                prim_map,
                                sm.behavior_labels(),
                            )
                            ev.snapshot_url = rel
                            snapshot_count += 1
                        event_file.write(json.dumps(ev.to_dict()) + "\n")
                        event_file.flush()
                        event_count += 1
                        snap = f" snap={ev.snapshot_url}" if ev.snapshot_url else ""
                        print(
                            f"[event] {ev.event_type} reason={ev.reason} zone={ev.zone_id} "
                            f"primary={prim_id} t={packet.timestamp_seconds:.1f}s "
                            f"at={ev.started_at}{snap}"
                        )

                states = sm.states_snapshot()
                prim_map = primary.primary_ids()
                prim_set = {v for v in prim_map.values() if v}
                if write_preview:
                    vis = packet.frame.copy()
                    _draw_zones(vis, zones, states, prim_map, sm.behavior_labels())
                    _draw_people(vis, people, prim_set)
                    wall_label = ""
                    if clip_start is not None:
                        from datetime import timedelta

                        wall_label = " " + (clip_start + timedelta(seconds=packet.timestamp_seconds)).isoformat()
                    prim_txt = ",".join(f"{k}:{v}" for k, v in prim_map.items() if v) or "-"
                    cv2.putText(
                        vis,
                        f"{camera_id} t={packet.timestamp_seconds:.1f}s{wall_label} "
                        f"events={event_count} primary={prim_txt}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if preview_scale != 1.0:
                        vis = cv2.resize(
                            vis,
                            None,
                            fx=preview_scale,
                            fy=preview_scale,
                            interpolation=cv2.INTER_AREA,
                        )
                    if writer is None:
                        h, w = vis.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(preview_path),
                            fourcc,
                            float(target_fps),
                            (w, h),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open preview writer: {preview_path}")
                    writer.write(vis)

                if packet.index > 0 and packet.index % (sample_every * 50) == 0:
                    elapsed = time.time() - wall_start
                    prim_dbg = primary.primary_ids()
                    print(
                        f"... frame {packet.index} video_t={packet.timestamp_seconds:.1f}s "
                        f"wall={elapsed:.1f}s states={states} primary={prim_dbg}"
                    )

    if writer is not None:
        writer.release()

    print(f"Done. events={event_count} snapshots={snapshot_count} → {events_path}")
    print(f"Evidence → {evidence_dir}")
    if write_preview:
        print(f"Preview → {preview_path}")
    return events_path, preview_path
