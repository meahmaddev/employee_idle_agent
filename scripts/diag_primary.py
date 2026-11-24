#!/usr/bin/env python3
"""Quick primary-track diagnostic on GPU."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idle_agent.detect import Detector
from idle_agent.primary import ZonePrimaryTracker
from idle_agent.sources import VideoSource
from idle_agent.state_machine import ZoneStateMachine, load_rules
from idle_agent.zones import load_zones, person_in_zone


def main() -> int:
    zones = load_zones(ROOT / "config" / "zones.yaml", camera_id="cam-01")
    print("zones", [(z.id, z.polygon) for z in zones])
    rules = load_rules(ROOT / "config" / "idle_rules.yaml", ROOT / "config" / "demo_overrides.yaml")
    pc = rules.get("primary_track", {})
    primary = ZonePrimaryTracker(
        enabled=True,
        lost_grace_seconds=float(pc.get("lost_grace_seconds", 5)),
        assign_dwell_seconds=float(pc.get("assign_dwell_seconds", 0.5)),
        match_center_dist=float(pc.get("match_center_dist", 0.08)),
        match_iou=float(pc.get("match_iou", 0.25)),
    )
    sm = ZoneStateMachine("cam-01", [z.id for z in zones], rules)
    det = Detector(model_name="yolov8s-pose.pt", conf_min=0.45, device="cuda:0")
    det.load()
    video = "/root/video/cam01_full_h264.mp4"
    counts = {"occ": 0, "empty": 0}
    with VideoSource(video, "file", max_frames=25 * 180) as src:
        every = max(1, int(round(src.source_fps / 8)))
        for pkt in src.frames():
            if pkt.index % every:
                continue
            people = det.infer(pkt.frame)
            z = zones[0]
            occ, person, pid = primary.update(z, pkt.timestamp_seconds, people)
            counts["occ" if occ else "empty"] += 1
            sm.update(
                z.id,
                pkt.timestamp_seconds,
                occ,
                person.motion_score if person else 0.0,
                pid,
                person.confidence if person else 0.0,
            )
            st = sm.states_snapshot()[z.id]
            if int(pkt.timestamp_seconds) % 15 == 0 and pkt.index % (every * 8) == 0:
                in_poly = sum(1 for p in people if person_in_zone(p.center_norm[0], p.center_norm[1], z))
                print(
                    f"t={pkt.timestamp_seconds:.1f}s occ={occ} prim={pid} state={st} "
                    f"people={len(people)} in_poly={in_poly} "
                    f"m={None if not person else round(person.motion_score, 4)}"
                )
    print("counts", counts, "final", sm.states_snapshot())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
