# Architecture — Employee Idle Time (Phase 1)

## Goal

Detect workstation **IDLE** (person present, low activity) and **AWAY** (no person during shift), measure duration, and POST events to the ERP/software API with retries and idempotency.

## Pipeline

```mermaid
flowchart LR
  CCTV[CCTV_RTSP] --> Ingest[Video_Ingest]
  Ingest --> Detect[Person_Detect_Track]
  Detect --> Zones[Zone_ROI_Engine]
  Zones --> IdleLogic[Idle_State_Machine]
  IdleLogic --> Events[Event_Queue]
  Events --> API[ERP_API_Adapter]
  Events --> Store[(Event_Store)]
  Config[Shift_Zones_Config] --> IdleLogic
  Config --> Zones
```

## Components

| Component | Role | Config |
|-----------|------|--------|
| Video ingest | RTSP sub-stream pull, frame sample @ 5–10 FPS | `config/cameras.yaml` |
| Person detect + track | YOLOv8 + ByteTrack; pose for motion | hardware profile |
| Zone ROI engine | Point-in-polygon occupancy per workstation | `config/zones.yaml` |
| Idle state machine | ACTIVE / IDLE / AWAY / BREAK + thresholds | `config/idle_rules.yaml` |
| Event queue | Redis Streams decouple CV from HTTP | `config/erp_api.yaml` |
| ERP adapter | HTTPS POST, auth, backoff, DLQ | `config/erp_api.yaml` |
| Event store | PostgreSQL audit of all emissions | `DATABASE_URL` |

## State machine

```mermaid
stateDiagram-v2
  [*] --> UNKNOWN
  UNKNOWN --> AWAY: no_person_in_zone
  UNKNOWN --> ACTIVE: person_and_motion
  AWAY --> ACTIVE: person_enters_and_moves
  ACTIVE --> IDLE: low_motion_for_T_idle
  IDLE --> ACTIVE: motion_resumes
  IDLE --> AWAY: person_leaves
  AWAY --> IDLE: person_returns_but_still
  ACTIVE --> BREAK: outside_shift
  IDLE --> BREAK: outside_shift
  AWAY --> BREAK: outside_shift
  BREAK --> UNKNOWN: shift_resumes
```

## Deployment (on-prem)

```mermaid
flowchart TB
  subgraph factory [Factory_LAN]
    Cams[CCTV_NVR]
    Edge[Inference_Server_GPU]
    DB[(PostgreSQL)]
    Redis[(Redis)]
    Cams -->|RTSP| Edge
    Edge --> Redis
    Edge --> DB
  end
  Redis -->|HTTPS_events| ERP[Your_ERP_API]
```

Video stays on the factory LAN. Only structured events (and optional evidence paths) leave via HTTPS.

## Privacy

Anonymous `track_id` only — no face recognition in Phase 1.

## Implementation roadmap (post-architecture)

1. Spike: RTSP → YOLO person detect on `cam-01`
2. Zones + idle state machine + local event log
3. ERP API adapter + retries
4. Pilot calibrate with `recordings/`
5. Multi-camera scale + optional admin UI

## Related docs

- [Camera & zone inventory](camera-zone-inventory.md)
- [Idle rules](idle-rules.md)
- [API contract](api-contract.md)
- [Hardware sizing](hardware-sizing.md)
- [Pilot recordings](pilot-recordings.md)
- [Phase 2 deferred](phase2-deferred.md)
