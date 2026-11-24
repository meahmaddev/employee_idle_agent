# Factory CCTV AI Agent — Employee Idle Time (Phase 1)

On-prem computer-vision pipeline that detects employee **idle** and **away** time at factory workstations from existing CCTV feeds, then POSTs structured events to your software/ERP API.

**Phase 1 scope:** Employee idle time only.  

## Architecture overview

```
CCTV (RTSP) → Video Ingest → YOLO Person Detect + ByteTrack
    → Zone ROI Engine → Idle State Machine → Event Queue
        → ERP API Adapter (+ PostgreSQL audit store)
```

Full design: [docs/architecture.md](docs/architecture.md)

## Repository layout

| Path | Purpose |
|------|---------|
| `config/cameras.yaml` | Pilot cameras, RTSP, coverage |
| `config/zones.yaml` | Workstation ROI polygons |
| `config/idle_rules.yaml` | Thresholds, shifts, breaks, escalations |
| `config/erp_api.yaml` | Outbound API endpoint & auth |
| `config/hardware.yaml` | On-prem GPU sizing for pilot |
| `docs/` | Architecture, API contract, checklists |
| `schemas/idle_event.schema.json` | Event payload JSON Schema |
| `openapi/erp-events.yaml` | OpenAPI for outbound events |
| `recordings/` | Place pilot CCTV sample clips here |

## Quick start — cam-01 offline spike

1. Place an MP4 under `recordings/cam-01/` (see [docs/pilot-recordings.md](docs/pilot-recordings.md)).
2. Follow [docs/cam01-spike.md](docs/cam01-spike.md):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m idle_agent --source file --camera-id cam-01
```

Outputs: `output/preview.mp4` and `output/events.jsonl`.

Live RTSP (after NVR credentials): same CLI with `--source rtsp --url "rtsp://..."`.

## Live view (watch what the model sees)

Add `--live-view-port 8090` to any run, then open `http://localhost:8090`.
The page shows person boxes, zones and the current state for every processed frame.

On a rented GPU (Vast.ai) the server only listens on the GPU itself, so forward
the port through SSH from your PC (add `-L` to the tunnel command you already use):

```powershell
ssh -p PORT root@HOST -N -R 9090:PHONE_IP:8080 -L 8090:localhost:8090
```

Press `Ctrl+C` to stop a live run; outputs (events, evidence, preview) are saved cleanly.

## Tech stack (locked)

- Ingest: RTSP / ONVIF (OpenCV / FFmpeg)
- Detect: YOLOv8n/s + ByteTrack + YOLOv8-pose
- Host: On-prem NVIDIA GPU (or Jetson for 1–2 cams)
- Services: Python, FastAPI, Redis Streams, PostgreSQL
- Outbound: HTTPS REST with retry + idempotency keys

## Privacy

Phase 1 uses anonymous track IDs only — no face recognition.
