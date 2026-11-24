# Cam-01 offline spike

Run person detect + idle/away state machine on a local recording under `recordings/cam-01/`.

## Prerequisites

- Python 3.10+ from [python.org](https://www.python.org/downloads/) (Windows Store stub is not enough — `python --version` must print a real version)
- NVIDIA GPU + CUDA recommended (Vast.ai 8 GB VRAM is enough). CPU works but is slow.
- Sample MP4 in `recordings/cam-01/` (already gitignored)

Or one-shot on Windows after Python is installed:

```powershell
.\scripts\run_cam01.ps1
# optional: .\scripts\run_cam01.ps1 --max-frames 120 --device cpu
```

## Setup

```powershell
cd "d:\Al Wahid Business Consultant\New Clients\Contech Digital\CCTV Camra"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Vast.ai (Ubuntu + CUDA), install a CUDA build of PyTorch first, then `pip install -r requirements.txt`.

## Run (full clip)

```powershell
$env:PYTHONPATH = "src"
python -m idle_agent --source file --camera-id cam-01
```

If multiple videos exist, pass the path explicitly:

```powershell
python -m idle_agent --source file --url "recordings/cam-01/NVR-7BB9_ch12_main_20261118150005_20261118152059.mp4"
```

## CPU smoke test (short)

```powershell
python -m idle_agent --source file --device cpu --max-frames 120 --target-fps 4
```

## Outputs

| File | Meaning |
|------|---------|
| `output/events.jsonl` | IDLE_*/AWAY_* events (demo thresholds: 15s idle / 20s away) |
| `output/preview.mp4` | Boxes, zone polygons, state labels |

Demo thresholds live in `config/demo_overrides.yaml`. Production durations remain in `config/idle_rules.yaml`.

## Later: live RTSP

```powershell
python -m idle_agent --source rtsp --url "rtsp://USER:PASS@NVR_HOST:554/Streaming/Channels/102" --camera-id cam-01
```

Prefer a factory LAN host for RTSP. Cloud GPUs usually cannot reach private NVR IPs without VPN.
