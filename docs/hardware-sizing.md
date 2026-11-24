# Hardware Sizing — Pilot Confirmed

Config: [`config/hardware.yaml`](../config/hardware.yaml)

## Decision

**Pilot runs on-prem** on a single NVIDIA GPU host processing **1–2 cameras at 5–10 FPS** (default **8 FPS**) using **YOLOv8s-pose + ByteTrack** on sub-streams (`640x360`).

## Confirmed minimum vs recommended

| Resource | Minimum (pilot) | Recommended |
|----------|-----------------|-------------|
| GPU | GTX 1660 / RTX 3050 (4 GB) | RTX 3060 / T4 (8 GB+) |
| CPU | 6 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Disk | 256 GB SSD | 512 GB+ (events + 30-day snapshots) |
| OS | Ubuntu 22.04 or Windows 10/11 + CUDA | Ubuntu 22.04 LTS |
| Alt edge | Jetson Orin Nano/NX with YOLOv8n-pose | — |

## Capacity estimate

| Setup | Cameras | Model | FPS/cam |
|-------|---------|-------|---------|
| Desktop GPU (recommended) | 2–4 | YOLOv8s-pose | ~8 |
| Jetson Orin | 1–2 | YOLOv8n-pose | 5–8 |

Stay under **75% GPU / 80% VRAM** util before adding cameras. Shard across workers when >4 cameras.

## Co-located pilot services

Redis + PostgreSQL + FastAPI may share the inference host for Phase 1. Bind DB/Redis to localhost. Only structured HTTPS events leave the LAN to ERP.

## Pre-flight checklist

- [ ] NVIDIA driver + CUDA installed; `nvidia-smi` works
- [ ] Host can open RTSP to NVR on LAN
- [ ] Host can reach ERP HTTPS endpoint
- [ ] ≥50 GB free for snapshots (30-day retention)
- [ ] Power / UPS suitable for continuous shift operation
