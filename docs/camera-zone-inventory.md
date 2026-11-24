# Camera & Zone Inventory (Phase 1 Pilot)

Source configs: [`config/cameras.yaml`](../config/cameras.yaml), [`config/zones.yaml`](../config/zones.yaml)

## Pilot scope

| Item | Target |
|------|--------|
| Cameras | 2 (`cam-01`, `cam-02`) |
| Zones | 4 workstation ROIs |
| Inference FPS | 5–10 (default 8) |
| Stream used for CV | Sub-stream (lower resolution) |

## Coverage map

| Camera | Location | Mount | Zones covered |
|--------|----------|-------|---------------|
| cam-01 | Assembly Hall A | overhead-angled | ws-assembly-1, ws-assembly-2 |
| cam-02 | Assembly Hall B | overhead-angled | ws-assembly-3, ws-assembly-4 |

## How to fill real credentials

1. From NVR UI or camera web page, copy RTSP URLs for **main** and **sub** streams.
2. Set environment variables (do not commit passwords):
   - `NVR_USER`
   - `NVR_PASSWORD`
3. Replace `PLACEHOLDER_NVR_HOST` and `PLACEHOLDER_BRAND` in `config/cameras.yaml`.
4. Capture one still from each sub-stream and redraw zone polygons in `config/zones.yaml` so each polygon tightly covers one workstation (person torso/legs visible).

## Zone drawing rules

- One zone = one workstation responsibility area.
- Prefer polygons that exclude aisles and neighboring stations.
- Coordinates are **normalized** (0–1) relative to frame width/height.
- After redraw, verify with a still overlay before going live.

## Verification checklist

- [ ] RTSP sub-stream opens in VLC / OpenCV for cam-01
- [ ] RTSP sub-stream opens for cam-02
- [ ] Still frame saved for each camera
- [ ] ROI polygons updated from placeholder halves
- [ ] Zones do not heavily overlap
- [ ] Bodies (not faces-only) visible in each ROI during work
