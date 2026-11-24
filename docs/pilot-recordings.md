# Pilot Recordings — Collection & Calibration Guide

Place clips under [`recordings/`](../recordings/). Do **not** commit large video files to git (see `.gitignore`).

## Purpose

Tune `motion.idle_threshold`, confirm durations, and validate zone polygons **before** live ERP alerts.

## What to collect

| Clip set | Duration | Content |
|----------|----------|---------|
| Per camera still | 1 frame | For drawing ROIs |
| Active work | 20–30 min | Normal productive motion in zone |
| Idle present | 10–15 min | Person in zone but not working (phone, standing still) |
| Away / empty | 10–15 min | Zone empty during work hours |
| Break transition | 5–10 min | Entering/leaving for break |
| Crowding / aisle | 5–10 min | Passers-by near zone edge (false-positive test) |

**Target total:** ~1–2 hours of video **per zone** (or per camera covering those zones).

## Folder layout

```
recordings/
  README.md
  cam-01/
    still.jpg
    active_work_001.mp4
    idle_present_001.mp4
    away_empty_001.mp4
    break_transition_001.mp4
    aisle_traffic_001.mp4
  cam-02/
    ...
  calibration/
    notes.md          # human notes after review
```

## Capture tips

1. Prefer **sub-stream** resolution matching inference (`640x360` or native sub).
2. Same time-of-day lighting as production shifts.
3. Export from NVR as MP4/H.264; keep original timestamps if possible.
4. Label files with camera id and scenario (see names above).
5. Blur faces in copies used off-site if privacy policy requires it; on-prem calibration can use raw.

## Calibration procedure

1. Overlay zones from `config/zones.yaml` on `still.jpg`; adjust polygons.
2. Run offline person detect + motion score on `active_work_*` → note typical motion scores.
3. Run on `idle_present_*` → set `idle_threshold` just above idle scores, below active scores.
4. Verify `idle_confirm_seconds` / `away_confirm_seconds` against labeled ground truth.
5. Record final values in `recordings/calibration/notes.md` and update `config/idle_rules.yaml` if changed.
6. Only then enable live POSTs to ERP.

## Collection checklist

- [ ] cam-01 still + 4 scenario clips
- [ ] cam-02 still + 4 scenario clips
- [ ] Zones redrawn from stills
- [ ] Thresholds written to calibration notes
- [ ] `idle_rules.yaml` updated if defaults changed
