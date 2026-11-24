# Idle Rules — Finalized Defaults

Config: [`config/idle_rules.yaml`](../config/idle_rules.yaml)

## States

| State | Meaning |
|-------|---------|
| `ACTIVE` | Person in zone AND motion ≥ `active_threshold` |
| `IDLE` | Person in zone AND motion ≤ `idle_threshold` for `idle_confirm_seconds` |
| `AWAY` | No person in zone for `away_confirm_seconds` during shift |
| `BREAK` | Outside shift, weekly off, holiday, or approved break — **no alerts** |

## Final thresholds (Phase 1)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Inference evaluation tick | 1 s | Stable state decisions |
| Motion window | 3 s | Smooth pose/flow noise |
| `idle_threshold` | 0.08 | Low joint displacement = idle |
| `active_threshold` | 0.12 | Hysteresis vs idle |
| `idle_confirm_seconds` | **180** (3 min) | Avoid chatter from brief pauses |
| `away_confirm_seconds` | **300** (5 min) | Short absences (washroom) ignored |
| `escalate_seconds` | **900** (15 min) | Supervisor escalation |
| `debounce_leave_seconds` | 5 | Ignore detection flicker |
| `end_grace_seconds` | 10 | Require sustained resume before END event |

## Shift & breaks (default)

- **Shift:** Mon–Sat 08:00–17:00 (`Asia/Karachi`)
- **Breaks:** 10:30–10:45, 13:00–13:45, 15:30–15:45
- **Weekly off:** Sunday
- Adjust in `config/idle_rules.yaml` to match ERP / HR schedule

## Event timeline example

1. 10:00:00 — Worker stops moving → internal candidate IDLE  
2. 10:03:00 — Confirm → emit `IDLE_STARTED`  
3. 10:18:00 — Still idle → emit `IDLE_ESCALATED`  
4. 10:20:00 — Motion resumes 10s → emit `IDLE_ENDED` (duration ≈ 1200s)

## Policy

Phase 1 = **alert + supervisor review** only (`auto_discipline: false`).
