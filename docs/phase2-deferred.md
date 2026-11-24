# Phase 2+ Deferred — Explicitly Out of Scope Until Idle Pilot Succeeds

This document records intentional deferral of non–idle-time use cases so Phase 1 stays focused.

## Deferred until idle-time pilot success criteria are met

| Use case | Status | Why deferred |
|----------|--------|--------------|
| Tools / fixed assets theft & tracking via CV | **Deferred** | Needs ERP issue/return sync, object detectors, exit-zone rules; different models and false-positive profile |
| Raw material theft monitoring via CV | **Deferred** | Needs material/area detectors, quantity heuristics, store-room camera design |
| Face recognition / biometric identity | **Out of scope** | Privacy; Phase 1 uses anonymous track IDs |
| Full production dashboard UI | **Optional Phase 1c** | Not required for event→API MVP |
| Automatic payroll/discipline actions | **Out of scope** | Human-in-the-loop review only |

## Idle pilot success criteria (gate for Phase 2)

From the architecture plan — do **not** start theft CV until:

1. Idle/away events on pilot zones have an acceptable false-positive rate after 1–2 weeks of threshold tuning.
2. Every qualifying event reliably reaches the ERP API (idempotent, retried).
3. System runs a full shift continuously without manual restart.

## When Phase 2 starts

Plan separately (same pattern as idle):

1. Tools/assets: ERP issue ledger ↔ CV object present/absent at exits and tool cribs.
2. Raw material: store and line cameras ↔ ERP material issue tickets.
3. Shared: same ingest, queue, and API adapter patterns; new detectors and event types only.

## Decision log

| Date | Decision |
|------|----------|
| 2026-11 | Phase 1 = employee idle time architecture only; theft use cases deferred |
