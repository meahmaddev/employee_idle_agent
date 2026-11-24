# ERP / Software API Contract

Configs: [`config/erp_api.yaml`](../config/erp_api.yaml)  
Schema: [`schemas/idle_event.schema.json`](../schemas/idle_event.schema.json)  
OpenAPI: [`openapi/erp-events.yaml`](../openapi/erp-events.yaml)

## Direction

The **CCTV idle agent is the client**. Your ERP/software exposes:

`POST {ERP_BASE_URL}/api/v1/vision/idle-events`

## Authentication

| Mode | Header | Env var |
|------|--------|---------|
| API key (default) | `X-API-Key: <token>` | `ERP_API_TOKEN` |
| Bearer JWT (alt) | `Authorization: Bearer <token>` | `ERP_API_TOKEN` |

Also set `ERP_BASE_URL` (e.g. `https://erp.yourcompany.com`).

## Event payload (confirmed fields)

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "IDLE_STARTED",
  "camera_id": "cam-01",
  "zone_id": "ws-assembly-3",
  "track_id": "t-4821",
  "started_at": "2026-11-17T10:00:00+05:00",
  "ended_at": null,
  "duration_seconds": 180,
  "confidence": 0.87,
  "snapshot_url": "optional path",
  "idempotency_key": "cam-01:ws-assembly-3:IDLE_STARTED:2026-11-17T10:00:00+05:00",
  "site_timezone": "Asia/Karachi",
  "motion_score": 0.04
}
```

### Event types

- `IDLE_STARTED` / `IDLE_ESCALATED` / `IDLE_ENDED`
- `AWAY_STARTED` / `AWAY_ESCALATED` / `AWAY_ENDED`

## Reliability rules

1. Send `Idempotency-Key` header = payload `idempotency_key`.
2. On `409` (duplicate) or prior success for same key → treat as success (no retry).
3. Retry on `408`, `429`, `5xx` with exponential backoff (2s → 60s, max 5 attempts).
4. Failures after max attempts → Redis dead-letter stream `idle_events_dlq` + local PostgreSQL audit row.
5. **Never block** the CV inference loop on ERP latency — queue asynchronously.

## Expected ERP response

- Prefer `202 Accepted` with `{ "status": "accepted", "event_id": "..." }`
- `200 OK` also accepted

## Inbound (optional Phase 1b)

If `inbound.enabled: true`, agent may poll `/api/v1/hr/shifts` hourly to refresh break/shift windows instead of static YAML.

## Integration checklist for ERP team

- [ ] Implement `POST /api/v1/vision/idle-events`
- [ ] Validate against JSON Schema
- [ ] Honor Idempotency-Key
- [ ] Issue API token for factory agent
- [ ] Confirm TLS certificate trusted on factory host
- [ ] (Optional) Expose shift/break endpoint
