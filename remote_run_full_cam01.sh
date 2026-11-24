#!/usr/bin/env bash
set -euo pipefail
export PATH="/venv/main/bin:$PATH"
cd /workspace/employee_idle_agent

VIDEO="/root/video/cam01_full_h264.mp4"
if [[ ! -f "$VIDEO" ]]; then
  VIDEO="/root/video/cam01_full.mp4"
fi

# Clear previous evidence from last run (keep folder)
rm -rf /workspace/employee_idle_agent/output/evidence
mkdir -p /workspace/employee_idle_agent/output/evidence

echo "== video =="
ls -lah "$VIDEO"

echo "== run full clip with evidence snapshots =="
export PYTHONPATH=src
python -m idle_agent \
  --source file \
  --url "$VIDEO" \
  --camera-id cam-01 \
  --device cuda:0 \
  --target-fps 8 \
  --preview-scale 0.5 \
  --clip-start "2026-11-18T15:00:05+05:00" \
  --output-dir /workspace/employee_idle_agent/output

echo "== results =="
ls -lah /workspace/employee_idle_agent/output/
ls -lah /workspace/employee_idle_agent/output/evidence/ | head -40
wc -l /workspace/employee_idle_agent/output/events.jsonl || true
head -n 5 /workspace/employee_idle_agent/output/events.jsonl || true
