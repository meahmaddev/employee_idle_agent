#!/usr/bin/env bash
set -euo pipefail
export PATH="/venv/main/bin:$PATH"
cd /workspace/employee_idle_agent

echo "== deps =="
pip install -q ultralytics opencv-python-headless PyYAML

VIDEO="/workspace/employee_idle_agent/recordings/cam-01/cam01_90s_h264.mp4"
if [[ ! -f "$VIDEO" ]]; then
  VIDEO="/root/video/NVR-7BB9_ch12_main_20261118150005_20261118152059.mp4"
fi
# Ensure latest zones.yaml is used (ws-counter-1 computer operator desk)


echo "== probe =="
python - <<PY
import cv2
cap = cv2.VideoCapture("$VIDEO")
print("video", "$VIDEO")
print("open", cap.isOpened(), "frames", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps", cap.get(cv2.CAP_PROP_FPS))
ok, frame = cap.read()
print("first_frame", ok, None if not ok else frame.shape)
cap.release()
PY

echo "== run spike =="
export PYTHONPATH=src
python -m idle_agent \
  --source file \
  --url "$VIDEO" \
  --camera-id cam-01 \
  --device cuda:0 \
  --target-fps 8 \
  --output-dir /workspace/employee_idle_agent/output

echo "== results =="
ls -lah /workspace/employee_idle_agent/output/
wc -l /workspace/employee_idle_agent/output/events.jsonl || true
head -n 5 /workspace/employee_idle_agent/output/events.jsonl || true
