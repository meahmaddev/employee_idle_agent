"""CLI for the cam-01 offline spike (file) and future live RTSP."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_clip_start(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    # Allow trailing Z
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SystemExit(
            f"Invalid --clip-start '{value}'. Use ISO8601, e.g. 2026-11-18T15:00:05+05:00"
        ) from exc
    if dt.tzinfo is None:
        # Assume Asia/Karachi offset if bare local time
        from datetime import timezone, timedelta

        dt = dt.replace(tzinfo=timezone(timedelta(hours=5)))
    return dt


def _default_cam01_video(root: Path) -> Path | None:
    cam_dir = root / "recordings" / "cam-01"
    if not cam_dir.is_dir():
        return None
    videos = sorted(
        [
            *cam_dir.glob("*.mp4"),
            *cam_dir.glob("*.avi"),
            *cam_dir.glob("*.mkv"),
            *cam_dir.glob("*.mov"),
        ]
    )
    return videos[0] if videos else None


def build_parser() -> argparse.ArgumentParser:
    root = _repo_root()
    default_video = _default_cam01_video(root)
    parser = argparse.ArgumentParser(
        description="Employee idle-time spike: file (cam-01) or RTSP → YOLO → events/preview",
    )
    parser.add_argument(
        "--source",
        choices=["file", "rtsp"],
        default="file",
        help="Video source type (default: file)",
    )
    parser.add_argument(
        "--url",
        default=str(default_video) if default_video else "",
        help="File path or RTSP URL (default: first mp4 under recordings/cam-01)",
    )
    parser.add_argument("--camera-id", default="cam-01")
    parser.add_argument(
        "--zones",
        default=str(root / "config" / "zones.yaml"),
    )
    parser.add_argument(
        "--idle-rules",
        default=str(root / "config" / "idle_rules.yaml"),
    )
    parser.add_argument(
        "--demo-overrides",
        default=str(root / "config" / "demo_overrides.yaml"),
        help="Short thresholds for offline testing; pass empty string to disable",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "output"),
    )
    parser.add_argument("--model", default="yolov8s-pose.pt")
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on source frames (useful for CPU smoke tests)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip writing output/preview.mp4",
    )
    parser.add_argument(
        "--clip-start",
        default=None,
        help="Wall-clock start of the recording (ISO8601), e.g. 2026-11-18T15:00:05+05:00",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.5,
        help="Scale factor for preview.mp4 (default 0.5 to save disk on long clips)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Allow `python -m idle_agent.cli` when repo src/ is on PYTHONPATH
    root = _repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from idle_agent.pipeline import run_pipeline

    args = build_parser().parse_args(argv)
    if not args.url:
        print(
            "No video URL/path provided and no file found in recordings/cam-01/. "
            "Place an MP4 there or pass --url.",
            file=sys.stderr,
        )
        return 2

    demo = args.demo_overrides if args.demo_overrides else None
    run_pipeline(
        source_type=args.source,
        source_uri=args.url,
        camera_id=args.camera_id,
        zones_path=args.zones,
        idle_rules_path=args.idle_rules,
        demo_overrides_path=demo,
        output_dir=args.output_dir,
        model_name=args.model,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        device=args.device,
        write_preview=not args.no_preview,
        clip_start=_parse_clip_start(args.clip_start),
        preview_scale=args.preview_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
