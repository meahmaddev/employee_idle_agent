"""Workstation ROI polygons (normalized or pixel coordinates)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import yaml

Point = Tuple[float, float]


@dataclass
class Zone:
    id: str
    name: str
    camera_id: str
    polygon: List[Point]
    enabled: bool = True
    expected_occupancy: int = 1
    # Chair pocket and keyboard/counter band. Default from the zone box when omitted.
    seat_polygon: Optional[List[Point]] = None
    work_polygon: Optional[List[Point]] = None

    def to_pixel_polygon(self, width: int, height: int) -> List[Tuple[int, int]]:
        return [(int(x * width), int(y * height)) for x, y in self.polygon]

    def polygon_pixels(self, polygon: Sequence[Point], width: int, height: int) -> List[Tuple[int, int]]:
        return [(int(x * width), int(y * height)) for x, y in polygon]


def _point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """Ray casting; polygon in same coordinate space as (x, y)."""
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _as_polygon(raw) -> Optional[List[Point]]:
    if not raw:
        return None
    return [(float(p[0]), float(p[1])) for p in raw]


def _band(polygon: Sequence[Point], y0_frac: float, y1_frac: float, x_pad: float = 0.0) -> List[Point]:
    """Axis-aligned band inside a polygon's bounding box. y fractions are top→bottom."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    width = x1 - x0
    height = y1 - y0
    xa = x0 - x_pad * width
    xb = x1 + x_pad * width
    ya = y0 + y0_frac * height
    yb = y0 + y1_frac * height
    return [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]


def load_zones(path: str | Path, camera_id: Optional[str] = None) -> List[Zone]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    zones: List[Zone] = []
    for item in data.get("zones", []):
        if camera_id and item.get("camera_id") != camera_id:
            continue
        if not item.get("enabled", True):
            continue
        poly = [(float(p[0]), float(p[1])) for p in item["polygon"]]
        # Seat = lower (closer-to-camera) chair band. Work = counter / keyboard band above it.
        seat = _as_polygon(item.get("seat_polygon")) or _band(poly, 0.52, 0.98, x_pad=0.08)
        work = _as_polygon(item.get("work_polygon")) or _band(poly, 0.18, 0.62, x_pad=0.0)
        zones.append(
            Zone(
                id=item["id"],
                name=item.get("name", item["id"]),
                camera_id=item.get("camera_id", camera_id or ""),
                polygon=poly,
                enabled=True,
                expected_occupancy=int(item.get("expected_occupancy", 1)),
                seat_polygon=seat,
                work_polygon=work,
            )
        )
    return zones


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    return _point_in_polygon(x, y, polygon)


def person_in_zone(
    cx_norm: float,
    cy_norm: float,
    zone: Zone,
) -> bool:
    return _point_in_polygon(cx_norm, cy_norm, zone.polygon)


def zones_for_person(
    cx_norm: float,
    cy_norm: float,
    zones: Iterable[Zone],
) -> List[Zone]:
    return [z for z in zones if person_in_zone(cx_norm, cy_norm, z)]
