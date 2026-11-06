"""Planar geometry helpers: body footprints, segment intersection, ray casting."""
from __future__ import annotations

import math

import numpy as np

from .kinematics import hitch_point, trailer_axle
from .vehicles import VehicleSpec

EDGE_NAMES = ("front", "right side", "rear", "left side")  # edges of [FL, FR, RR, RL]
CORNER_NAMES = ("front-left", "front-right", "rear-right", "rear-left")


def rect_corners(cx: float, cy: float, heading: float, front: float, rear: float, half_w: float) -> np.ndarray:
    """Corners [FL, FR, RR, RL] of a rectangle anchored at (cx, cy)."""
    ux, uy = math.cos(heading), math.sin(heading)
    nx, ny = -uy, ux
    return np.array(
        [
            [cx + front * ux + half_w * nx, cy + front * uy + half_w * ny],
            [cx + front * ux - half_w * nx, cy + front * uy - half_w * ny],
            [cx - rear * ux - half_w * nx, cy - rear * uy - half_w * ny],
            [cx - rear * ux + half_w * nx, cy - rear * uy + half_w * ny],
        ]
    )


def vehicle_corners(state, spec: VehicleSpec) -> np.ndarray:
    x, y, t1, _ = state
    return rect_corners(x, y, t1, spec.body_front, spec.body_rear, spec.body_width / 2)


def trailer_corners(state, spec: VehicleSpec) -> np.ndarray:
    a = trailer_axle(state, spec)
    return rect_corners(a[0], a[1], state[3], spec.trailer_front, spec.trailer_rear, spec.trailer_width / 2)


def footprint(state, spec: VehicleSpec) -> dict:
    return {
        "vehicle": vehicle_corners(state, spec),
        "trailer": trailer_corners(state, spec),
        "hitch": hitch_point(state, spec),
        "trailer_axle": trailer_axle(state, spec),
    }


def poly_edges(corners: np.ndarray) -> np.ndarray:
    """(4,2) corners -> (4,2,2) closed edges."""
    return np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def segments_intersect(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise intersection mask between segments A (n,2,2) and B (m,2,2) -> (n,m)."""
    p = A[:, None, 0, :]
    r = A[:, None, 1, :] - p
    q = B[None, :, 0, :]
    s = B[None, :, 1, :] - q
    denom = _cross(r, s)
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(qp, s) / denom
        u = _cross(qp, r) / denom
    return (np.abs(denom) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)


def ray_distance(origin: np.ndarray, direction: np.ndarray, segs: np.ndarray, max_range: float) -> float:
    """Distance along a unit ray to the nearest segment in segs (m,2,2), capped."""
    if len(segs) == 0:
        return max_range
    q = segs[:, 0, :]
    s = segs[:, 1, :] - q
    r = direction
    denom = r[0] * s[:, 1] - r[1] * s[:, 0]
    qp = q - origin
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / denom
        u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / denom
    ok = (np.abs(denom) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)
    if not ok.any():
        return max_range
    return float(min(max_range, t[ok].min()))


def points_in_rect(points: np.ndarray, center, heading: float, half_len: float, half_w: float, tol: float = 1e-6) -> np.ndarray:
    d = points - np.asarray(center)
    ux, uy = math.cos(heading), math.sin(heading)
    a = d[:, 0] * ux + d[:, 1] * uy
    b = -d[:, 0] * uy + d[:, 1] * ux
    return (np.abs(a) <= half_len + tol) & (np.abs(b) <= half_w + tol)


def perimeter_samples(front: float, rear: float, half_w: float, spacing: float) -> np.ndarray:
    """Body-frame points along the perimeter of a rectangle (for clearance checks)."""
    pts = []
    corners = [(front, half_w), (front, -half_w), (-rear, -half_w), (-rear, half_w)]
    for i in range(4):
        a = np.array(corners[i])
        b = np.array(corners[(i + 1) % 4])
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / spacing)))
        for j in range(n):
            pts.append(a + (b - a) * (j / n))
    return np.array(pts)
