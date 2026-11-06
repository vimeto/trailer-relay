"""Kinematic single-track model of a vehicle towing an off-axle-hitched trailer.

Per unit of signed travel distance s of the vehicle's rear axle (s < 0 = reverse):

    dx/ds      = cos(theta1)
    dy/ds      = sin(theta1)
    dtheta1/ds = tan(delta) / L
    dtheta2/ds = (sin(beta) - M * cos(beta) * tan(delta) / L) / L2,   beta = theta1 - theta2

The trailer equation follows from the hitch velocity component perpendicular to the
trailer (hitch at rear_axle - M*u1). The model is exact for low-speed motion
without tyre slip, which is the regime of a reversing course.
"""
from __future__ import annotations

import math

import numpy as np

from .vehicles import VehicleSpec

TWO_PI = 2.0 * math.pi


def wrap(a):
    """Wrap an angle (scalar or array) into [-pi, pi)."""
    return (a + math.pi) % TWO_PI - math.pi


def hitch_angle(state) -> float:
    return float(wrap(state[2] - state[3]))


def _deriv(t1, t2, k, M, L2, sin=math.sin, cos=math.cos):
    b = t1 - t2
    return cos(t1), sin(t1), k, (sin(b) - M * k * cos(b)) / L2


def step(state: np.ndarray, steer: float, d: float, spec: VehicleSpec) -> np.ndarray:
    """Advance `state` by signed distance d with constant steering (midpoint rule)."""
    x, y, t1, t2 = state
    k = math.tan(steer) / spec.wheelbase
    M, L2 = spec.hitch_offset, spec.trailer_length
    _, _, a1, a2 = _deriv(t1, t2, k, M, L2)
    mt1, mt2 = t1 + 0.5 * d * a1, t2 + 0.5 * d * a2
    cx, cy, m1, m2 = _deriv(mt1, mt2, k, M, L2)
    return np.array([x + d * cx, y + d * cy, t1 + d * m1, t2 + d * m2])


def step_batch(x, y, t1, t2, steer, d, spec: VehicleSpec):
    """Vectorised `step` over arrays of states / steering angles."""
    k = np.tan(steer) / spec.wheelbase
    M, L2 = spec.hitch_offset, spec.trailer_length
    b = t1 - t2
    a2 = (np.sin(b) - M * k * np.cos(b)) / L2
    mt1 = t1 + 0.5 * d * k
    mt2 = t2 + 0.5 * d * a2
    mb = mt1 - mt2
    m2 = (np.sin(mb) - M * k * np.cos(mb)) / L2
    return x + d * np.cos(mt1), y + d * np.sin(mt1), t1 + d * k, t2 + d * m2


def hitch_point(state, spec: VehicleSpec) -> np.ndarray:
    x, y, t1, _ = state
    return np.array([x - spec.hitch_offset * math.cos(t1), y - spec.hitch_offset * math.sin(t1)])


def trailer_axle(state, spec: VehicleSpec) -> np.ndarray:
    h = hitch_point(state, spec)
    t2 = state[3]
    return np.array([h[0] - spec.trailer_length * math.cos(t2), h[1] - spec.trailer_length * math.sin(t2)])


def trailer_rear_center(state, spec: VehicleSpec) -> np.ndarray:
    a = trailer_axle(state, spec)
    t2 = state[3]
    return np.array([a[0] - spec.trailer_rear * math.cos(t2), a[1] - spec.trailer_rear * math.sin(t2)])


def hitch_rate_reverse(beta: float, steer: float, spec: VehicleSpec) -> float:
    """d|beta|-direction rate per metre of *reverse* travel (d = -1)."""
    k = math.tan(steer) / spec.wheelbase
    M, L2 = spec.hitch_offset, spec.trailer_length
    # dbeta/ds = k - (sin b - M k cos b)/L2 ; reverse => multiply by -1
    return -(k - (math.sin(beta) - M * k * math.cos(beta)) / L2)


def critical_hitch_angle(spec: VehicleSpec) -> float:
    """Largest |beta| from which full counter-steer can still reduce |beta| in reverse.

    Beyond this angle the trailer keeps folding while reversing regardless of the
    steering: the driver must pull forward. If no such angle exists below the
    mechanical limit, the mechanical limit is returned.
    """
    dmax = spec.max_steer

    def recoverable(b: float) -> bool:
        # best achievable dbeta for beta>0 is with the steer that makes it most negative
        return min(hitch_rate_reverse(b, s, spec) for s in (-dmax, dmax)) < 0.0

    lo, hi = 1e-4, spec.max_hitch
    if recoverable(hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if recoverable(mid):
            lo = mid
        else:
            hi = mid
    return lo


def reversing_radius(spec: VehicleSpec, beta: float) -> float:
    """Trailer-axle turning radius in a steady reversing turn at hitch angle beta."""
    # In steady state both bodies rotate about a common centre. For the trailer
    # the centre lies on its axle line; with hitch radius Rh: sin(beta') geometry.
    # Use the simple bound Rt = L2 / tan(beta) (exact for M = 0).
    return spec.trailer_length / max(math.tan(abs(beta)), 1e-6)
