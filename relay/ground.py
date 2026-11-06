"""Uneven ground: a deterministic disturbance on the trailer's heading.

Pure kinematics keep a perfectly straight rig straight forever when reversing
with the wheels centred, so a straight course would be trivial. Real ground
(camber, ruts, bumps) keeps nudging the trailer, and because reversing a
trailer is unstable those nudges grow unless the driver keeps correcting.

The field is a smooth function of the trailer axle's position, seeded from
the course. Every driver who takes the same path meets the same bumps, so
runs stay deterministic and comparable across models.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

DEFAULT_RMS_DEG_PER_M = 0.5  # RMS yaw nudge on the trailer, degrees per metre travelled


class GroundField:
    def __init__(self, key: str, rms_deg_per_m: float = DEFAULT_RMS_DEG_PER_M, waves: int = 5):
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        ang = rng.uniform(0, 2 * math.pi, waves)
        lam = rng.uniform(3.0, 12.0, waves)  # wavelengths in metres
        self.kx = 2 * math.pi / lam * np.cos(ang)
        self.ky = 2 * math.pi / lam * np.sin(ang)
        self.phase = rng.uniform(0, 2 * math.pi, waves)
        w = rng.uniform(0.5, 1.0, waves)
        # a sum of unit sinusoids has RMS sqrt(sum(w^2)/2); normalise to the target
        self.amp = math.radians(rms_deg_per_m) * w / math.sqrt(float(np.sum(w ** 2)) / 2.0) if rms_deg_per_m else w * 0.0
        self.mean = 0.0
        self.rms = math.radians(rms_deg_per_m)

    def __call__(self, x: float, y: float) -> float:
        """Trailer yaw rate (rad per metre travelled) at ground point (x, y)."""
        return float(np.sum(self.amp * np.sin(self.kx * x + self.ky * y + self.phase)))


__all__ = ["GroundField", "DEFAULT_RMS_DEG_PER_M"]
