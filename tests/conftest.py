import math
from pathlib import Path

import numpy as np
import pytest

from relay.seeder import generate_track, load_or_generate_relay
from relay.track import Primitive, Track
from relay.vehicles import get_vehicle

ROOT = Path(__file__).resolve().parent.parent


def state_for_trailer_at(track: Track, s: float, beta: float = 0.0, lateral: float = 0.0) -> np.ndarray:
    """Full state with the trailer axle at arc length s (offset `lateral` to the route-left)."""
    spec = get_vehicle(track.vehicle_key)
    x = np.interp(s, track.s, track.centerline[:, 0])
    y = np.interp(s, track.s, track.centerline[:, 1])
    h = np.interp(s, track.s, np.unwrap(track.headings))
    x, y = x - lateral * math.sin(h), y + lateral * math.cos(h)
    t2 = h + math.pi
    t1 = t2 + beta
    hx, hy = x + spec.trailer_length * math.cos(t2), y + spec.trailer_length * math.sin(t2)
    rx, ry = hx + spec.hitch_offset * math.cos(t1), hy + spec.hitch_offset * math.sin(t1)
    return np.array([rx, ry, t1, t2])


def make_straight(vehicle="car", length=40.0, halfwidth=2.5, slack=1.0) -> Track:
    spec = get_vehicle(vehicle)
    return Track.from_primitives(f"{vehicle}-test-straight", spec, "straight", "easy", 0,
                                 [Primitive("straight", length=length, halfwidth=halfwidth)], dock_slack=slack)


@pytest.fixture
def straight_car() -> Track:
    return make_straight("car", 40.0, 2.5)


@pytest.fixture
def wide_car() -> Track:
    return make_straight("car", 80.0, 25.0)


@pytest.fixture(scope="session")
def car_lturn() -> Track:
    return generate_track("car", "lturn", "medium", seed=11)


@pytest.fixture(scope="session")
def mini_relay(tmp_path_factory):
    return load_or_generate_relay("mini", 0, tmp_path_factory.mktemp("tracks"))
