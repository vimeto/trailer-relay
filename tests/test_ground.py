import math

import numpy as np
import pytest

from relay.ground import GroundField
from relay.sim import Leg, LegConfig


def test_field_is_deterministic_and_has_target_rms():
    a, b, c = GroundField("course|1"), GroundField("course|1"), GroundField("course|2")
    pts = np.random.default_rng(0).uniform(-200, 200, (4000, 2))
    va = np.array([a(x, y) for x, y in pts])
    assert np.array_equal(va, [b(x, y) for x, y in pts])
    assert not np.allclose(va, [c(x, y) for x, y in pts])
    assert math.degrees(np.sqrt(np.mean(va ** 2))) == pytest.approx(0.5, rel=0.25)
    assert abs(np.mean(va)) < 0.2 * np.sqrt(np.mean(va ** 2))


def test_zero_rms_is_pure_kinematics(straight_car):
    leg = Leg(straight_car, config=LegConfig(time_budget=500, ground_rms_deg_per_m=0.0))
    for _ in range(5):
        leg.drive(0, -2.0)
    assert leg.hitch == pytest.approx(0.0, abs=1e-12)


def test_uneven_ground_makes_blind_reversing_drift(straight_car):
    # wheels centred all the way: on ideal ground the hitch would stay at exactly 0
    leg = Leg(straight_car, config=LegConfig(time_budget=500))
    for _ in range(10):
        if leg.drive(0, -1.0).stop_reason != "completed":
            break
    assert abs(math.degrees(leg.hitch)) > 3.0


def test_same_commands_same_outcome(straight_car):
    cmds = [(0, -2.0), (10, -1.5), (-8, -2.0), (0, 1.0), (5, -3.0)]
    runs = []
    for _ in range(2):
        leg = Leg(straight_car, config=LegConfig(time_budget=500))
        for s, d in cmds:
            leg.drive(s, d)
        runs.append((leg.state.copy(), leg.t, leg.touches))
    assert np.array_equal(runs[0][0], runs[1][0]) and runs[0][1:] == runs[1][1:]
