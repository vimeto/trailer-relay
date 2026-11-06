import math

import numpy as np
import pytest

from relay.kinematics import critical_hitch_angle, hitch_angle, step, step_batch, trailer_axle, wrap
from relay.vehicles import VEHICLES, get_vehicle


def drive(state, steer, dist, spec, ds=0.01):
    n = int(abs(dist) / ds)
    for _ in range(n):
        state = step(state, steer, math.copysign(ds, dist), spec)
    return state


@pytest.mark.parametrize("key", list(VEHICLES))
def test_trailer_axle_has_no_lateral_slip(key):
    spec = get_vehicle(key)
    s = np.array([0.0, 0.0, 0.3, 0.1])
    for steer, d in [(0.3, 0.01), (-0.4, -0.01), (0.1, -0.01)]:
        a0 = trailer_axle(s, spec)
        s2 = step(s, steer, d, spec)
        a1 = trailer_axle(s2, spec)
        v = a1 - a0
        h = 0.5 * (s[3] + s2[3])
        lateral = -v[0] * math.sin(h) + v[1] * math.cos(h)
        assert abs(lateral) < 1e-5 * max(1.0, np.linalg.norm(v) * 100)
        s = s2


def test_vehicle_circle_radius():
    spec = get_vehicle("car")
    steer = math.radians(20)
    R = spec.wheelbase / math.tan(steer)
    s = drive(np.array([0.0, 0.0, 0.0, 0.0]), steer, 2 * math.pi * R, spec)
    assert np.hypot(s[0], s[1]) < 0.05
    assert abs(wrap(s[2])) < 0.01


def test_straight_reverse_stays_straight():
    spec = get_vehicle("car")
    s = drive(np.array([0.0, 0.0, 0.0, 0.0]), 0.0, -20.0, spec)
    assert abs(hitch_angle(s)) < 1e-12
    assert s[0] == pytest.approx(-20.0, abs=0.02)


def test_forward_is_stable_reverse_is_unstable():
    spec = get_vehicle("car")
    b0 = math.radians(20)
    fwd = drive(np.array([0.0, 0.0, b0, 0.0]), 0.0, 10.0, spec)
    assert abs(hitch_angle(fwd)) < 0.5 * b0
    b1 = math.radians(5)
    rev = drive(np.array([0.0, 0.0, b1, 0.0]), 0.0, -3.0, spec)
    assert abs(hitch_angle(rev)) > b1


def test_reversing_trailer_goes_opposite_to_steering():
    for spec in VEHICLES.values():
        s = drive(np.array([0.0, 0.0, 0.0, 0.0]), -spec.max_steer / 2, -0.5, spec)
        assert hitch_angle(s) > 0, spec.key  # steer right -> trailer rear swings left (beta > 0)


@pytest.mark.parametrize("key", ["car", "mower"])
def test_critical_angle_semantics(key):
    spec = get_vehicle(key)
    bc = critical_hitch_angle(spec)
    assert bc < spec.max_hitch  # these two have a critical angle below the mechanical limit
    steers = np.linspace(-spec.max_steer, spec.max_steer, 21)
    above = math.radians(math.degrees(bc) + 3)
    for st in steers:  # beyond critical: no steering can reduce |beta| while reversing
        s = step(np.array([0.0, 0.0, above, 0.0]), st, -0.02, spec)
        assert abs(hitch_angle(s)) > above
    below = math.radians(math.degrees(bc) - 3)
    best = min(abs(hitch_angle(step(np.array([0.0, 0.0, below, 0.0]), st, -0.02, spec))) for st in steers)
    assert best < below


def test_short_trailers_have_smaller_critical_angle():
    assert critical_hitch_angle(get_vehicle("mower")) < critical_hitch_angle(get_vehicle("truck"))
    assert critical_hitch_angle(get_vehicle("truck")) == pytest.approx(get_vehicle("truck").max_hitch)


def test_batch_matches_scalar():
    spec = get_vehicle("tractor")
    rng = np.random.default_rng(0)
    for _ in range(20):
        st = rng.normal(size=4)
        steer, d = rng.uniform(-0.6, 0.6), rng.uniform(-0.2, 0.2)
        a = step(st, steer, d, spec)
        b = step_batch(*(np.array([v]) for v in st), np.array([steer]), d, spec)
        assert np.allclose(a, np.array(b).ravel())


@pytest.mark.parametrize("key", ["car", "mower", "tractor", "truck"])
@pytest.mark.parametrize("beta0", [12.0, -12.0])
def test_prompt_sign_rule_matches_physics(key, beta0):
    """prompts.SYSTEM_PROMPT says: while reversing, positive steer drives the hitch angle down, negative drives it up."""
    import math
    import numpy as np
    from relay.kinematics import step, hitch_angle
    from relay.vehicles import VEHICLES
    spec = VEHICLES[key]
    s0 = np.array([0.0, 0.0, 0.0, -math.radians(beta0)])
    delta = math.copysign(spec.max_steer, beta0)  # steer towards the side the trailer swung to
    s = s0.copy()
    for _ in range(20):
        s = step(s, delta, -0.01, spec)
    assert abs(hitch_angle(s)) < abs(math.radians(beta0))
    s = s0.copy()
    for _ in range(20):
        s = step(s, -delta, -0.01, spec)
    assert abs(hitch_angle(s)) > abs(math.radians(beta0))
