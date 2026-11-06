import math

import numpy as np
import pytest

from relay.kinematics import critical_hitch_angle
from relay.sim import Leg, LegConfig

from .conftest import make_straight, state_for_trailer_at


def move_time(spec, D, v):
    """Trapezoidal speed profile (triangular for short moves)."""
    return D / v + v / spec.accel if D >= v * v / spec.accel else 2 * math.sqrt(D / spec.accel)


def test_straight_reverse_accounting(straight_car):
    leg = Leg(straight_car, config=LegConfig(time_budget=500))
    s0 = leg.progress_s
    r = leg.drive(0, -3.0)
    assert r.stop_reason == "completed" and r.traveled == pytest.approx(3.0, abs=1e-6)
    assert leg.t == pytest.approx(1.0 + move_time(leg.spec, 3.0, leg.spec.speed_rev), abs=1e-6)
    assert leg.progress_s - s0 == pytest.approx(3.0, abs=0.02)
    assert leg.status == "REVERSING" and leg.commands == 1


def test_shuttle_costs_time(straight_car):
    leg = Leg(straight_car, config=LegConfig(time_budget=500))
    leg.drive(0, -1.0)
    t = leg.t
    leg.drive(0, 1.0)
    assert leg.shuttles == 1
    assert leg.t - t == pytest.approx(1.0 + leg.spec.shuttle_s + move_time(leg.spec, 1.0, leg.spec.speed_fwd), abs=1e-6)
    assert any(e["type"] == "shuttle" for e in leg.events)


def test_steering_slews_and_dry_steering(straight_car):
    leg = Leg(straight_car)
    leg.drive(30, -0.5)
    assert math.degrees(leg.steer) == pytest.approx(15.0, abs=0.6)  # 30 deg/m for the car
    t = leg.t
    r = leg.drive(-30, 0)
    assert math.degrees(leg.steer) == pytest.approx(-30.0)
    assert r.traveled == 0 and leg.t - t == pytest.approx(1.0 + 45.0 / leg.spec.steer_rate_deg_per_s, abs=1e-6)


def test_clamping_and_bad_input(straight_car):
    leg = Leg(straight_car)
    r = leg.drive(100, -50)
    assert r.steer_deg_applied == pytest.approx(34.0) and r.distance_applied == -10.0
    assert len(r.notes) == 2
    r = leg.drive(float("nan"), "x")
    assert r.distance_applied == 0 and not leg.done


def test_contact_pins_and_blocks_then_recovers(straight_car):
    leg = Leg(straight_car, config=LegConfig(time_budget=1000))
    r = leg.drive(34, 10)  # full left lock forward: the nose swings into the rail
    assert r.stop_reason == "contact" and r.contact is not None
    assert leg.touches == 1 and leg.status == "PINNED"
    assert r.contact["rail"] in ("rail", "start wall") and r.contact["body"] == "vehicle"
    t = leg.t
    r2 = leg.drive(34, 10)  # push into the same rail again
    assert r2.stop_reason == "blocked" and r2.traveled < 0.05
    assert leg.touches == 1 and leg.blocked == 1 and leg.t > t
    r3 = leg.drive(34, -2.0)  # back away along the same arc
    assert r3.traveled > 0.5 and leg.status != "PINNED"
    assert [e["type"] for e in leg.events].count("contact") == 1


def test_jackknife_blocks_reverse_until_pulling_forward(wide_car):
    leg = Leg(wide_car, config=LegConfig(time_budget=1000))
    spec = leg.spec
    leg.state = state_for_trailer_at(wide_car, 20.0, beta=math.radians(spec.max_hitch_deg - 4))
    r = leg.drive(-spec.max_steer_deg, -5)  # steering right folds beta>0 further
    assert r.stop_reason == "jackknife" and leg.jackknifes == 1 and leg.status == "JACKKNIFED"
    assert abs(math.degrees(leg.hitch)) < spec.max_hitch_deg
    r = leg.drive(-spec.max_steer_deg, -5)
    assert r.stop_reason == "blocked" and leg.jackknifes == 1
    r = leg.drive(0, 6)
    assert r.traveled == pytest.approx(6) and leg.status != "JACKKNIFED"


def test_committed_state_for_car(wide_car):
    leg = Leg(wide_car, config=LegConfig(time_budget=1000))
    bc = critical_hitch_angle(leg.spec)
    leg.state = state_for_trailer_at(wide_car, 20.0, beta=bc + math.radians(4))
    b0 = abs(leg.hitch)
    leg.drive(leg.spec.max_steer_deg, -0.3)  # even full counter-steer cannot recover
    assert leg.status == "COMMITTED" and abs(leg.hitch) > b0
    assert leg.stall_time["COMMITTED"] >= 0


def test_finishes_when_parked(straight_car):
    leg = Leg(straight_car)
    leg.state = state_for_trailer_at(straight_car, straight_car.goal_s - 0.05)
    leg._s_hint = None
    r = leg.drive(0, -0.05)
    assert leg.outcome == "FINISHED" and r.stop_reason == "finished"
    assert leg.drive(0, -1).stop_reason == "rejected"
    assert leg.events[-1]["type"] == "finish"


def test_not_parked_when_misaligned():
    tr = make_straight("car", 40, 4.0, slack=2.0)
    leg = Leg(tr)
    leg.state = state_for_trailer_at(tr, tr.goal_s, beta=0)
    leg.state[3] += math.radians(20)  # trailer twisted 20 deg in the bay
    leg.state[2] += math.radians(20)
    from relay.geometry import trailer_corners
    assert not tr.is_parked(trailer_corners(leg.state, leg.spec), leg.state[3])


def test_time_budget_and_command_limits(straight_car):
    leg = Leg(straight_car, config=LegConfig(time_budget=4.0))
    leg.drive(0, -10)
    assert leg.outcome == "TIME_UP" and leg.t == pytest.approx(4.0, abs=0.05)
    leg = Leg(straight_car, config=LegConfig(time_budget=1000, max_commands=3))
    for _ in range(3):
        leg.drive(0, 0)
    assert leg.outcome == "DNF" and "command limit" in leg.outcome_reason


def test_touch_limit_disqualifies(straight_car):
    leg = Leg(straight_car, config=LegConfig(time_budget=1000, max_touches=1))
    for _ in range(5):
        if leg.done:
            break
        leg.drive(34, 10)
        leg.drive(34, -1.5)
    assert leg.outcome == "DNF" and leg.touches == 2


def test_trace_is_time_ordered_and_metrics(straight_car):
    leg = Leg(straight_car)
    for s, d in [(0, -3), (10, -2), (-10, 1)]:
        leg.drive(s, d)
    t = np.array(leg.trace["t"])
    assert (np.diff(t) >= 0).all() and len(t) > 10
    assert len({len(v) for v in leg.trace.values()}) == 1
    m = leg.metrics()
    assert m["outcome"] == "RUNNING" and m["odometer"] == pytest.approx(6.0, abs=1e-6)
    assert 0 < m["progress"] < 1 and m["short_m"] > 0


def test_thinking_can_cost_sim_time(straight_car):
    leg = Leg(straight_car, config=LegConfig(time_budget=100, think_s_per_1k_tokens=2.0))
    leg.charge_thinking(1500)
    assert leg.t == pytest.approx(3.0)
    assert leg.events[-1]["type"] == "think"


def test_speed_is_capped_and_faster_is_quicker(straight_car):
    sp = straight_car and Leg(straight_car).spec
    slow = Leg(straight_car, config=LegConfig(time_budget=500))
    r1 = slow.drive(0, -6.0, speed_mps=0.6)
    fast = Leg(straight_car, config=LegConfig(time_budget=500))
    r2 = fast.drive(0, -6.0, speed_mps=99)
    assert r1.speed_mps == pytest.approx(0.6) and r2.speed_mps == pytest.approx(sp.speed_rev_max)
    assert any("capped" in n for n in r2.notes)
    assert r2.time_cost < r1.time_cost
    assert r1.time_cost == pytest.approx(1.0 + move_time(sp, 6.0, 0.6), abs=1e-6)
    assert max(fast.trace["v"]) == 0 and min(fast.trace["v"]) == pytest.approx(-sp.speed_rev_max)


def test_steering_slews_per_second_so_speed_costs_authority(wide_car):
    out = {}
    for v in (0.5, 3.0):
        leg = Leg(wide_car, config=LegConfig(time_budget=1000, ground_rms_deg_per_m=0))
        leg.drive(30, -1.0, speed_mps=v)
        out[v] = math.degrees(leg.steer)
    sp = Leg(wide_car).spec
    assert out[0.5] == pytest.approx(min(30, sp.steer_rate_deg_per_s * 1.0 / 0.5), abs=0.5)
    assert out[3.0] == pytest.approx(sp.steer_rate_deg_per_s * 1.0 / 3.0, abs=0.5)
    assert out[3.0] < out[0.5]


def test_invalid_speed_falls_back_to_default(straight_car):
    leg = Leg(straight_car, config=LegConfig(time_budget=500))
    r = leg.drive(0, -1.0, speed_mps="fast")
    assert r.speed_mps == pytest.approx(leg.spec.speed_rev)
    r = leg.drive(0, -1.0, speed_mps=-2)
    assert r.speed_mps == pytest.approx(leg.spec.speed_rev)


def test_look_turns_gaze_and_costs_time(straight_car):
    leg = Leg(straight_car, config=LegConfig(time_budget=500))
    assert leg.gaze_deg == 180.0
    r = leg.look(90)
    assert r["ok"] and leg.gaze_deg == 90.0
    assert r["time_cost"] == pytest.approx(leg.cfg.look_overhead_s + 90 / leg.spec.head_rate_deg_per_s)
    leg.look(-170)  # shortest way round from +90 is 100 deg
    assert leg.t == pytest.approx(r["time_cost"] + leg.cfg.look_overhead_s + 100 / leg.spec.head_rate_deg_per_s)
    assert leg.look(540)["gaze_deg"] == 180.0
    assert [e["yaw_deg"] for e in leg.events if e["type"] == "look"] == [90.0, -170.0, 180.0]
    assert leg.commands == 0
