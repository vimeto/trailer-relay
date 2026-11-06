"""Intiaanikäännös (turnaround) courses: geometry, the two phases, certification, observation and prompt."""
import json
import math

import numpy as np
import pytest

from relay.autopilot import Autopilot
from relay.geometry import trailer_corners, vehicle_corners
from relay.kinematics import wrap
from relay.observe import observe, render_report
from relay.seeder import RELAYS, generate_track, relay_from_dict, relay_to_dict
from relay.sim import Leg, LegConfig
from relay.track import Track
from relay.turnaround import TurnTrack, generate_turn_track
from relay.vehicles import VEHICLES


def straight_pose(spec, axle_xy, trailer_heading):
    """State with the trailer axle at axle_xy and the rig straight (vehicle ahead of the trailer)."""
    u = np.array([math.cos(trailer_heading), math.sin(trailer_heading)])
    rear = np.asarray(axle_xy) + (spec.trailer_length + spec.hitch_offset) * u
    return np.array([rear[0], rear[1], trailer_heading, trailer_heading])


@pytest.mark.parametrize("vehicle", list(VEHICLES))
def test_layout_is_valid(vehicle):
    tr = generate_turn_track(vehicle, seed=3, certify=False)
    leg = Leg(tr)
    assert tr.task == "turnaround" and tr.n_phases == 2
    assert leg._collision(leg.state) is None  # start pose is clear
    assert not tr.phase_done(leg.state, leg.spec, 0)
    g = tr.grid(0)
    for i in (0, 1):  # both target zones lie inside the kerbs
        zc = tr.target_corners(i)
        zc = zc + 0.03 * (zc.mean(0) - zc)
        assert (g.clearance(zc[:, 0], zc[:, 1]) > -0.3).all()
    # the reference poses complete their phases
    geo = tr.geo
    st1 = straight_pose(leg.spec, geo["goal1"], geo["headings"][0])
    assert tr.phase_done(st1, leg.spec, 0)
    sp = leg.spec
    h2 = geo["headings"][1]
    st2 = np.array([*geo["goal2"], h2, h2])
    assert tr.phase_done(st2, sp, 1)
    # ... and are clear of the kerbs
    for st in (st1, st2):
        pts = np.concatenate([vehicle_corners(st, sp), trailer_corners(st, sp)])
        assert (g.clearance(pts[:, 0], pts[:, 1]) > 0).all()
    # progress is monotone through the reference poses and goal_s matches phase 2's reference
    s1, _ = tr.progress_of(st1, sp, 0)
    s2, _ = tr.progress_of(st2, sp, 1)
    assert 0 < s1 < s2 and s2 == pytest.approx(tr.goal_s, abs=0.05)


def test_serialisation_round_trip():
    tr = generate_turn_track("car", seed=4, certify=False)
    d = json.loads(json.dumps(tr.to_dict()))
    assert d["template"] == "turn" and {z["name"] for z in d["geometry"]["zones"]} == {"side road", "exit zone"}
    tr2 = Track.from_dict(d)
    assert isinstance(tr2, TurnTrack)
    assert np.allclose(tr2.geo["walls"], tr.geo["walls"]) and tr2.goal_s == pytest.approx(tr.goal_s)
    assert np.allclose(tr2.start_state, tr.start_state)


def test_mirrored_and_rotated_layouts_agree():
    from relay.turnaround import turn_params
    from relay.vehicles import get_vehicle

    sp = get_vehicle("car")
    p = turn_params(sp)
    a = TurnTrack.build("a", sp, "medium", 0, p, 1, 0.0)
    b = TurnTrack.build("b", sp, "medium", 0, p, -1, 1.1)
    assert a.goal_s == pytest.approx(b.goal_s)
    # the side road is on the driver's left when side=+1 (start heading east, side road to the north)
    assert a.zone_corners(0)[:, 1].min() > a.start_state[1]
    # mirrored + rotated: the side road lies on the driver's right
    st = b.start_state
    rel = b.geo["zones"][0]["center"] - st[:2]
    assert -math.sin(st[2]) * rel[0] + math.cos(st[2]) * rel[1] < 0


def test_autopilot_does_both_phases():
    tr = generate_turn_track("car", seed=0, certify=False)
    leg = Leg(tr, config=LegConfig(time_budget=1e5, max_commands=200, max_touches=0))
    Autopilot().run_leg(leg)  # dispatches to the two-phase driver
    m = leg.metrics()
    assert m["outcome"] == "FINISHED" and m["touches"] == 0 and m["phase"] == 2 and m["phases"] == 2
    ph = [e for e in leg.events if e["type"] == "phase"]
    assert len(ph) == 1 and ph[0]["phase"] == 2 and ph[0]["of"] == 2 and ph[0]["label"]
    fin = next(e for e in leg.events if e["type"] == "finish")
    assert fin["t"] > ph[0]["t"]
    assert abs(wrap(leg.state[2] - tr.geo["headings"][1])) <= math.radians(20)


def test_generate_track_dispatch_and_certification():
    tr = generate_track("mower", "turn", "medium", seed=0)
    assert isinstance(tr, TurnTrack) and tr.meta["certified"] and tr.par["touches"] == 0
    assert RELAYS["turnaround"] and all(t == "turn" for _, t, _ in RELAYS["turnaround"])
    rel = relay_from_dict(json.loads(json.dumps(relay_to_dict({"name": "x", "seed": 0, "id": "x-s0", "tracks": [tr]}))))
    assert isinstance(rel["tracks"][0], TurnTrack) and rel["tracks"][0].par == tr.par


def test_observation_is_phase_aware():
    tr = generate_turn_track("car", seed=0, certify=False)
    leg = Leg(tr)
    o = observe(leg, "sketch")
    assert o["phase"] == {"index": 1, "of": 2, "task": tr.phase_info(0)["name"], "direction": "reverse", "goal": tr.phase_info(0)["goal"]}
    assert o["route"]["direction"] == "reverse" and "point_uv" in o["route"]
    m = o["local_map"]
    assert all(v > 0 for _, v in m["left_rail"]) and all(v <= 0 for _, v in m["right_rail"])
    assert len(m["bay_corners"]) == 4
    txt = render_report(o)
    assert "PHASE 1/2 (REVERSE)" in txt and "SIDE ROAD" in txt and "kerb on your LEFT" in txt and "B target zone" in txt
    # jump to the phase-1 reference pose: the next check flips to phase 2
    leg.state = straight_pose(leg.spec, tr.geo["goal1"], tr.geo["headings"][0])
    assert not leg._goal_reached() and leg.phase == 1
    o2 = observe(leg, "local")
    assert o2["phase"]["index"] == 2 and o2["phase"]["direction"] == "forward" and o2["route"]["direction"] == "forward"
    assert "EXIT ZONE" in render_report(o2)


def test_standard_observation_unchanged():
    """The turnaround branches must not change what standard-course drivers see."""
    tr = generate_track("car", "lturn", "medium", seed=2, certify=False)
    o = observe(Leg(tr), "local")
    assert "phase" not in o and set(o["route"]) == {"distance", "bearing_deg"}
    txt = render_report(o)
    assert "PHASE" not in txt and "rail on your LEFT" in txt and "trailer has reversed" in txt


def test_turnaround_prompt_and_briefing():
    from relay.prompts import briefing, system_prompt

    for args in (("local", "tools"), ("local", "code", "continuous", "realtime"), ("cone", "tools", "continuous")):
        std, turn = system_prompt(*args), system_prompt(*args, task="turnaround")
        assert std == system_prompt(*args, task="relay")
        assert "intiaanikäännös" in turn and "PHASE 1 (REVERSE)" in turn and "THE TURNAROUND" in turn
        assert "trailer" in turn and " rail" not in turn and "handoff bay" not in turn
    assert "obs also has phase{" in system_prompt("local", "code", "continuous", task="turnaround")
    tr = generate_turn_track("car", seed=0, certify=False)
    b = briefing(Leg(tr), 0, 4, "R", control="continuous")
    assert "intiaanikäännös" in b and ("on your LEFT" in b) == (tr.side > 0) and "facing away from the exit zone" in b
