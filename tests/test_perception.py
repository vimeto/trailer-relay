"""Driver's-seat perception (relay/perception.py), asymmetric courses and the naive / replay baselines."""
import math

import numpy as np
import pytest

from relay.observe import observe, render_report
from relay.perception import GLANCE_TOL_DEG, attach, perceive, rig_for, wall_points
from relay.prompts import briefing, system_prompt, tools_for
from relay.seeder import generate_track
from relay.sim import Leg
from relay.track import Track
from relay.vehicles import VEHICLES


@pytest.fixture(scope="module")
def car_track():
    return generate_track("car", "straight", "easy", seed=3, certify=False)


@pytest.fixture(scope="module")
def truck_track():
    return generate_track("truck", "straight", "easy", seed=3, certify=False)


def settle(leg, gaze, t=1.5):
    leg.set_gaze(gaze)
    leg.control_step(t)
    return observe(leg, "driver")


def test_rigs_are_plausible():
    for k, sp in VEHICLES.items():
        r = rig_for(sp)
        assert r.gaze_min < 0 < r.gaze_max
        for m in r.mirrors:
            assert abs(m.axis_deg) > 150  # mirrors look backwards
            assert (m.pos[1] > 0) == (m.name == "left_mirror")
    assert not rig_for(VEHICLES["mower"]).mirrors and rig_for(VEHICLES["mower"]).screen is None
    assert rig_for(VEHICLES["tractor"]).screen is None
    assert rig_for(VEHICLES["car"]).screen is not None and rig_for(VEHICLES["truck"]).cab_block_deg


def test_driver_obs_has_no_privileged_map(car_track):
    leg = Leg(car_track)
    o = observe(leg, "driver")
    for k in ("local_map", "sensors", "route", "global", "sketch"):
        assert k not in o
    assert {"head", "rig", "views", "target"} <= set(o)
    assert o["head"]["gaze_deg"] == 0.0  # starts looking ahead


def test_mirror_and_camera_only_when_looked_at(car_track):
    leg = Leg(car_track)
    rig = attach(leg)
    o = settle(leg, 0.0)
    assert set(o["views"]) == {"direct", "periphery"}
    o = settle(leg, rig.mirror_gaze["left_mirror"])
    assert o["head"]["looking_at"] == "left_mirror" and "left_mirror" in o["views"]
    assert all(abs(b) > 150 for b, _ in o["views"]["left_mirror"]["walls"])  # it shows what is behind
    o = settle(leg, rig.mirror_gaze["left_mirror"] + GLANCE_TOL_DEG + 3)
    assert "left_mirror" not in o["views"]
    o = settle(leg, rig.screen_gaze)
    assert "camera" in o["views"] and o["views"]["camera"]["walls"]


def test_head_has_limits_and_does_not_wrap(car_track):
    leg = Leg(car_track)
    rig = attach(leg)
    leg.set_gaze(170)
    assert leg.gaze_target == rig.gaze_max
    leg.control_step(2.0)
    leg.set_gaze(-150)
    gz = []
    for _ in range(30):
        leg.control_step(0.05)
        gz.append(leg.gaze_deg)
    # from +160 to -150 the head sweeps through 0 (310 degrees at the head rate), not across the back
    assert any(abs(g) < 20 for g in gz)
    assert all(rig.gaze_min - 1e-6 <= g <= rig.gaze_max + 1e-6 for g in gz)


def test_nothing_seen_while_turning(car_track):
    leg = Leg(car_track)
    attach(leg)
    leg.set_gaze(150)
    leg.control_step(0.2)  # 240 deg/s: still on its way
    o = observe(leg, "driver")
    assert not o["head"]["settled"] and o["views"] == {}
    assert "STILL TURNING" in render_report(o)


def test_trailer_blocks_the_view_and_truck_cab_blocks_the_back(car_track, truck_track):
    leg = Leg(car_track)
    attach(leg)
    o = settle(leg, 160)
    behind = [b for b, d in o["views"]["direct"]["walls"] if abs(b) > 177 and d > 6]
    assert not behind  # straight back, beyond the trailer, is hidden by the trailer body
    leg = Leg(truck_track)
    rig = attach(leg)
    o = settle(leg, rig.gaze_max)
    assert all(abs(b) <= rig.cab_block_deg + 1e-6 for b, _ in o["views"]["direct"]["walls"])


def test_distance_error_grows_with_range_and_is_deterministic(car_track):
    def errors():
        leg = Leg(car_track)
        rig = attach(leg, seed=7)
        out = []
        for g in (rig.mirror_gaze["left_mirror"], 0.0, rig.mirror_gaze["right_mirror"]):
            o = settle(leg, g, 1.0)
            W = wall_points(car_track, leg.spec)
            fr_p, h = leg.state[:2], leg.state[2]
            R = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
            for v in o["views"].values():
                org = fr_p + R @ np.array(v["origin"])
                for b, d in v["walls"]:
                    u = np.array([math.cos(h + math.radians(b)), math.sin(h + math.radians(b))])
                    # true distance of the nearest wall point along that bearing
                    rel = W - org
                    along = rel @ u
                    perp = np.abs(rel @ np.array([-u[1], u[0]]))
                    ok = (along > 0) & (perp < 0.6)
                    if ok.any():
                        dt = float(along[ok][np.argmin(perp[ok])])
                        out.append((dt, abs(d - dt)))
        return out
    a, b = errors(), errors()
    assert a == b
    near = [e for d, e in a if d < 4]
    far = [e for d, e in a if d > 9]
    assert near and far and np.median(far) > 2 * np.median(near)


def test_target_line_of_sight(car_track):
    leg = Leg(car_track)
    attach(leg)
    o = observe(leg, "driver")
    # straight course, rig in line with the bay: the trailer body hides it from the driver's seat
    assert o["target"]["in_sight"] is False


def test_driver_prompt_and_briefing(car_track):
    s = system_prompt("driver", "code", "continuous", "paused")
    assert "YOUR EYES" in s and "local_map" not in s and "head{gaze_deg" in s
    assert "what you see" in s
    assert [t.name for t in tools_for("driver", "tools", "continuous")] == ["drive", "look"]
    leg = Leg(car_track)
    attach(leg)
    b = briefing(leg, 0, 4, "R", control="continuous", obs_level="driver")
    assert "your eyes:" in b and "left mirror at gaze" in b and "reversing camera: screen" in b
    assert "your eyes" not in briefing(leg, 0, 4, "R", control="continuous")


def test_standard_prompt_untouched():
    for lvl in ("local", "cone"):
        assert "YOUR EYES" not in system_prompt(lvl, "code", "continuous", "paused")


# ---------------------------------------------------------------------------- asymmetric courses
@pytest.fixture(scope="module")
def asym_track():
    return generate_track("car", "lturn", "medium", seed=5, certify=False, asym=True)


def test_asym_track_is_wider_on_one_side_and_round_trips(asym_track):
    t = asym_track
    assert t.meta["asym"]["yard_side"] in (-1, 1)
    extra = (t.hw_l - t.halfwidth) + (t.hw_r - t.halfwidth)
    assert extra.max() > 1.5 * VEHICLES["car"].max_width and extra.min() >= -1e-9
    end_mid = 0.5 * (t.left_rail[-1] + t.right_rail[-1])
    assert np.linalg.norm(end_mid - t.centerline[-1]) > 0.7 * VEHICLES["car"].max_width  # rails' middle is off the bay
    t2 = Track.from_dict(t.to_dict())
    assert np.allclose(t2.left_rail, t.left_rail, atol=0.01) and np.allclose(t2.right_rail, t.right_rail, atol=0.01)
    g = t.grid()
    side = t.meta["asym"]["yard_side"]
    p = t.centerline[-5] + side * (t.halfwidth[-5] + 0.5) * t.normals[-5]  # inside the yard, outside the nominal corridor
    assert g.clearance(p[0], p[1]) > 0


def test_asym_track_certifies():
    t = generate_track("mower", "alley", "medium", seed=2, asym=True)
    assert t.meta["certified"] and t.par["outcome"] == "FINISHED"


def test_symmetric_tracks_unchanged():
    a = generate_track("car", "lturn", "medium", seed=5, certify=False)
    assert a.hw_left is None and "hw_left" not in a.to_dict()["geometry"]


# ---------------------------------------------------------------------------- baselines
def test_naive_baseline_drives_through_the_normal_pipeline(car_track):
    from relay.agent import AgentConfig
    from relay.runner import make_driver

    drv = make_driver("scripted:ctrl-camera", AgentConfig(obs_level="driver", mode="code", control="continuous"))
    leg = Leg(car_track)
    drv.run_leg(leg, 0, 1)
    assert leg.done and "gaze" in leg.trace and len(leg.trace["gaze"]) == len(leg.trace["t"])
    assert any(abs(g - rig_for(leg.spec).screen_gaze) < 1 for g in leg.trace["gaze"])


# ---------------------------------------------------------------------------- peripheral vision and levels
def test_periphery_is_close_rough_and_outside_the_direct_cone(car_track):
    leg = Leg(car_track)
    rig = attach(leg)
    o = settle(leg, 0.0)
    per = o["views"]["periphery"]
    assert per["walls"], "the rails beside the car are within reach of the corner of the eye"
    for b, d in per["walls"]:
        assert rig.fov_deg / 2 - 5 < abs(b) <= rig.periphery_deg + 15  # bearing noise is a few degrees
        assert d <= rig.periphery_range * 1.4 and abs(d / 0.25 - round(d / 0.25)) < 1e-6
    assert "CORNER OF YOUR EYE" in render_report(o)


def test_levels():
    from relay.perception import LEVELS

    car = VEHICLES["car"]
    assert rig_for(car, "v1").periphery_range == 0
    assert rig_for(car, "easy").camera_range > rig_for(car, "medium").camera_range > rig_for(car, "hard").camera_range
    assert LEVELS["easy"].noise < LEVELS["medium"].noise < LEVELS["hard"].noise


def test_v1_level_has_no_periphery(car_track):
    leg = Leg(car_track)
    attach(leg, level="v1")
    assert "periphery" not in settle(leg, 0.0)["views"]
    b = briefing(leg, 0, 4, "R", control="continuous", obs_level="driver")
    assert "corner of your eye" not in b
