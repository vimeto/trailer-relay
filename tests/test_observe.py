import math

import numpy as np
import pytest

from relay.observe import LEVELS, describe_result, local_map, observe, render_report, sensors
from relay.sim import Leg

from .conftest import make_straight, state_for_trailer_at


def test_levels_control_information(straight_car):
    leg = Leg(straight_car)
    o_s, o_l, o_k, o_f, o_c, o_cs = (observe(leg, lv) for lv in LEVELS if lv != "driver")
    assert "local_map" not in o_s and "global" not in o_s
    assert "local_map" in o_l and "global" not in o_l
    assert "global" in o_f
    # vision-only levels: no parking sensors or route arrow, a view cone instead
    for o in (o_c, o_cs):
        assert "sensors" not in o and "route" not in o and o["view"]["gaze_deg"] == 180.0
    assert "sketch" in o_cs and "sketch" not in o_c
    with pytest.raises(ValueError):
        observe(leg, "god-mode")
    # driver's seat (its own leg: observing at this level switches the leg to the human head model)
    o_d = observe(Leg(straight_car), "driver")
    assert "local_map" not in o_d and "sensors" not in o_d and "route" not in o_d and "head" in o_d


def test_local_map_is_relative_and_limited(car_lturn):
    leg = Leg(car_lturn)
    m = local_map(leg)
    pts = np.array(m["left_rail"] + m["right_rail"])
    assert len(pts) > 5
    R = leg.spec.view_radius
    assert np.hypot(pts[:, 0], pts[:, 1]).max() <= R + leg.spec.combo_length
    # no absolute world coordinates anywhere in the default report
    o = observe(leg, "local")
    flat = str(o)
    x, y = leg.state[:2]
    assert f"{x:.2f}" not in flat and f"{y:.2f}" not in flat


def test_rear_frame_orientation_on_centred_straight():
    tr = make_straight("car", 60, 3.0)
    leg = Leg(tr)
    m = local_map(leg)
    L, Rr = np.array(m["left_rail"]), np.array(m["right_rail"])
    assert np.allclose(L[:, 1], 3.0, atol=0.06) and np.allclose(Rr[:, 1], -3.0, atol=0.06)
    assert L[:, 0].max() > 5  # rail continues behind the trailer (positive u)
    s = sensors(leg)
    expect = 3.0 - leg.spec.trailer_width / 2
    assert s["trailer"]["left"] == pytest.approx(expect, abs=0.02)
    assert s["trailer"]["right"] == pytest.approx(expect, abs=0.02)
    assert s["trailer"]["rear"] == s["range"]  # nothing straight behind within range


def test_bay_appears_only_when_near():
    tr = make_straight("car", 70, 3.0)
    leg = Leg(tr)
    assert "bay_corners" not in local_map(leg)
    leg.state = state_for_trailer_at(tr, tr.goal_s - 4.0)
    leg._s_hint = None
    m = local_map(leg)
    assert "bay_corners" in m and len(m["bay_corners"]) == 4
    wall_u = np.array(m["dock_wall"])[:, 0]
    assert (wall_u > 0).all()  # the dock wall is behind the trailer


def test_report_text(straight_car):
    leg = Leg(straight_car)
    r = leg.drive(-20, -1.5)
    txt = render_report(observe(leg), r)
    for key in ("RESULT:", "STATUS:", "CLOCK:", "HITCH ANGLE:", "PROXIMITY", "LOCAL MAP", "ROUTE:"):
        assert key in txt
    assert "swung to your LEFT" in txt  # steering right in reverse swings the trailer left
    assert "reversed 1.50 m" in txt


def test_contact_described(straight_car):
    leg = Leg(straight_car)
    r = leg.drive(34, 10)
    s = describe_result(r)
    assert "STOPPED: contact" in s and "corner touched the" in s


def test_sketch_shows_rig_rails_and_frame(straight_car):
    from relay.observe import sketch
    leg = Leg(straight_car)
    sk = sketch(leg)
    txt = "\n".join(sk["rows"])
    assert "+" in txt and "T" in txt and "V" in txt and "#" in txt
    rows = sk["rows"]
    r_plus = next(i for i, r in enumerate(rows) if "+" in r)
    r_v = max(i for i, r in enumerate(rows) if "V" in r)
    assert r_v > r_plus  # vehicle is drawn below (in front of) the trailer rear; reversing goes up
    # straight course, rig centred: rails appear on both sides of the trailer column
    row = rows[r_plus]
    c = row.index("+")
    assert "#" in row[:c] and "#" in row[c:]
    rep = render_report(observe(leg, "sketch"))
    assert "SKETCH" in rep and "SKETCH" not in render_report(observe(leg, "local"))


def test_cone_only_reports_what_the_gaze_covers(car_lturn):
    from relay.observe import visible
    leg = Leg(car_lturn)
    back = local_map(leg, cone=True)
    n_back = len(back["left_rail"]) + len(back["right_rail"])
    assert n_back > 0 and back["cone"]
    leg.look(0)  # look straight ahead, away from the course behind the trailer
    ahead = local_map(leg, cone=True)
    # what is seen ahead is disjoint from what is seen behind: every listed point is inside the new cone
    assert ahead["left_rail"] + ahead["right_rail"] != back["left_rail"] + back["right_rail"]
    from relay.observe import _rail_samples
    for pts in _rail_samples(leg):
        v = visible(leg, pts)
        d = np.hypot(*(pts - leg.state[:2]).T)
        assert not (v & (d > leg.spec.view_range + 0.4 * leg.spec.body_front + 1e-6)).any()


def test_cone_sketch_masks_unseen_cells(car_lturn):
    leg = Leg(car_lturn)
    rows = observe(leg, "cone-sketch")["sketch"]["rows"]
    txt = "".join(rows)
    assert "?" in txt and "T" in txt and "V" in txt and "#" in txt
    leg.look(0)
    rows2 = observe(leg, "cone-sketch")["sketch"]["rows"]
    assert "".join(rows2).count("?") > txt.count("?")  # the course is behind; looking ahead hides most of it
    rep = render_report(observe(leg, "cone-sketch"))
    assert "VIEW: you are looking straight ahead" in rep and "PROXIMITY" not in rep and "? not in view" in rep
