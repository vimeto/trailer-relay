import json
import math

import numpy as np
import pytest

from relay.geometry import segments_intersect, trailer_corners
from relay.seeder import DIFFICULTY, TEMPLATES, generate_track, relay_from_dict, relay_to_dict, swept_band
from relay.sim import Leg
from relay.track import Primitive, Track, build_centerline
from relay.vehicles import VEHICLES, get_vehicle

from .conftest import state_for_trailer_at


def test_build_centerline_straight_and_arc():
    pts, hs, hws, s = build_centerline([Primitive("straight", length=10, halfwidth=2), Primitive("arc", radius=5, angle=math.pi / 2, halfwidth=2)])
    assert s[-1] == pytest.approx(10 + 5 * math.pi / 2, rel=1e-3)
    assert np.allclose(pts[-1], [15, 5], atol=1e-6)
    assert hs[-1] == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("template", TEMPLATES)
def test_generation_is_deterministic_and_seed_dependent(template):
    a = generate_track("car", template, "medium", seed=5, certify=False)
    b = generate_track("car", template, "medium", seed=5, certify=False)
    c = generate_track("car", template, "medium", seed=6, certify=False)
    assert np.array_equal(a.centerline, b.centerline)
    assert not np.allclose(a.centerline[-1], c.centerline[-1])


@pytest.mark.parametrize("vehicle", list(VEHICLES))
@pytest.mark.parametrize("template", TEMPLATES)
def test_generated_geometry_is_valid(vehicle, template):
    tr = generate_track(vehicle, template, "medium", seed=2, certify=False)
    L, R = tr.left_rail, tr.right_rail
    # the two rails never cross each other
    assert not segments_intersect(np.stack([L[:-1], L[1:]], 1), np.stack([R[:-1], R[1:]], 1)).any()
    # the start pose is collision-free and not already parked
    leg = Leg(tr)
    assert leg._collision(leg.state) is None
    assert not tr.is_parked(trailer_corners(leg.state, leg.spec), leg.state[3])
    # the bay is inside the corridor and the goal pose is a parked pose
    g = tr.grid()
    dc = tr.dock_corners
    dc = dc + 0.03 * (dc.mean(0) - dc)  # just inside the bay (the corners sit on the rails/wall)
    assert (g.clearance(dc[:, 0], dc[:, 1]) > -0.3).all()
    st = state_for_trailer_at(tr, tr.goal_s)
    assert tr.is_parked(trailer_corners(st, leg.spec), st[3])


def test_curves_flare_for_offtracking():
    spec = get_vehicle("truck")
    r_in, r_out = swept_band(spec, 25.0)
    assert r_out - r_in > spec.trailer_width + 1.0  # the semi sweeps much more than its width
    tr = generate_track("truck", "lturn", "medium", seed=1, certify=False)
    arc_hw = [p.halfwidth for p in tr.primitives if p.kind == "arc"][0]
    straight_hw = tr.primitives[0].halfwidth
    assert arc_hw > straight_hw


def test_json_roundtrip(car_lturn):
    d = json.loads(json.dumps(car_lturn.to_dict()))
    t2 = Track.from_dict(d)
    assert np.allclose(t2.centerline, car_lturn.centerline, atol=1e-3)
    assert t2.par == car_lturn.par and t2.goal_s == pytest.approx(car_lturn.goal_s, abs=1e-3)
    # rebuilding from primitives alone (no geometry) yields the same course
    d.pop("geometry")
    t3 = Track.from_dict(d)
    assert np.allclose(t3.centerline, car_lturn.centerline, atol=1e-6)


def test_certification_records_par(car_lturn):
    assert car_lturn.meta["certified"] is True
    assert car_lturn.par["outcome"] == "FINISHED" and car_lturn.par["touches"] == 0
    assert car_lturn.par["time"] > 0


def test_relay_roundtrip(mini_relay):
    d = relay_to_dict(mini_relay)
    r2 = relay_from_dict(json.loads(json.dumps(d)))
    assert r2["id"] == mini_relay["id"] and len(r2["tracks"]) == len(mini_relay["tracks"])
    assert all(t.meta.get("certified") for t in mini_relay["tracks"])


def test_difficulty_orders_widths():
    ws = [generate_track("car", "straight", d, seed=1, certify=False).halfwidth.mean() for d in ("easy", "medium", "hard")]
    assert ws[0] > ws[1] > ws[2]
    assert set(DIFFICULTY) == {"easy", "medium", "hard"}


@pytest.mark.slow
@pytest.mark.parametrize("vehicle", ["car", "tractor", "mower"])
@pytest.mark.parametrize("template", TEMPLATES)
def test_every_template_certifies(vehicle, template):
    tr = generate_track(vehicle, template, "medium", seed=3)
    assert tr.meta["certified"], (vehicle, template, tr.meta)
