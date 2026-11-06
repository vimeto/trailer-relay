import math

import numpy as np

from relay.geometry import perimeter_samples, points_in_rect, ray_distance, rect_corners, segments_intersect


def S(*pts):
    return np.array(pts, dtype=float).reshape(-1, 2, 2)


def test_segment_intersection_cases():
    a = S([[0, 0], [2, 2]])
    assert segments_intersect(a, S([[0, 2], [2, 0]]))[0, 0]  # crossing
    assert not segments_intersect(a, S([[0, 1], [1, 2]]))[0, 0]  # parallel, offset
    assert segments_intersect(a, S([[2, 2], [3, 0]]))[0, 0]  # touching at an endpoint
    assert not segments_intersect(a, S([[3, 3], [4, 0]]))[0, 0]  # disjoint
    m = segments_intersect(S([[0, 0], [1, 0]], [[0, 1], [1, 1]]), S([[0.5, -1], [0.5, 0.5]]))
    assert m.tolist() == [[True], [False]]


def test_ray_distance():
    wall = S([[5, -1], [5, 1]])
    assert ray_distance(np.array([0.0, 0]), np.array([1.0, 0]), wall, 10) == 5
    assert ray_distance(np.array([0.0, 0]), np.array([-1.0, 0]), wall, 10) == 10
    assert ray_distance(np.array([0.0, 0]), np.array([1.0, 0]), wall, 3) == 3
    assert ray_distance(np.array([0.0, 0]), np.array([1.0, 0]), np.zeros((0, 2, 2)), 7) == 7


def test_rect_corners_and_containment():
    c = rect_corners(1, 2, math.pi / 2, front=3, rear=1, half_w=0.5)
    assert np.allclose(c[0], [0.5, 5])  # front-left is +x... rotated: left of north is west
    assert np.allclose(c[2], [1.5, 1])  # rear-right
    pts = np.array([[1, 2], [1, 4.9], [1, 5.2], [1.6, 2]])
    assert points_in_rect(pts, (1, 3), math.pi / 2, 2, 0.5).tolist() == [True, True, False, False]


def test_perimeter_samples_cover_edges():
    p = perimeter_samples(2, 1, 0.5, 0.25)
    assert p[:, 0].max() == 2 and p[:, 0].min() == -1
    assert np.isclose(np.abs(p[:, 1]).max(), 0.5)
    assert len(p) >= 2 * (3 / 0.25)
