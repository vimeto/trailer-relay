"""Vehicle + trailer specifications.

Conventions (used everywhere in the package):
  * Units are metres, seconds and radians (degrees only at the driver-facing API).
  * The vehicle state is (x, y, theta1, theta2): x, y is the centre of the
    vehicle's rear axle, theta1 is the vehicle heading and theta2 the trailer
    heading. A heading points in the direction the body faces when driving forward.
  * The hitch sits `hitch_offset` (M) behind the rear axle (negative = ahead of it,
    as for a semi-trailer fifth wheel). The trailer axle sits `trailer_length` (L2)
    behind the hitch.
  * Steering is positive to the LEFT (counter-clockwise yaw when driving forward).
  * Hitch angle beta = theta1 - theta2. beta > 0 means the trailer's rear has
    swung to the driver's LEFT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class VehicleSpec:
    key: str
    name: str
    trailer_name: str
    wheelbase: float  # L
    hitch_offset: float  # M (behind rear axle; negative = ahead)
    trailer_length: float  # L2 (hitch -> trailer axle)
    max_steer_deg: float
    max_hitch_deg: float  # mechanical (jackknife) limit
    body_front: float  # rear axle -> front bumper
    body_rear: float  # rear axle -> rear bumper
    body_width: float
    trailer_front: float  # trailer axle -> front of trailer body
    trailer_rear: float  # trailer axle -> rear of trailer body
    trailer_width: float
    speed_rev: float  # default (comfortable) speed when reversing, m/s
    speed_fwd: float  # default speed when pulling forward, m/s
    steer_rate_deg_per_s: float  # how fast the front wheels can be turned (rolling or stationary)
    shuttle_s: float  # time to change between forward and reverse
    sensor_range: float  # proximity sensor range
    view_radius: float  # radius of the local rail map given to the driver
    max_cmd_distance: float  # max |distance| per drive command
    sim_ds: float  # integration step (m)
    # ---- driver-selectable speed and its limits (sim v3) ---------------------
    speed_rev_max: float = 2.0  # m/s
    speed_fwd_max: float = 3.0  # m/s
    accel: float = 0.8  # m/s^2, used for both speeding up and braking
    # ---- vision (the "cone" observation) ----------------------------------
    view_fov_deg: float = 120.0  # field of view of the driver's gaze
    view_range: float = 20.0  # how far the driver can make out the rails
    head_rate_deg_per_s: float = 240.0  # how fast the driver turns to look around

    # ---- derived quantities -------------------------------------------------
    @property
    def max_steer(self) -> float:
        return math.radians(self.max_steer_deg)

    @property
    def steer_rate_deg_per_m(self) -> float:
        """Steering slew per metre at the default reversing speed (slower driving = more steering per metre)."""
        return self.steer_rate_deg_per_s / self.speed_rev

    def max_speed(self, direction: int) -> float:
        return self.speed_fwd_max if direction > 0 else self.speed_rev_max

    def default_speed(self, direction: int) -> float:
        return self.speed_fwd if direction > 0 else self.speed_rev

    @property
    def max_hitch(self) -> float:
        return math.radians(self.max_hitch_deg)

    @property
    def combo_length(self) -> float:
        """Front bumper of the vehicle to the rear of the trailer, when straight."""
        return self.body_front + self.hitch_offset + self.trailer_length + self.trailer_rear

    @property
    def max_width(self) -> float:
        return max(self.body_width, self.trailer_width)

    @property
    def critical_hitch(self) -> float:
        from .kinematics import critical_hitch_angle

        return critical_hitch_angle(self)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["critical_hitch_deg"] = math.degrees(self.critical_hitch)
        d["combo_length"] = self.combo_length
        return d


VEHICLES: dict[str, VehicleSpec] = {
    # Ride-on mower with a short garden cart: short trailer => twitchy, small
    # critical angle. The hardest one to keep straight.
    "mower": VehicleSpec(
        key="mower", name="Ride-on mower", trailer_name="garden cart",
        wheelbase=1.25, hitch_offset=0.30, trailer_length=0.85,
        max_steer_deg=45, max_hitch_deg=85,
        body_front=1.55, body_rear=0.25, body_width=1.05,
        trailer_front=0.55, trailer_rear=0.45, trailer_width=0.95,
        speed_rev=0.8, speed_fwd=1.2, steer_rate_deg_per_s=48,
        shuttle_s=1.5, sensor_range=4.0, view_radius=7.0, max_cmd_distance=6.0, sim_ds=0.02,
        speed_rev_max=1.6, speed_fwd_max=2.5, accel=0.8, view_range=10.0,
    ),
    # Family car with a single-axle box trailer on a tow ball.
    "car": VehicleSpec(
        key="car", name="Car", trailer_name="box trailer",
        wheelbase=2.75, hitch_offset=1.05, trailer_length=2.6,
        max_steer_deg=34, max_hitch_deg=75,
        body_front=3.6, body_rear=1.0, body_width=1.85,
        trailer_front=2.0, trailer_rear=0.9, trailer_width=1.7,
        speed_rev=1.2, speed_fwd=2.0, steer_rate_deg_per_s=36,
        shuttle_s=2.5, sensor_range=6.0, view_radius=14.0, max_cmd_distance=10.0, sim_ds=0.04,
        speed_rev_max=3.0, speed_fwd_max=5.0, accel=1.2, view_range=22.0,
    ),
    # Farm tractor with a single-axle tipping trailer on a drawbar.
    "tractor": VehicleSpec(
        key="tractor", name="Tractor", trailer_name="tipping trailer",
        wheelbase=2.5, hitch_offset=1.3, trailer_length=3.6,
        max_steer_deg=45, max_hitch_deg=80,
        body_front=3.0, body_rear=0.9, body_width=2.3,
        trailer_front=3.2, trailer_rear=1.4, trailer_width=2.4,
        speed_rev=1.0, speed_fwd=1.8, steer_rate_deg_per_s=35,
        shuttle_s=3.0, sensor_range=7.0, view_radius=16.0, max_cmd_distance=10.0, sim_ds=0.05,
        speed_rev_max=2.2, speed_fwd_max=4.0, accel=0.7, view_range=25.0,
    ),
    # Semi-truck: fifth wheel slightly ahead of the drive axle, long semi-trailer.
    # Slow to respond, but needs a lot of room and folds if you hold full lock.
    "truck": VehicleSpec(
        key="truck", name="Semi truck", trailer_name="semi-trailer",
        wheelbase=3.9, hitch_offset=-0.35, trailer_length=10.2,
        max_steer_deg=42, max_hitch_deg=85,
        body_front=5.2, body_rear=1.0, body_width=2.55,
        trailer_front=11.6, trailer_rear=1.9, trailer_width=2.55,
        speed_rev=1.3, speed_fwd=2.2, steer_rate_deg_per_s=20,
        shuttle_s=4.0, sensor_range=10.0, view_radius=30.0, max_cmd_distance=20.0, sim_ds=0.08,
        speed_rev_max=2.5, speed_fwd_max=4.0, accel=0.5, view_range=40.0,
    ),
}


def get_vehicle(key: str) -> VehicleSpec:
    try:
        return VEHICLES[key]
    except KeyError as e:  # pragma: no cover - trivial
        raise KeyError(f"unknown vehicle {key!r}; choose from {sorted(VEHICLES)}") from e
