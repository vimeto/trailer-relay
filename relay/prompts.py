"""Driver-facing text: system prompt, leg briefing, tool schema."""
from __future__ import annotations

from .models.base import ToolSpec
from .sim import Leg

PROMPT_VERSION = 3  # v2: explicit sign mapping for the hitch rule; v3: speed control, look-around (cone), code mode
# the continuous-control sections are versioned separately (discrete-control text is unchanged by them)
CONTROL_PROMPT_VERSION = 1  # 1: continuous drive(steer, speed, duration) / set_controller + keep_driving; paused vs realtime clock
# the turnaround (intiaanikäännös) task text is versioned separately too (relay-course prompts are unchanged by it)
DRIVER_PROMPT_VERSION = 2  # 1: driver's-seat perception (head, mirrors, reversing camera, human depth judgement, target line of sight)
#                            2: + the corner of your eye (peripheral vision, close by only)
TURN_PROMPT_VERSION = 1  # 1: two phases (reverse the whole rig into the side road, drive out forward to the exit zone)

DRIVE_TOOL = ToolSpec(
    name="drive",
    description=(
        "Drive the vehicle. steer_deg = front-wheel angle (positive = wheels turned LEFT, negative = RIGHT, "
        "as seen from the driver's seat facing forward). distance_m = signed distance for the vehicle to travel: "
        "NEGATIVE = reverse, POSITIVE = pull forward, 0 = turn the wheels while standing still. "
        "speed_mps = cruising speed in m/s, or null for the vehicle's comfortable default. "
        "The vehicle stops at the end of the command and you get a driver report back."
    ),
    parameters={
        "type": "object",
        "properties": {
            "steer_deg": {"type": "number", "description": "Front-wheel angle in degrees (+ left, - right)."},
            "distance_m": {"type": "number", "description": "Signed distance in metres (- reverse, + forward)."},
            "speed_mps": {"type": ["number", "null"], "description": "Speed in m/s (null = default; capped at the vehicle's maximum)."},
        },
        "required": ["steer_deg", "distance_m", "speed_mps"],
        "additionalProperties": False,
    },
)

LOOK_TOOL = ToolSpec(
    name="look",
    description=(
        "Turn your head / eyes. yaw_deg = where you look relative to the vehicle's forward direction: 0 = straight ahead, "
        "+90 = to your left, -90 = to your right, 180 = straight back. You see a 120° cone around that direction. "
        "Takes a moment (more for a bigger head turn); you get a new report with what you now see."
    ),
    parameters={
        "type": "object",
        "properties": {"yaw_deg": {"type": "number", "description": "Gaze direction in degrees (0 ahead, +left, 180 back)."}},
        "required": ["yaw_deg"],
        "additionalProperties": False,
    },
)

RUN_PROGRAM_TOOL = ToolSpec(
    name="run_program",
    description=(
        "Run a Python 3 program (standard library only, no files, no network) that drives the vehicle for you. "
        "Inside the program these functions exist: drive(steer_deg, distance_m, speed_mps=None) -> {'result': {...}, 'obs': {...}}, "
        "look(yaw_deg) -> obs, observe() -> obs, report() -> the text driver report; LegOver is raised once the leg has ended. "
        "You only hear back when the program finishes: its printed output (truncated), any error, and the final driver report."
    ),
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "The full Python program to run."}},
        "required": ["code"],
        "additionalProperties": False,
    },
)

DRIVE_CONT_TOOL = ToolSpec(
    name="drive",
    description=(
        "Set the controls and keep driving. The vehicle does NOT stop at the end: it keeps these controls until your next command. "
        "steer_deg = target front-wheel angle (positive = wheels LEFT, negative = RIGHT). speed_mps = target SIGNED speed in m/s: "
        "NEGATIVE = reverse, POSITIVE = forward, 0 = brake to a standstill. duration_s = seconds (0-30) to drive like this before your "
        "next report; you are woken earlier if something happens (contact, jackknife, critical hitch angle, leg over)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "steer_deg": {"type": "number", "description": "Target front-wheel angle in degrees (+ left, - right)."},
            "speed_mps": {"type": "number", "description": "Target signed speed in m/s (- reverse, + forward, 0 = stop)."},
            "duration_s": {"type": "number", "description": "Seconds to keep driving before the next report (0-30)."},
        },
        "required": ["steer_deg", "speed_mps", "duration_s"],
        "additionalProperties": False,
    },
)

LOOK_CONT_TOOL = ToolSpec(
    name="look",
    description=(
        "Turn your head / eyes while the vehicle keeps driving. yaw_deg = where you look relative to the vehicle's forward direction: "
        "0 = ahead, +90 = left, -90 = right, 180 = straight back. You see a 120° cone around it. The report comes once your head has turned."
    ),
    parameters=LOOK_TOOL.parameters,
)

LOOK_DRIVER_TOOL = ToolSpec(
    name="look",
    description=(
        "Turn your head while the vehicle keeps driving. yaw_deg = where you look relative to the vehicle's forward direction: "
        "0 = ahead, +90 = left, -90 = right; your briefing gives your head's limits and the directions of your mirrors (and the "
        "camera screen). The report comes once your head has turned."
    ),
    parameters=LOOK_TOOL.parameters,
)

SET_CONTROLLER_TOOL = ToolSpec(
    name="set_controller",
    description=(
        "Install a controller: a Python 3 program (standard library only, no files, no network) that defines control(obs). "
        "It is called 10 times per second and drives until you replace it. check_in_s = seconds of driving (2-120) before your next "
        "report; you are woken earlier on contact, jackknife, a controller error, wake(), or when the leg ends."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The full controller program; must define control(obs)."},
            "check_in_s": {"type": "number", "description": "Seconds of driving before your next report (2-120)."},
        },
        "required": ["code", "check_in_s"],
        "additionalProperties": False,
    },
)

KEEP_DRIVING_TOOL = ToolSpec(
    name="keep_driving",
    description="Let the current controller keep driving for check_in_s seconds (2-120), then report (earlier if something happens).",
    parameters={
        "type": "object",
        "properties": {"check_in_s": {"type": "number", "description": "Seconds of driving before your next report (2-120)."}},
        "required": ["check_in_s"],
        "additionalProperties": False,
    },
)

SYSTEM_PROMPT = """\
You are the driver in a trailer-reversing relay. In each leg you drive one vehicle towing a trailer, \
and you must REVERSE the trailer along a narrow railed course into the handoff bay at the far end, \
without the vehicle or the trailer touching the rails. The leg ends automatically as soon as the whole \
trailer is inside the handoff bay and lined up with it (within 12 degrees), so finish with the trailer straight.

CONTROLS: drive(steer_deg, distance_m, speed_mps).
- steer_deg: front-wheel angle. Positive = wheels turned LEFT, negative = RIGHT (you sit in the driver's seat facing forward, \
looking back over your shoulder / in the mirrors).
- distance_m: negative = reverse, positive = pull forward, 0 = turn the wheels while standing still.
- speed_mps: null = the vehicle's comfortable speed (in your briefing). Faster saves time but costs control: the wheels turn \
at a fixed rate per SECOND, so at higher speed they turn less per metre; bumps jolt the trailer harder; and speeding up and braking \
take time (short moves never reach full speed). The maximum speed is in your briefing.
- The wheels turn gradually while rolling, so a short move may end before the wheels reach the requested angle.
- You may call drive several times in one reply; the calls run in order without you seeing the intermediate reports.

HOW A TRAILER BEHAVES WHEN REVERSING (basic driving-school knowledge):
- The trailer's rear goes the opposite way to the steering: steer RIGHT and the trailer's rear swings to your LEFT.
- To straighten a trailer that has swung to one side, steer TOWARDS that side (follow it) while reversing.
  In this interface's numbers: while REVERSING, positive steer_deg drives the hitch angle down (towards negative) and \
negative steer_deg drives it up. So if the hitch angle is +20° (trailer rear swung LEFT), steer positive (LEFT) to bring it back \
towards 0; if it is -20°, steer negative (RIGHT). To make the trailer's rear go LEFT, first steer RIGHT (negative), then follow it.
- Reversing is unstable: a small hitch angle grows by itself if you do not correct it. Short trailers react much faster than long ones.
- Beyond a critical hitch angle (vehicle-specific, you are not told it) no amount of steering can straighten the trailer \
while reversing: it only folds further. The fix is to pull FORWARD with the wheels steered to straighten it, then try again.
- At the mechanical limit the trailer is JACKKNIFED: reversing is blocked and you must pull forward.
- When pulling forward the trailer straightens out by itself.

COSTS (the clock is what matters, like a timed show-ground reversing contest):
- every command costs 1 s of reaction time; changing between forward and reverse costs extra (gear change);
- touching a rail stops you immediately and adds a time penalty; too many touches = disqualified;
- pushing against a rail you already touch, or reversing a jackknifed trailer, is BLOCKED and just wastes time;
- there is a time budget and a command budget per leg. If time runs out, the distance you still had to go is recorded.

THE DRIVER REPORT you get after each command: result of the command, status, clock, steering, hitch angle \
(positive = trailer's rear swung to your LEFT), proximity sensors (metres to the nearest rail), \
a route arrow, and a LOCAL MAP of nearby rail points. Map coordinates (u, v) are relative to the centre of the \
trailer's rear end: u = metres straight behind the trailer (the direction you are reversing into), v = metres to your LEFT \
(negative = to your right). The trailer body occupies u in [-trailer length, 0] and v in [-half width, +half width].

Think briefly about the geometry, then act. Prefer moves of a sensible length (roughly a fifth to half of the combination length) \
and correct early.\
"""


CONE_SECTION = """

WHAT YOU CAN SEE (vision only): there are NO parking sensors and no route arrow. You see the rails only inside a 120-degree \
cone around where you are looking, up to a limited distance. Use look(yaw_deg) to look around (0 = ahead, +90 = left, \
-90 = right, 180 = straight back); looking costs a little time. The report lists only the rail points and bay corners \
you currently see, so remember what you saw before."""

CODE_SECTION = """

CODE MODE: you do not send single commands. You write a Python program with run_program(code); we run it and it drives \
for you through drive(...), look(...), observe() and report(). Each drive() inside a program costs only 0.25 s of reaction \
time (instead of 1 s), but starting a program costs 2 s, and you only hear back when it ends: its print() output, any error, \
and the final report. observe() returns a dict with keys: status, clock{t,budget,left}, counters{commands,max_commands,touches,\
max_touches,gear_changes}, steer_deg, hitch_deg, max_hitch_deg, progress{covered,total,remaining}, speed{default_rev,max_rev,\
default_fwd,max_fwd}, local_map{left_rail,right_rail: [[u,v],...], bay_corners?, dock_wall?}, plus sensors{trailer{...},vehicle{...},range} \
and route{distance,bearing_deg} unless you are in vision-only mode (then view{gaze_deg,fov_deg,range}). status is one of \
READY, REVERSING, PULLING_FORWARD, COMMITTED, JACKKNIFED, PINNED while the leg runs, then FINISHED, TIME_UP or DNF; once the \
leg is over every call raises LegOver. hitch_deg has the same sign as in the reports, and the steering rule above applies \
unchanged. Closed-loop control (small steps, re-reading the hitch angle every step) works well. Print what you need to know for your next program."""


CONT_TOOLS_CONTROLS = """\
CONTROLS (continuous driving): drive(steer_deg, speed_mps, duration_s). The vehicle does NOT stop between your commands: \
it keeps the steering and speed you set until you change them.
- steer_deg: target front-wheel angle. Positive = wheels turned LEFT, negative = RIGHT (you sit in the driver's seat facing forward, \
looking back over your shoulder / in the mirrors). The wheels turn towards it at the rate in your briefing.
- speed_mps: target SIGNED speed in m/s: NEGATIVE = reverse, POSITIVE = forward, 0 = brake to a standstill. The vehicle speeds up and \
brakes at the rate in your briefing (capped at its maximum). Switching between forward and reverse means braking to 0, a gear change, \
then speeding up again. Faster costs control: the wheels turn at a fixed rate per SECOND, so at speed they turn less per metre, and \
bumps jolt the trailer harder.
- duration_s: how many seconds (0 to 30) to drive with these controls before your next report. You are woken EARLY (the report \
starts with WOKE) if the vehicle or trailer touches a rail, the trailer jackknifes or goes past the critical angle, or the leg ends; \
the remaining commands of that reply are then not executed.
- You may give several drive commands in one reply: they run one after another (a short driving plan).
"""

CONT_CODE_CONTROLS = """\
CONTROLS: you drive by writing a CONTROLLER: a Python 3 program (standard library only, no files, no network) that defines a \
function control(obs). Once installed, control(obs) is called 10 times per second of driving and its answer is applied at once; \
the vehicle keeps rolling between calls and never stops unless your controller tells it to.
- control(obs) returns (steer_deg, speed_mps), or (steer_deg, speed_mps, look_deg), or a dict with those keys.
  steer_deg: target front-wheel angle. Positive = wheels turned LEFT, negative = RIGHT (as seen from the driver's seat facing forward). \
speed_mps: target SIGNED speed in m/s: NEGATIVE = reverse, POSITIVE = forward, 0 = brake to a standstill. look_deg (optional): where \
to look (0 ahead, +90 left, -90 right, 180 straight back).
  The wheels turn towards the target at the rate in your briefing; the vehicle speeds up and brakes at its acceleration; switching \
between forward and reverse means braking to 0, a gear change, then speeding up. Faster costs control: the wheels turn at a fixed \
rate per SECOND, so at speed they turn less per metre, and bumps jolt the trailer harder.
- obs is a dict: t (sim seconds), dt (seconds between calls), status, clock{t,budget,left}, counters{commands,max_commands,touches,\
max_touches,gear_changes}, steer_deg (actual wheel angle), hitch_deg, max_hitch_deg, progress{covered,total,remaining}, \
speed{default_rev,max_rev,default_fwd,max_fwd}, motion{v_mps (actual signed speed), target_steer_deg, target_speed_mps, hold_s (seconds \
the vehicle must still stand: gear change or penalty)}, local_map{left_rail,right_rail: [[u,v],...], bay_corners?, dock_wall?}, \
plus sensors{trailer{...},vehicle{...},range} and route{distance,bearing_deg} unless you are in vision-only mode (then view{gaze_deg,\
fov_deg,range}). Everything has the same meaning and signs as in the text reports. status is one of READY, STOPPED, REVERSING, \
PULLING_FORWARD, COMMITTED, JACKKNIFED, PINNED while the leg runs.
- Module-level variables persist between calls (use them for state: phases, filters, what you saw). print() output is collected and \
shown to you (truncated). Call wake("message") to get your attention without stopping (e.g. when your controller is unsure).
- If control() raises an exception, returns something invalid or takes longer than 0.5 s, the controller is removed, the vehicle \
brakes to a stop and you are woken with the error.
TOOLS: set_controller(code, check_in_s) installs a new controller (it replaces the running one and its variables start fresh; if the \
new code fails to start, the old controller keeps driving). keep_driving(check_in_s) lets the current controller continue. You get \
a report after check_in_s seconds (2 to 120) of driving, or EARLIER (the report starts with WOKE) on contact, jackknife, a controller \
error, wake(), or the end of the leg.
"""

CLOCK_PAUSED = """

THE CLOCK: the simulation is paused while you think. Each decision takes effect after a {reaction:g} s reaction time, during which \
the previous controls{ctl} keep acting."""

CLOCK_REALTIME = """

THE CLOCK RUNS WHILE YOU THINK (real time): while you are deciding, {what}. The time you take \
to answer is driving time, so a long think at speed near a rail can end in contact. Each report tells you what happened while you \
were thinking. {advice} (Your first decision, before you set off, is free planning time.)"""

DRIVER_SECTION = """

YOUR EYES (driver's seat): there are NO parking sensors, NO route arrow and NO map. You see what a driver sees, and every \
distance is your own judgement.
- HEAD: you see where you look. gaze_deg is relative to the vehicle's forward direction (0 = ahead, +90 = left, -90 = right). \
Your head turns at a fixed rate between the limits in your briefing; it cannot turn through the back of your head, so going from \
far left to far right sweeps through straight ahead. While your head is turning you see nothing (and for 0.1 s after it stops).
- DIRECT VIEW: a 120-degree cone around your gaze, out to your view range, from your eye. Your own trailer hides what is behind it.
- CORNER OF YOUR EYE: beyond the direct cone, out to ±100 degrees around your gaze, you also notice barriers CLOSE BY (within \
the range in your briefing), very roughly: enough to tell that a corner of your vehicle is about to touch a rail, not to steer by.
- MIRRORS: looking AT a mirror (your gaze within 7 degrees of the direction given in your briefing) adds what the mirror shows: a \
narrow view backwards along that side of the rig. The right mirror is convex: a wider view, distances less reliable.
- REVERSING CAMERA (if your briefing lists one): looking at the dashboard screen adds the camera on the trailer's rear: a wide view \
straight back from the trailer, short range. It moves with the trailer.
- Every seen point is [bearing_deg, distance_m]: the bearing in the VEHICLE frame (0 = straight ahead of the vehicle, + = left, \
+-180 = straight back) as seen from that view's origin, and the distance from that origin. Origins are in the vehicle frame \
(x = metres ahead of the vehicle's rear axle, y = metres to its left); a point is at (x0 + d*cos(b), y0 + d*sin(b)). Bearings are \
true directions (mirror images are already un-mirrored).
- Your depth judgement is human: a few cm at 2 m, about half a metre at 10 m, worse in the mirrors and the camera's edges. The error \
drifts slowly, so averaging does not remove it, and far distances are rounded coarsely. Bearings are good where you look and poorer \
in the corner of your eye. Far away you make out fewer separate points.
- You see barrier points (rails) and, when visible, the painted corners of the bay. Painted markings are hidden behind barriers. The \
bay is painted on the ground and need not be centred between the rails: some courses are wider on one side.
- TARGET: whether there is a line of sight from your eye to the centre of the bay (barriers and your own trailer block it) and, if \
so, its rough bearing and distance, whatever you are looking at.
- RIG: your own rig is known exactly: the hitch and the trailer's rear centre in the vehicle frame, and the trailer's heading \
relative to the vehicle (= minus the hitch angle)."""

DRIVER_OBS = """head{gaze_deg, target_deg, min_deg, max_deg, rate_deg_s, settled, looking_at (None, 'left_mirror', 'right_mirror' or \
'camera')}, rig{eye, hitch, trailer_rear: [x, y]; trailer_heading_deg, trailer_half_width, vehicle_front, vehicle_half_width, \
mirrors{left_mirror, right_mirror: the gaze_deg that looks at it}, screen_gaze_deg (None = no camera)}, views{direct?, periphery?, \
left_mirror?, right_mirror?, camera?: each {origin [x, y], axis_deg, fov_deg, range, walls [[bearing_deg, distance_m], ...], target_corners [...]}} \
(empty while your head turns), target{in_sight, bearing_deg?, dist_m?}"""

CONT_GOAL_TOOLS = """
Think briefly, then act. Correct early (a small hitch angle grows by itself), and keep the rig moving: standing still wastes time."""

CONT_GOAL_CODE = """
Aim for a controller that drives the whole leg without stopping and without touching anything: closed-loop on the hitch angle and \
the distances to the rails, re-reading obs every call. Then let it drive; replace it only when it needs fixing."""


def _continuous_prompt(obs_level: str, mode: str, clock: str, reaction_s: float) -> str:
    s = SYSTEM_PROMPT
    a = s.index("CONTROLS: drive(steer_deg, distance_m, speed_mps).")
    b_marker = "- You may call drive several times in one reply; the calls run in order without you seeing the intermediate reports.\n"
    b = s.index(b_marker) + len(b_marker)
    s = s[:a] + (CONT_CODE_CONTROLS if mode == "code" else CONT_TOOLS_CONTROLS) + s[b:]
    s = s.replace("- every command costs 1 s of reaction time; changing between forward and reverse costs extra (gear change);",
                  "- standing still wastes time; changing between forward and reverse costs extra (gear change);")
    s = s.replace("there is a time budget and a command budget per leg", "there is a time budget and a budget of decisions (tool calls) per leg")
    s = s.replace("THE DRIVER REPORT you get after each command: result of the command, status,",
                  "THE DRIVER REPORT you get after each decision: what happened since, your current motion, status,")
    s = s[: s.index("\n\nThink briefly about the geometry")]
    if obs_level == "driver":
        a = s.index("proximity sensors (metres to the nearest rail), a route arrow, and a LOCAL MAP")
        b = s.index("v in [-half width, +half width].") + len("v in [-half width, +half width].")
        s = s[:a] + "where your head is turned, your rig, the target's line of sight and what you SEE (see YOUR EYES below)." + s[b:]
        a = s.index("local_map{left_rail,right_rail:")
        b = s.index("fov_deg,range}).") + len("fov_deg,range}).")
        s = s[:a] + DRIVER_OBS + "." + s[b:]
        s += DRIVER_SECTION
        if mode == "code":
            s += ("\nReturn look_deg from control() to turn your head (obs.head.gaze_deg is where you look now). Module-level "
                  "variables are your memory of what you saw.")
        else:
            s += "\nUse look(yaw_deg) to turn your head while the vehicle keeps driving."
    if obs_level in ("cone", "cone-sketch"):
        s = s.replace("proximity sensors (metres to the nearest rail), a route arrow, and a LOCAL MAP of nearby rail points",
                      "what you SEE inside your view cone")
        cone = CONE_SECTION
        if mode == "code":
            cone = cone.replace("Use look(yaw_deg) to look around", "Return look_deg from control() to look around (your head turns at a "
                                "fixed rate; obs.view.gaze_deg is where you look now)").replace("; looking costs a little time", "")
        else:
            cone = cone.replace("; looking costs a little time", " while the vehicle keeps driving")
        s += cone
    if clock == "realtime":
        s += CLOCK_REALTIME.format(what="your controller keeps driving" if mode == "code" else "the vehicle keeps driving with your last steering and speed",
                                   advice=("So make your controller safe on its own." if mode == "code" else
                                           "Before thinking hard, slow down or stop (speed 0); when things are calm, keep moving."))
    else:
        s += CLOCK_PAUSED.format(reaction=reaction_s, ctl=" (or controller)" if mode == "code" else "")
    s += "\n" + (CONT_GOAL_CODE if mode == "code" else CONT_GOAL_TOOLS)
    if obs_level == "driver":
        s = s.replace("the hitch angle and the distances to the rails", "the hitch angle and what you see")
    return s


TURN_TASK = """\
You are the driver in a trailer manoeuvring contest. In each leg you drive one vehicle towing a trailer and must TURN THE RIG \
AROUND on a two-lane road by using a side road (an "intiaanikäännös"), without the vehicle or the trailer touching the kerbs:
  PHASE 1 (REVERSE): you start facing away from where you must end up, with your trailer just past a side road. Reverse the WHOLE \
rig (vehicle and trailer) into the side road. Phase 1 is done as soon as every corner of the vehicle and the trailer is inside the \
side road and the trailer is in line with it (within 30 degrees).
  PHASE 2 (FORWARD): drive out of the side road, turning back the way you came, and continue to the EXIT ZONE at the end of the road. \
The leg ends automatically as soon as the whole rig is inside the exit zone with the vehicle facing back down the road (within 20 degrees).
Your briefing says on which side the side road is. The report always says which phase you are in and what finishes it."""

TURN_SECTION = """

THE TURNAROUND, phase by phase:
- Phase 1 is a 90-degree reverse into the side road: the trailer leads, so set the trailer's angle up early and let the vehicle follow \
it in. Swinging the trailer into the side road swings your vehicle's front OUT towards the far kerb; keep an eye on it.
- Phase 2 is forward driving: the trailer follows inside the vehicle's path and cuts corners, so drive the vehicle wide around the \
junction corner. When pulling forward the trailer straightens by itself.
- If a phase goes wrong, shuttle: a short move the other way (forward in phase 1, reverse in phase 2) to get a better angle is normal.
- The map coordinates (u, v) stay attached to the trailer's rear in both phases: u = metres straight behind the trailer, v = metres \
to your LEFT. In phase 2 what lies ahead of your vehicle has u < -(combination length). The ROUTE arrow points along a guide line \
for the current phase: in phase 1 measured from the trailer's rear (0 = straight back), in phase 2 from the vehicle's front \
(0 = straight ahead); it is a hint, any path that gets there counts. The current phase's target zone (side road, then exit zone) \
is listed by its corners."""

TURN_CODE_NOTE = """ In code, obs also has phase{index (1 or 2), of, task, direction ('reverse' or 'forward'), goal}; local_map.bay_corners \
holds the current phase's target zone, and route also has direction, from and point_uv (the guide point in (u, v)). Your program \
must handle both phases (or you replace it when the phase changes)."""


def _turnaround(s: str, mode: str) -> str:
    import re

    a = s.index("You are the driver in a trailer-reversing relay.")
    b = s.index("so finish with the trailer straight.") + len("so finish with the trailer straight.")
    s = s[:a] + TURN_TASK + s[b:]
    s = re.sub(r"\brails\b", "kerbs", s)
    s = re.sub(r"\brail\b", "kerb", s)
    s = s.replace(", bay_corners?, dock_wall?}", ", bay_corners? (the current phase's target zone)}")
    s = s.replace("(the direction you are reversing into)", "(the direction you reverse into)")
    anchor = "\n\nCOSTS ("
    i = s.index(anchor)
    s = s[:i] + TURN_SECTION + (TURN_CODE_NOTE if mode == "code" else "") + s[i:]
    return s


def system_prompt(obs_level: str = "local", mode: str = "tools", control: str = "discrete", clock: str = "paused",
                  reaction_s: float = 0.5, task: str = "relay") -> str:
    s = _system_prompt(obs_level, mode, control, clock, reaction_s)
    return _turnaround(s, mode) if task == "turnaround" else s


def _system_prompt(obs_level: str, mode: str, control: str, clock: str, reaction_s: float) -> str:
    if control == "continuous":
        return _continuous_prompt(obs_level, mode, clock, reaction_s)
    s = SYSTEM_PROMPT
    if obs_level in ("cone", "cone-sketch"):
        s = s.replace("proximity sensors (metres to the nearest rail), a route arrow, and a LOCAL MAP of nearby rail points",
                      "what you SEE inside your view cone")
        s += CONE_SECTION
    if mode == "code":
        s = s.replace("CONTROLS: drive(steer_deg, distance_m, speed_mps).", "CONTROLS (inside your programs): drive(steer_deg, distance_m, speed_mps).")
        s = s.replace("- You may call drive several times in one reply; the calls run in order without you seeing the intermediate reports.\n", "")
        s = s.replace("- every command costs 1 s of reaction time;", "- every drive() costs 0.25 s of reaction time inside a program, every program start 2 s;")
        s += CODE_SECTION
    return s


def tools_for(obs_level: str = "local", mode: str = "tools", control: str = "discrete") -> list[ToolSpec]:
    if control == "continuous":
        if mode == "code":
            return [SET_CONTROLLER_TOOL, KEEP_DRIVING_TOOL]
        if obs_level == "driver":
            return [DRIVE_CONT_TOOL, LOOK_DRIVER_TOOL]
        return [DRIVE_CONT_TOOL, LOOK_CONT_TOOL] if obs_level in ("cone", "cone-sketch") else [DRIVE_CONT_TOOL]
    if mode == "code":
        return [RUN_PROGRAM_TOOL]
    return [DRIVE_TOOL, LOOK_TOOL] if obs_level in ("cone", "cone-sketch") else [DRIVE_TOOL]


def _eyes_line(leg: Leg) -> str:
    from .perception import rig_for

    r = getattr(leg, "rig", None) or rig_for(leg.spec)
    parts = [f"- your eyes: you sit at ({r.eye[0]:.2f}, {r.eye[1]:+.2f}) m in the vehicle frame; your head turns "
             f"{r.gaze_min:+.0f}° to {r.gaze_max:+.0f}° at {leg.spec.head_rate_deg_per_s:g}°/s; you see ±{r.fov_deg / 2:g}° "
             f"around your gaze up to {r.range:g} m"
             + (f", and in the corner of your eye (to ±{r.periphery_deg:g}°) barriers within {r.periphery_range:g} m" if r.periphery_range else "")]
    if r.cab_block_deg:
        parts.append(f"the cab hides everything further back than ±{r.cab_block_deg:g}° (use the mirrors)")
    if r.mirrors:
        parts.append("mirrors: " + ", ".join(f"{m.name.replace('_', ' ')} at gaze {r.mirror_gaze[m.name]:+.0f}° "
                                              f"(shows ±{m.fov_deg / 2:g}° around {m.axis_deg:+.0f}°)" for m in r.mirrors))
    else:
        parts.append("no mirrors")
    parts.append(f"reversing camera: screen at gaze {r.screen_gaze:+.0f}° (shows ±{r.camera_fov_deg / 2:g}° behind the trailer, "
                 f"up to {r.camera_range:g} m)" if r.screen is not None else "no reversing camera")
    return "; ".join(parts) + "\n"


def briefing(leg: Leg, leg_index: int, n_legs: int, report: str, relay_note: str = "", control: str = "discrete",
             obs_level: str = "local") -> str:
    sp = leg.spec
    per_cmd = f"; max {sp.max_cmd_distance:g} m per command" if control == "discrete" else ""
    return (
        f"LEG {leg_index + 1} of {n_legs}{(' — ' + relay_note) if relay_note else ''}\n"
        f"Vehicle: {sp.name} towing a {sp.trailer_name}.\n"
        f"- vehicle: wheelbase {sp.wheelbase:g} m, {sp.body_front + sp.body_rear:g} m long x {sp.body_width:g} m wide; "
        f"max steering ±{sp.max_steer_deg:g}°; the wheels turn at {sp.steer_rate_deg_per_s:g}° per second\n"
        f"- hitch: {abs(sp.hitch_offset):g} m {'behind' if sp.hitch_offset >= 0 else 'ahead of'} the rear axle; "
        f"trailer axle {sp.trailer_length:g} m behind the hitch\n"
        f"- trailer body: {sp.trailer_front + sp.trailer_rear:g} m long x {sp.trailer_width:g} m wide "
        f"(rear end {sp.trailer_rear:g} m behind its axle); mechanical hitch limit ±{sp.max_hitch_deg:g}°\n"
        f"- whole combination: {sp.combo_length:.1f} m; speeds (default / max): reverse {sp.speed_rev:g} / {sp.speed_rev_max:g} m/s, "
        f"forward {sp.speed_fwd:g} / {sp.speed_fwd_max:g} m/s; acceleration and braking {sp.accel:g} m/s²; "
        f"gear change {sp.shuttle_s:g} s{per_cmd}\n"
        f"{_eyes_line(leg) if obs_level == 'driver' else ''}"
        f"{_course_line(leg)}; time budget {leg.budget:.0f} s; "
        f"{leg.cfg.max_commands} {'commands' if control == 'discrete' else 'decisions'} max; touch penalty {leg.cfg.touch_penalty_s:g} s"
        f"{'' if control == 'discrete' else f' (you stand still for it); jackknife penalty {leg.cfg.jackknife_penalty_s:g} s'}\n\n"
        f"{_start_line(leg)} Here is your first report:\n\n{report}"
    )


def _course_line(leg: Leg) -> str:
    tr = leg.track
    if tr.task != "turnaround":
        extra = ("; the bay is painted on the ground and is NOT centred between the rails (the course is wider on one side in places)"
                 if tr.meta.get("asym") else "")
        return f"- course: about {tr.goal_s - leg.start_s:.0f} m of reversing ({tr.template} layout){extra}"
    p = tr.params
    return (f"- course: intiaanikäännös; road {2 * p['wm']:.1f} m wide, side road {2 * p['wb']:.1f} m wide and {p['Db']:.0f} m deep "
            f"(its corners are rounded, radius {p['rc']:.1f} m); exit zone {p['Le']:.0f} m long; about {tr.goal_s - leg.start_s:.0f} m "
            "of driving in total")


def _start_line(leg: Leg) -> str:
    if leg.track.task != "turnaround":
        return "You are at the start, stopped, wheels straight, facing away from the course."
    side = "LEFT" if leg.track.side > 0 else "RIGHT"
    return (f"You are stopped on the road, wheels straight, facing away from the exit zone (it is behind you, at the end of the road "
            f"you came along). The side road opens on your {side}, just behind your trailer's rear: reverse into it.")
