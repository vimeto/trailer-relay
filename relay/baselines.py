"""Naive controllers: the difficulty yardstick for an observation mode / course family.

Each one is the "obvious" controller a good model writes on its first try - hold the hitch at the angle that aims
the trailer at the middle of the rails a few metres back - and is installed through the normal set_controller tool,
so it drives through exactly the same pipeline (sandboxed process, 10 Hz, same obs) as a model. If a naive baseline
still parks at autopilot pace, the variant is not hard enough to separate models.

    scripted:ctrl-local     local map (exact rails in the trailer's frame); what the best models write today
    scripted:ctrl-shoulder  driver's seat: look back over the left shoulder, use the direct view
    scripted:ctrl-camera    driver's seat: watch the reversing-camera screen (falls back to the shoulder)
"""
from __future__ import annotations

import re

from .models.base import Message, ToolCall

_LAW = """
import math

LT = {lt}      # hitch -> trailer rear (m)
LOOK = {look}  # look-ahead behind the trailer (m)
SPEED = {speed}

def centre_law(h, left, right):
    # left/right: [(u, v)] rail points in the trailer-rear frame (u behind, v left). Returns (steer, speed).
    global WIDTH
    def near(pts):
        c = [v for u, v in pts if abs(u - LOOK) < 1.5]
        return sum(c) / len(c) if c else None
    l, r = near(left), near(right)
    if l is not None and r is not None:
        WIDTH = l - r
        y = (l + r) / 2
    elif l is not None:
        y = l - WIDTH / 2
    elif r is not None:
        y = r + WIDTH / 2
    else:
        y = 0.0
    curv = 2 * y / (LOOK * LOOK + y * y)
    desired = max(-32, min(32, math.degrees(math.atan(LT * curv))))
    steer = max(-45, min(45, 0.52 * h + 1.7 * (h - desired)))
    return steer, -SPEED if abs(h) < 25 else -0.6 * SPEED

WIDTH = 3.0
"""

LOCAL = _LAW + """
def control(obs):
    m = obs['local_map']
    return centre_law(obs['hitch_deg'], m['left_rail'], m['right_rail'])
"""

DRIVER = _LAW + """
GAZE = None
LAST = (0.0, -SPEED)

def seen_in_trailer_frame(obs, names):
    rig = obs['rig']
    rx, ry = rig['trailer_rear']
    ph = math.radians(rig['trailer_heading_deg'])
    eu = (-math.cos(ph), -math.sin(ph))
    ev = (-math.sin(ph), math.cos(ph))
    left, right = [], []
    for name in names:
        v = obs['views'].get(name)
        if not v:
            continue
        ox, oy = v['origin']
        for b, d in v['walls']:
            x = ox + d * math.cos(math.radians(b)) - rx
            y = oy + d * math.sin(math.radians(b)) - ry
            u, w = x * eu[0] + y * eu[1], x * ev[0] + y * ev[1]
            if u > -0.5:
                (left if w > 0 else right).append((u, w))
    return left, right

def control(obs):
    global LAST
    head = obs['head']
    gaze = GAZE(obs)
    if not head['settled'] or not obs['views']:
        return LAST[0], LAST[1], gaze
    names = ['camera'] if 'camera' in obs['views'] else ['direct']
    left, right = seen_in_trailer_frame(obs, names)
    LAST = centre_law(obs['hitch_deg'], left, right)
    return LAST[0], LAST[1], gaze
"""

SHOULDER = DRIVER.replace("GAZE = None", "GAZE = lambda obs: 0.95 * obs['head']['max_deg']")
CAMERA = DRIVER.replace("GAZE = None", "GAZE = lambda obs: obs['rig']['screen_gaze_deg'] if obs['rig']['screen_gaze_deg'] is not None "
                                       "else 0.95 * obs['head']['max_deg']")


def _dims(messages) -> tuple[float, float, float, float]:
    """(hitch->trailer rear, default reverse speed, camera range or 0, view range) from the briefing."""
    text = next((m.text for m in messages if m.role == "user"), "")
    lt = float(re.search(r"trailer axle ([\d.]+) m behind the hitch", text).group(1))
    rear = float(re.search(r"rear end ([\d.]+) m behind its axle", text).group(1))
    speed = float(re.search(r"reverse ([\d.]+) /", text).group(1))
    cam = re.search(r"behind the trailer, up to ([\d.]+) m", text)
    return lt + rear, speed, float(cam.group(1)) if cam else 0.0, 0.0


def _installer(template: str, camera: bool = False):
    def policy(system, messages):
        n = sum(1 for m in messages if m.role == "assistant")
        if n:
            return Message(role="assistant", text="", tool_calls=[ToolCall(id=f"bl_{n}", name="keep_driving", arguments={"check_in_s": 120})])
        lt, speed, cam, _ = _dims(messages)
        look = 5.5 if not (camera and cam) else round(min(5.5, 0.8 * cam), 2)
        code = template.format(lt=round(lt, 2), look=look, speed=round(0.8 * speed, 2))
        return Message(role="assistant", text="naive centring controller", tool_calls=[
            ToolCall(id="bl_0", name="set_controller", arguments={"code": code, "check_in_s": 120})])
    return policy


POLICIES = {"ctrl-local": _installer(LOCAL), "ctrl-shoulder": _installer(SHOULDER), "ctrl-camera": _installer(CAMERA, camera=True)}


# ---------------------------------------------------------------------------- replaying a model's own controllers
def leg_controllers(run_file: str) -> list[list[str]]:
    """Per leg, the controller programs a run installed (in order)."""
    import json
    from pathlib import Path

    rec = json.loads(Path(run_file).read_text())
    out = []
    for L in sorted(rec["legs"], key=lambda x: x["index"]):
        codes = [c["args"]["code"] for e in L.get("transcript") or [] for c in e.get("calls") or []
                 if c.get("name") == "set_controller" and (c.get("args") or {}).get("code")]
        out.append(codes)
    return out


def replay_policy(run_file: str):
    """Install, for leg k, the LAST controller the original run installed on its leg k (the one that finished the leg),
    then let it drive. Answers 'would these exact programs still work here?'."""
    per_leg = leg_controllers(run_file)

    def policy(system, messages):
        n = sum(1 for m in messages if m.role == "assistant")
        if n:
            return Message(role="assistant", text="", tool_calls=[ToolCall(id=f"rp_{n}", name="keep_driving", arguments={"check_in_s": 120})])
        text = next((m.text for m in messages if m.role == "user"), "")
        k = int(re.search(r"LEG (\d+) of", text).group(1)) - 1
        codes = per_leg[k] if k < len(per_leg) else []
        if not codes:
            return Message(role="assistant", text="no controller to replay", tool_calls=[
                ToolCall(id="rp_0", name="keep_driving", arguments={"check_in_s": 120})])
        return Message(role="assistant", text=f"replaying controller {len(codes)}/{len(codes)}", tool_calls=[
            ToolCall(id="rp_0", name="set_controller", arguments={"code": codes[-1], "check_in_s": 120})])
    return policy
