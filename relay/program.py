"""Code mode: the model writes a program that drives, we run it, and it only hears back when the program ends.

The program runs in a separate Python process (`python -I -S`: standard library only) and controls the
leg through a tiny RPC API over pipes, so the simulator stays authoritative and cannot be tampered with:

    drive(steer_deg, distance_m, speed_mps=None) -> {"result": {...}, "obs": {...}}
    look(yaw_deg)                                -> obs dict   (gaze; matters for the vision-cone observation)
    observe()                                    -> obs dict   (same information as the text reports, as data)
    report()                                     -> the text driver report
    LegOver                                      -> raised by any call once the leg has ended

When the program exits, the model gets its stdout (truncated), any error traceback, and the final report.

Isolation: on macOS the child runs under `sandbox-exec` with no network, no file writes outside its temp
directory and no reads of the home directory (so it cannot read API keys); plus CPU / file-size rlimits
and a wall-clock timeout. Elsewhere only the rlimits and timeout apply.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import selectors
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from .observe import observe, render_report
from .sim import Leg

BOOT = r'''
import json, os, sys, traceback
_rq = os.fdopen(int(sys.argv[1]), "w", buffering=1)
_rs = os.fdopen(int(sys.argv[2]), "r")


class LegOver(Exception):
    """The leg has ended (parked, out of time, disqualified) or the program hit its call limit."""


def _call(op, **kw):
    _rq.write(json.dumps({"op": op, **kw}) + "\n")
    _rq.flush()
    line = _rs.readline()
    if not line:
        raise SystemExit("the simulator closed the connection")
    r = json.loads(line)
    if r.get("leg_over"):
        raise LegOver(r.get("message", "leg over"))
    if "error" in r:
        raise ValueError(r["error"])
    return r["value"]


def drive(steer_deg, distance_m, speed_mps=None):
    return _call("drive", steer_deg=steer_deg, distance_m=distance_m, speed_mps=speed_mps)


def look(yaw_deg):
    return _call("look", yaw_deg=yaw_deg)


def observe():
    return _call("observe")


def report():
    return _call("report")


_src = open(sys.argv[3]).read()
_g = {"__name__": "__main__", "drive": drive, "look": look, "observe": observe, "report": report, "LegOver": LegOver}
try:
    exec(compile(_src, "program.py", "exec"), _g)
except LegOver as e:
    print(f"[program stopped: {e}]")
except SystemExit:
    raise
except BaseException:
    traceback.print_exc()
    sys.exit(1)
'''


def _sandbox_profile(workdir: Path) -> str:
    home = str(Path.home().resolve())
    allow_read = sorted({str(Path(p).resolve()) for p in (sys.base_prefix, sys.prefix, Path(sys.executable).resolve().parent.parent)})
    reads = " ".join(f'(subpath "{p}")' for p in allow_read + [str(workdir.resolve())])
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{workdir.resolve()}") (literal "/dev/null") (literal "/dev/dtracehelper"))
(deny file-read* (subpath "{home}"))
(allow file-read* {reads})
"""


def _limits():  # runs in the child before exec
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    os.setsid()


@dataclass
class ProgramResult:
    stdout: str
    error: str | None
    commands: list[int]
    calls: int
    wall_s: float
    timed_out: bool = False

    def to_dict(self):
        return {"stdout": self.stdout, "error": self.error, "commands": self.commands, "calls": self.calls,
                "wall_s": round(self.wall_s, 2), "timed_out": self.timed_out}


class ProgramRunner:
    def __init__(self, leg: Leg, obs_level: str = "local", wall_timeout_s: float = 60.0, max_calls: int = 400,
                 stdout_limit: int = 4000, sandbox: bool = True):
        self.leg = leg
        self.obs_level = obs_level
        self.wall_timeout_s = wall_timeout_s
        self.max_calls = max_calls
        self.stdout_limit = stdout_limit
        self.sandbox = sandbox and sys.platform == "darwin" and shutil.which("sandbox-exec") is not None

    # ---------------------------------------------------------------- one request from the program
    def _handle(self, req: dict, state: dict) -> dict:
        leg = self.leg
        state["calls"] += 1
        if state["calls"] > self.max_calls:
            return {"leg_over": True, "message": f"program call limit ({self.max_calls}) reached"}
        op = req.get("op")
        if op == "observe":
            return {"value": observe(leg, self.obs_level)}
        if op == "report":
            return {"value": render_report(observe(leg, self.obs_level))}
        if leg.done:
            return {"leg_over": True, "message": f"{leg.outcome}: {leg.outcome_reason}"}
        if op == "drive":
            try:
                steer, dist = float(req.get("steer_deg")), float(req.get("distance_m"))
            except (TypeError, ValueError):
                return {"error": "drive needs numeric steer_deg and distance_m"}
            r = leg.drive(steer, dist, req.get("speed_mps"))
            state["commands"].append(r.command)
            return {"value": {"result": r.to_dict(), "obs": observe(leg, self.obs_level)}}
        if op == "look":
            r = leg.look(req.get("yaw_deg"))
            if not r["ok"]:
                return {"error": r.get("note", "look refused")}
            return {"value": observe(leg, self.obs_level)}
        return {"error": f"unknown operation {op!r}"}

    # ---------------------------------------------------------------- run one program
    def run(self, code: str) -> ProgramResult:
        t0 = time.time()
        state = {"calls": 0, "commands": []}
        with tempfile.TemporaryDirectory(prefix="relay-prog-") as td:
            td = Path(td).resolve()
            (td / "boot.py").write_text(BOOT)
            (td / "program.py").write_text(code)
            rq_r, rq_w = os.pipe()  # program -> simulator
            rs_r, rs_w = os.pipe()  # simulator -> program
            cmd = [sys.executable, "-I", "-S", str(td / "boot.py"), str(rq_w), str(rs_r), str(td / "program.py")]
            if self.sandbox:
                cmd = ["sandbox-exec", "-p", _sandbox_profile(td)] + cmd
            env = {"PATH": "/usr/bin:/bin", "TMPDIR": str(td), "HOME": str(td), "PYTHONIOENCODING": "utf-8"}
            p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=td, env=env,
                                 pass_fds=(rq_w, rs_r), preexec_fn=_limits)
            os.close(rq_w)
            os.close(rs_r)
            out_chunks: dict[str, list[bytes]] = {"out": [], "err": []}

            def pump(stream, key):
                # keep the first `cap` bytes and a rolling window of the last `cap` bytes
                cap = 2 * self.stdout_limit
                head, tail, dropped = bytearray(), bytearray(), 0
                for chunk in iter(lambda: stream.read(4096), b""):
                    room = cap - len(head)
                    head += chunk[:room]
                    tail += chunk[room:] if room > 0 else chunk
                    if len(tail) > cap:
                        dropped += len(tail) - cap
                        del tail[: len(tail) - cap]
                out_chunks[key] = [bytes(head), b"\n...[output truncated]...\n" if dropped else b"", bytes(tail)]

            pumps = [threading.Thread(target=pump, args=(p.stdout, "out"), daemon=True),
                     threading.Thread(target=pump, args=(p.stderr, "err"), daemon=True)]
            for th in pumps:
                th.start()
            req_f = os.fdopen(rq_r, "r")
            resp_f = os.fdopen(rs_w, "w", buffering=1)
            sel = selectors.DefaultSelector()
            sel.register(req_f, selectors.EVENT_READ)
            timed_out = False
            try:
                while True:
                    left = self.wall_timeout_s - (time.time() - t0)
                    if left <= 0:
                        timed_out = True
                        break
                    if not sel.select(timeout=min(left, 1.0)):
                        if p.poll() is not None:
                            break
                        continue
                    line = req_f.readline()
                    if not line:
                        break
                    try:
                        req = json.loads(line)
                    except json.JSONDecodeError:
                        resp = {"error": "malformed request"}
                    else:
                        resp = self._handle(req, state)
                    try:
                        resp_f.write(json.dumps(resp) + "\n")
                        resp_f.flush()
                    except BrokenPipeError:
                        break
            finally:
                if timed_out or p.poll() is None:
                    try:
                        p.wait(timeout=0.5 if not timed_out else 0)
                    except subprocess.TimeoutExpired:
                        pass
                    if p.poll() is None:
                        try:
                            os.killpg(p.pid, 9)
                        except (ProcessLookupError, PermissionError):
                            p.kill()
                p.wait()
                for th in pumps:
                    th.join(timeout=2)
                sel.close()
                for f in (req_f, resp_f):
                    try:
                        f.close()
                    except (BrokenPipeError, OSError):
                        pass
        stdout = b"".join(out_chunks["out"]).decode("utf-8", "replace")
        stderr = b"".join(out_chunks["err"]).decode("utf-8", "replace")
        if len(stdout) > self.stdout_limit:
            stdout = stdout[: self.stdout_limit // 2] + "\n...[output truncated]...\n" + stdout[-self.stdout_limit // 2:]
        error = None
        if timed_out:
            error = f"the program was stopped after {self.wall_timeout_s:.0f} s of wall-clock time"
        elif p.returncode != 0:
            error = (stderr.strip()[-2000:] or f"exit code {p.returncode}")
            if p.returncode in (-9, -24, 152):
                error += " (killed: CPU limit)"
        return ProgramResult(stdout=stdout, error=error, commands=state["commands"], calls=state["calls"],
                             wall_s=time.time() - t0, timed_out=timed_out)
