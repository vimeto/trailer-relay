"""Continuous code mode: a model-written controller that runs persistently and is called every control tick.

The model's program must define `control(obs)`. A separate sandboxed Python process (same isolation as
code mode, see program.py) loads it once and then answers one request per tick:

    simulator -> controller : the observation dict (JSON line)
    controller -> simulator : {"value": <whatever control returned>, "out": <printed text>, "wake": [messages]}
                              or {"error": <traceback>, "out": ...}

control(obs) returns (steer_deg, speed_mps), optionally (steer_deg, speed_mps, look_deg), or a dict with the
keys steer_deg / speed_mps / look_deg. Module-level variables persist between ticks. `wake(message)` asks
for the driver's (the model's) attention without stopping anything.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import resource
import selectors
import shutil
import subprocess
import sys
import tempfile
import time

from .program import _sandbox_profile

CTRL_BOOT = r'''
import io, json, os, sys, traceback
_rq = os.fdopen(int(sys.argv[1]), "w", buffering=1)
_rs = os.fdopen(int(sys.argv[2]), "r")
_buf = io.StringIO()
_wake = []


def wake(message="check in"):
    """Ask the driver (the model) to look at the situation. Driving continues."""
    _wake.append(str(message)[:300])


def _send(d):
    _rq.write(json.dumps(d) + "\n")
    _rq.flush()


def _out():
    s = _buf.getvalue()
    _buf.seek(0)
    _buf.truncate()
    return s[-1500:]


def _err():
    return "".join(traceback.format_exc(limit=6).splitlines(True)[-12:])


sys.stdout = _buf
_g = {"__name__": "__controller__", "wake": wake}
try:
    exec(compile(open(sys.argv[3]).read(), "controller.py", "exec"), _g)
    if not callable(_g.get("control")):
        raise NameError("the program must define a function control(obs)")
except BaseException:
    _send({"error": _err(), "out": _out()})
    sys.exit(1)
_send({"ready": True, "out": _out()})
for _line in _rs:
    _obs = json.loads(_line)
    try:
        _v = _g["control"](_obs)
        if isinstance(_v, tuple):
            _v = list(_v)
        _msg = {"value": _v, "out": _out(), "wake": _wake[:]}
        _wake.clear()
        _send(_msg)
    except BaseException:
        _send({"error": _err(), "out": _out(), "wake": _wake[:]})
        _wake.clear()
'''


class ControllerError(Exception):
    def __init__(self, message: str, out: str = ""):
        super().__init__(message)
        self.out = out


def parse_command(value) -> tuple[float, float, float | None]:
    """Validate what control() returned -> (steer_deg, speed_mps, look_deg or None)."""
    look = None
    if isinstance(value, dict):
        if "steer_deg" not in value or "speed_mps" not in value:
            raise ControllerError(f"control() returned a dict without steer_deg / speed_mps: {str(value)[:120]}")
        steer, speed, look = value["steer_deg"], value["speed_mps"], value.get("look_deg")
    elif isinstance(value, list) and len(value) in (2, 3):
        steer, speed = value[0], value[1]
        look = value[2] if len(value) == 3 else None
    else:
        raise ControllerError(f"control() must return (steer_deg, speed_mps) or a dict; it returned {str(value)[:120]}")
    try:
        steer, speed = float(steer), float(speed)
        look = None if look is None else float(look)
    except (TypeError, ValueError):
        raise ControllerError(f"control() returned non-numbers: {str(value)[:120]}") from None
    if not (math.isfinite(steer) and math.isfinite(speed)) or (look is not None and not math.isfinite(look)):
        raise ControllerError(f"control() returned a non-finite number: {str(value)[:120]}")
    return steer, speed, look


def _ctrl_limits(cpu_s: int):
    def f():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        os.setsid()
    return f


class ControllerProcess:
    """One installed controller (a sandboxed child process). Not thread-safe: one caller at a time."""

    def __init__(self, code: str, tick_timeout_s: float = 0.5, start_timeout_s: float = 5.0, cpu_s: int = 300,
                 sandbox: bool = True):
        self.code = code
        self.tick_timeout_s = tick_timeout_s
        self.sandbox = sandbox and sys.platform == "darwin" and shutil.which("sandbox-exec") is not None
        self.ticks = 0
        self._td = tempfile.TemporaryDirectory(prefix="relay-ctrl-")
        td = Path(self._td.name).resolve()
        (td / "boot.py").write_text(CTRL_BOOT)
        (td / "controller.py").write_text(code)
        rq_r, rq_w = os.pipe()  # controller -> simulator
        rs_r, rs_w = os.pipe()  # simulator -> controller
        cmd = [sys.executable, "-I", "-S", str(td / "boot.py"), str(rq_w), str(rs_r), str(td / "controller.py")]
        if self.sandbox:
            cmd = ["sandbox-exec", "-p", _sandbox_profile(td)] + cmd
        env = {"PATH": "/usr/bin:/bin", "TMPDIR": str(td), "HOME": str(td), "PYTHONIOENCODING": "utf-8"}
        self._errf = td / "stderr.txt"
        with open(self._errf, "wb") as errf:  # a file, not a pipe: a chatty controller must not block on a full pipe
            self.p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errf, cwd=td,
                                      env=env, pass_fds=(rq_w, rs_r), preexec_fn=_ctrl_limits(cpu_s))
        os.close(rq_w)
        os.close(rs_r)
        self._req = os.fdopen(rq_r, "r")
        self._resp = os.fdopen(rs_w, "w", buffering=1)
        self._sel = selectors.DefaultSelector()
        self._sel.register(self._req, selectors.EVENT_READ)
        self.alive = True
        first = self._read(start_timeout_s, "starting")
        self.startup_out = first.get("out", "")
        if "error" in first:
            self.close()
            raise ControllerError(first["error"], self.startup_out)

    def _read(self, timeout: float, what: str) -> dict:
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                self.close()
                raise ControllerError(f"the controller took longer than {timeout:g} s ({what}) and was stopped")
            if self._sel.select(timeout=left):
                line = self._req.readline()
                if not line:
                    err = self._stderr()
                    self.close()
                    raise ControllerError("the controller process died" + (f": {err}" if err else "")
                                          + (" (killed: CPU limit)" if self.p.returncode in (-9, -24, 152) else ""))
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    self.close()
                    raise ControllerError("the controller sent malformed data") from None
            elif self.p.poll() is not None:
                continue  # let the readline above see EOF

    def _stderr(self) -> str:
        try:
            self.p.wait(timeout=1)
            return self._errf.read_bytes()[-800:].decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001
            return ""

    def tick(self, obs: dict) -> tuple[tuple[float, float, float | None], str, list[str]]:
        """One control tick -> ((steer_deg, speed_mps, look_deg), printed output, wake messages)."""
        if not self.alive:
            raise ControllerError("the controller is not running")
        self.ticks += 1
        try:
            self._resp.write(json.dumps(obs, separators=(",", ":")) + "\n")
            self._resp.flush()
        except (BrokenPipeError, OSError):
            err = self._stderr()
            self.close()
            raise ControllerError("the controller process died" + (f": {err}" if err else "")) from None
        r = self._read(self.tick_timeout_s, "one control() call")
        out, wake = r.get("out", ""), list(r.get("wake") or [])
        if "error" in r:
            self.close()
            e = ControllerError(r["error"], out)
            e.wake = wake
            raise e
        try:
            cmd = parse_command(r.get("value"))
        except ControllerError as e:
            self.close()
            e.out = out
            raise
        return cmd, out, wake

    def close(self):
        if not self.alive:
            return
        self.alive = False
        for f in (self._resp, self._req):
            try:
                f.close()
            except OSError:
                pass
        try:
            self._sel.close()
        except Exception:  # noqa: BLE001
            pass
        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, 9)
            except (ProcessLookupError, PermissionError):
                self.p.kill()
        try:
            self.p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        self._td.cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
