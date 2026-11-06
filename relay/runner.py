"""Run relays with any driver and store self-contained run records for the viewer."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import datetime as dt
import json
import os
from pathlib import Path
import threading
import time
import uuid

from .agent import HARNESS_VERSION, AgentConfig, LLMDriver
from .autopilot import Autopilot
from .costs import Budget
from .models.registry import make_adapter
from .prompts import CONTROL_PROMPT_VERSION, PROMPT_VERSION, TURN_PROMPT_VERSION
from .seeder import load_or_generate_relay, relay_to_dict
from .sim import SIM_VERSION, Leg, LegConfig
from .vehicles import VEHICLES

FAIL_PENALTY_S = 60.0
_index_lock = threading.Lock()

PRETTY = {"gpt-6-luna": "GPT-6 Luna", "gpt-6-sol": "GPT-6 Sol", "gpt-6.1-sol": "GPT-6.1 Sol", "gpt-6.1-terra": "GPT-6.1 Terra",
          "gpt-6-astra": "GPT-6 Astra"}
LIVE_INTERVAL_S = 2.5


def pretty_label(spec: str) -> str:
    if spec == "autopilot":
        return "Autopilot"
    model = spec.split(":", 1)[-1].split("@")[0]
    return PRETTY.get(model, model)


def route_of(spec: str) -> str:
    """How the model is reached: 'api' (billed per token), a subscription CLI ('codex', 'claude-code', 'agy')
    or 'local' (no model)."""
    if spec == "autopilot" or spec.startswith("scripted"):
        return "local"
    for r in ("codex", "claude-code", "agy"):
        if spec.startswith(r + ":"):
            return r
    return "api"


def leg_config_for(mode: str, base: LegConfig | None = None, control: str = "discrete") -> LegConfig:
    """Code mode: commands issued by a program are cheap in time (0.25 s) and may be many (400).
    Continuous control: the command budget counts the model's decisions (tool calls), not controller ticks."""
    cfg = base or LegConfig()
    if control == "continuous":
        return cfg
    if mode == "code":
        d = asdict(cfg)
        d.update(command_overhead_s=0.25, max_commands=max(cfg.max_commands, 400))
        cfg = LegConfig(**d)
    return cfg


def make_driver(spec: str, agent_cfg: AgentConfig | None = None, turn_log=None, budget: Budget | None = None, **adapter_kw):
    if spec == "autopilot":
        return Autopilot()
    return LLMDriver(make_adapter(spec, **adapter_kw), agent_cfg, label=pretty_label(spec), log=turn_log, budget=budget)


def _snapshot(leg: Leg, i: int, transcript: list) -> dict:
    """A running leg as it stands now (another thread is driving it: copy, and trim to a consistent length)."""
    tr = {k: list(v) for k, v in leg.trace.items()}
    n = min((len(v) for v in tr.values()), default=0)
    tr = {k: v[:n] for k, v in tr.items()}
    m = leg.metrics()
    m["outcome"] = None
    return {"index": i, "vehicle": leg.track.vehicle_key, "track": leg.track.name, "metrics": m, "trace": tr,
            "events": list(leg.events), "transcript": list(transcript), "usage": {}, "api_error": None, "turns": len(transcript),
            "briefing": None, "wall_s": None, "running": True, "gaze0_deg": 180.0}


class LiveWriter:
    """Writes runs/live/<run_id>.json every few seconds while a run is in progress (the viewer polls it)."""

    def __init__(self, runs_dir: str | Path, run_id: str, build, interval: float = LIVE_INTERVAL_S):
        self.dir = Path(runs_dir) / "live"
        self.path = self.dir / f"{run_id}.json"
        self.build, self.interval = build, interval
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.write()
        self._th.start()
        return self

    def write(self):
        try:
            rec = self.build()
            rec["live"] = True
            rec["updated"] = round(time.time(), 2)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, separators=(",", ":")))
            os.replace(tmp, self.path)
            update_index(self.dir.parent)
        except Exception as e:  # noqa: BLE001 - live view is best effort
            print(f"  (live snapshot failed: {type(e).__name__}: {e})", flush=True)

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.write()

    def close(self):
        self._stop.set()
        if self._th.is_alive():
            self._th.join(timeout=5)
        self.path.unlink(missing_ok=True)


def run_relay(relay: dict, driver, driver_spec: str, leg_cfg: LegConfig | None = None, agent_cfg: AgentConfig | None = None,
              label: str | None = None, trial: int = 0, log=print, checkpoint=None, parallel_legs: bool = False,
              live_dir: str | Path | None = None) -> dict:
    """Drive every leg in order. `checkpoint(record)` is called with a partial record after each leg.
    With `live_dir`, a live snapshot of the whole run (running legs included) is kept in <live_dir>/live/."""
    leg_cfg = leg_cfg or LegConfig()
    tracks = relay["tracks"]
    legs_out = []
    clock = 0.0
    t_wall = time.time()
    run_id = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    created = dt.datetime.now().isoformat(timespec="seconds")
    running: dict[int, tuple] = {}  # leg index -> (Leg, transcript list)
    done: dict[int, dict] = {}
    lock = threading.Lock()

    def live_record():
        with lock:
            items = dict(done)
            items.update({i: _snapshot(lg, i, tx) for i, (lg, tx) in running.items() if i not in done})
        legs, clock = [], 0.0
        for i in sorted(items):
            L = dict(items[i], start_clock=round(clock, 2))
            clock += L["metrics"]["time"]
            legs.append(L)
        return _record(run_id, created, relay, legs, driver, driver_spec, label, trial, leg_cfg, agent_cfg, t_wall, partial=True)

    def run_one(i, tr, note=""):
        leg = Leg(tr, config=leg_cfg)
        tx: list = []
        with lock:
            running[i] = (leg, tx)
        w0 = time.time()

        def on_update(transcript):
            tx[:] = transcript
        try:
            extra = (driver.run_leg(leg, i, len(tracks), note, on_update=on_update) if isinstance(driver, LLMDriver)
                     else driver.run_leg(leg))
        except Exception as e:  # noqa: BLE001 - a harness bug ends this leg, not the whole relay
            import traceback
            err = f"harness error: {type(e).__name__}: {e}"
            if log:
                log(f"  !! leg {i + 1}: {err}\n{traceback.format_exc()}")
            if not leg.done:
                leg.abort(err)
            extra = {"transcript": list(tx), "usage": {}, "api_error": err}
        finally:
            with lock:
                running.pop(i, None)
        m = leg.metrics()
        out = {
            "index": i, "vehicle": tr.vehicle_key, "track": tr.name, "metrics": m, "trace": leg.trace, "events": leg.events,
            "transcript": extra.get("transcript", []), "usage": extra.get("usage", {}), "api_error": extra.get("api_error"),
            "turns": extra.get("turns"), "briefing": (extra.get("messages") or [{}])[0].get("text"), "wall_s": round(time.time() - w0, 1),
            "gaze0_deg": 180.0,
        }
        for k in ("notes_in", "note"):  # learning studies (relay/learning.py)
            if k in extra:
                out[k] = extra[k]
        if log:
            log(f"  [{label or driver_spec} t{trial}] leg {i + 1}/{len(tracks)} {tr.vehicle_key:7s} {m['outcome']:9s} "
                f"time {m['time']:6.1f}s cmds {m['commands']:3d} touches {m['touches']} shuttles {m['shuttles']} "
                f"jk {m['jackknifes']} progress {m['progress'] * 100:5.1f}% wall {out['wall_s']:.0f}s")
        return out

    def prefix(done: dict) -> list:
        # legs are scored on one relay clock: leg i starts when leg i-1 ends
        out, clock = [], 0.0
        for i in range(len(tracks)):
            if i not in done:
                break
            L = dict(done[i], start_clock=round(clock, 2))
            clock += L["metrics"]["time"]
            out.append(L)
        return out

    live = LiveWriter(live_dir, run_id, live_record).start() if live_dir else None

    def finish(i, out):
        with lock:
            done[i] = out

    try:
        _drive_all(tracks, run_one, finish, done, prefix, parallel_legs, checkpoint,
                   lambda: _record(run_id, created, relay, prefix(done), driver, driver_spec, label, trial, leg_cfg, agent_cfg, t_wall, partial=True))
    finally:
        if live:
            live.close()
    legs_out = prefix(done)
    return _record(run_id, created, relay, legs_out, driver, driver_spec, label, trial, leg_cfg, agent_cfg, t_wall)


def _drive_all(tracks, run_one, finish, done, prefix, parallel_legs, checkpoint, partial_record):
    if parallel_legs and len(tracks) > 1:
        # legs are independent episodes (fresh conversation, fresh vehicle), so they can be driven concurrently
        with ThreadPoolExecutor(max_workers=len(tracks)) as ex:
            futs = {ex.submit(run_one, i, tr): i for i, tr in enumerate(tracks)}
            saved = 0
            for f in as_completed(futs):
                finish(futs[f], f.result())
                n = len(prefix(done))
                if checkpoint and saved < n < len(tracks):
                    saved = n
                    checkpoint(partial_record())
    else:
        clock = 0.0
        for i, tr in enumerate(tracks):
            finish(i, run_one(i, tr, f"relay clock so far {clock:.0f} s" if i else "first leg"))
            clock += done[i]["metrics"]["time"]
            if checkpoint and i < len(tracks) - 1:
                checkpoint(partial_record())


def _record(run_id, created, relay, legs_out, driver, driver_spec, label, trial, leg_cfg, agent_cfg, t_wall, partial=False) -> dict:
    tracks = relay["tracks"]
    finished = sum(1 for L in legs_out if L["metrics"]["outcome"] == "FINISHED")
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
    for L in legs_out:
        for k in usage:
            usage[k] += (L["usage"] or {}).get(k, 0)
    totals = {
        "legs": len(tracks), "legs_run": len(legs_out), "legs_finished": finished,
        "relay_time": round(sum(L["metrics"]["time"] for L in legs_out), 2),
        "adjusted_time": round(sum(L["metrics"]["time"] if L["metrics"]["outcome"] == "FINISHED" else L["metrics"]["budget"] + FAIL_PENALTY_S for L in legs_out), 2),
        "relay_progress": round(sum(L["metrics"]["progress"] for L in legs_out), 3),
        "touches": sum(L["metrics"]["touches"] for L in legs_out),
        "shuttles": sum(L["metrics"]["shuttles"] for L in legs_out),
        "jackknifes": sum(L["metrics"]["jackknifes"] for L in legs_out),
        "commands": sum(L["metrics"]["commands"] for L in legs_out),
        "stops": sum(L["metrics"].get("stops", 0) for L in legs_out),
        "stopped_time": round(sum(L["metrics"].get("stopped_time", 0.0) for L in legs_out), 2),
        "usage": usage, "wall_s": round(time.time() - t_wall, 1),
        "api_errors": sum(1 for L in legs_out if L["api_error"]),
        "cost_usd": round(sum((L["usage"] or {}).get("cost_usd", 0.0) for L in legs_out), 6),
        "api_equiv_usd": round(sum((L["usage"] or {}).get("api_equiv_usd", 0.0) for L in legs_out), 6),
    }
    rec = {
        "run_id": run_id,
        "created": created,
        "driver": {"spec": driver_spec, "label": label or getattr(driver, "label", driver_spec), "trial": trial},
        "config": {"leg": asdict(leg_cfg), "agent": asdict(agent_cfg) if agent_cfg else None, "prompt_version": PROMPT_VERSION,
                   "sim_version": SIM_VERSION, "harness_version": HARNESS_VERSION,
                   "mode": (agent_cfg.mode if agent_cfg else "tools"), "route": route_of(driver_spec),
                   "control": (agent_cfg.control if agent_cfg else "discrete"),
                   "clock": (agent_cfg.clock if agent_cfg and agent_cfg.control == "continuous" else None),
                   "control_prompt_version": (CONTROL_PROMPT_VERSION if agent_cfg and agent_cfg.control == "continuous" else None),
                   "task": ("turnaround" if any(t.task == "turnaround" for t in tracks) else "relay"),
                   "turn_prompt_version": (TURN_PROMPT_VERSION if any(t.task == "turnaround" for t in tracks) else None)},
        "relay": relay_to_dict(relay), "legs": legs_out, "totals": totals,
        "vehicles": {k: {**VEHICLES[k].to_dict(), "rig": _rig(k, agent_cfg.perception if agent_cfg else None)}
                     for k in {t.vehicle_key for t in tracks}},
    }
    if agent_cfg and agent_cfg.obs_level == "driver":
        from .perception import PERCEPTION_VERSION, rig_for
        from .prompts import DRIVER_PROMPT_VERSION

        rec["config"]["perception"] = {"version": PERCEPTION_VERSION, "prompt_version": DRIVER_PROMPT_VERSION,
                                       "level": agent_cfg.perception,
                                       "rigs": {k: rig_for(VEHICLES[k], agent_cfg.perception).to_dict() for k in {t.vehicle_key for t in tracks}}}
    if partial:
        rec["partial"] = True
    return rec


def _rig(vehicle: str, level: str | None = None) -> dict:
    from .perception import DEFAULT_LEVEL, rig_for

    return rig_for(VEHICLES[vehicle], level or DEFAULT_LEVEL).to_dict()


def save_run(record: dict, runs_dir: str | Path = "runs") -> Path:
    d = Path(runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{record['run_id']}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, separators=(",", ":")))
    os.replace(tmp, p)
    update_index(d)
    return p


def summarize(record: dict, file: str) -> dict:
    return {"file": file, "run_id": record["run_id"], "created": record["created"], "driver": record["driver"], "partial": record.get("partial", False),
            "relay": {"id": record["relay"]["id"], "name": record["relay"]["name"], "seed": record["relay"]["seed"]},
            "totals": record["totals"], "config": record.get("config"),
            "legs": [{"vehicle": L["vehicle"], **{k: L["metrics"][k] for k in ("outcome", "time", "touches", "progress", "shuttles", "jackknifes", "commands")}}
                     for L in record["legs"]],
            **({"learning": record["learning"]} if record.get("learning") else {}),
            **({"live": True, "updated": record.get("updated")} if record.get("live") else {})}


def update_index(runs_dir: str | Path = "runs") -> Path:
    d = Path(runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    with _index_lock:
        items = []
        finished = set()
        for p in sorted(d.glob("*.json")):
            if p.name == "index.json":
                continue
            try:
                items.append(summarize(json.loads(p.read_text()), p.name))
                finished.add(items[-1]["run_id"])
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        for p in sorted((d / "live").glob("*.json")) if (d / "live").is_dir() else []:
            try:
                rec = json.loads(p.read_text())
                if rec["run_id"] in finished:
                    continue
                it = summarize(rec, f"live/{p.name}")
                stale = time.time() - (rec.get("updated") or 0) > 600  # the writer died: show it as an unfinished run
                it.update(live=not stale, updated=rec.get("updated"), partial=stale)
                items.append(it)
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        out = d / "index.json"
        out.write_text(json.dumps({"runs": items}, indent=1))
    return out


def run_trials(models: list[str], relay_name: str = "standard", seed: int = 0, trials: int = 1, parallel: int = 4,
               runs_dir="runs", tracks_dir="tracks", leg_cfg: LegConfig | None = None, agent_cfg: AgentConfig | None = None,
               adapter_kw: dict | None = None, log=print, verbose: bool = False, parallel_legs: bool = True,
               budget: Budget | None = None, live: bool = True) -> list[Path]:
    relay = load_or_generate_relay(relay_name, seed, tracks_dir)
    jobs = [(m, t) for m in models for t in range(trials)]
    paths: list[Path] = []
    agent_cfg = agent_cfg or AgentConfig()
    leg_cfg = leg_config_for(agent_cfg.mode, leg_cfg, agent_cfg.control)
    budget = budget or Budget(Path(runs_dir) / "ledger" / "spend.json")  # always keep the ledger

    def job(m, t):
        drv = make_driver(m, agent_cfg, turn_log=log if verbose else None, budget=budget, **(adapter_kw or {}))
        auto = m == "autopilot"
        rec = run_relay(relay, drv, m, LegConfig() if auto else leg_cfg, None if auto else agent_cfg, label=pretty_label(m), trial=t,
                        log=log, checkpoint=None if live else (lambda r: save_run(r, runs_dir)),
                        parallel_legs=parallel_legs and not auto, live_dir=runs_dir if live else None)
        return save_run(rec, runs_dir)

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futs = {ex.submit(job, m, t): (m, t) for m, t in jobs}
        for f in as_completed(futs):
            m, t = futs[f]
            try:
                paths.append(f.result())
            except Exception as e:  # noqa: BLE001 - report and continue other trials
                log(f"  !! {m} trial {t} crashed: {type(e).__name__}: {e}")
    return paths
