"""The LLM driver: runs the tool-calling loop for one leg."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

from .continuous import Engine
from .costs import Budget, BudgetExceeded
from .models.base import Message, ModelAdapter, ModelError, Usage
from .observe import observe, render_report
from .program import ProgramRunner
from .prompts import briefing, system_prompt, tools_for
from .sim import Leg

# bump when the agent loop / adapters change what the model sees
# 1: original loop; 2: reasoning models get their own reasoning replayed across turns (+ summaries recorded)
# 3: bounded history (4 full reports, <= 40 abridged), sparser local map, look/run_program tools
# 4: codex route runs with all agent tools disabled (before, codex's read-only sandbox let the model run shell
#    commands that could READ the disk; Sol/Terra never did, Luna did in ~45% of turns)
HARNESS_VERSION = 4  # (continuous control is a separate loop, recorded as config.control / clock)


@dataclass
class AgentConfig:
    obs_level: str = "local"
    perception: str = "medium"  # obs_level "driver" only: difficulty level (relay/perception.py LEVELS)
    mode: str = "tools"  # "tools": one tool call per command; "code": the model writes programs that drive
    keep_full_reports: int = 4  # older reports are abridged to one line in the request
    max_abridged_reports: int = 40  # and beyond that only a stub is kept (bounded context)
    max_no_tool_turns: int = 3
    max_calls_per_turn: int = 6
    max_turns: int = 200
    program_start_s: float = 2.0  # sim seconds charged for starting a program (code mode)
    program_wall_s: float = 60.0
    max_program_calls: int = 400
    # continuous control (control="continuous"): the vehicle keeps rolling between decisions
    control: str = "discrete"  # "discrete": every command ends standing still; "continuous": held controls / a controller
    clock: str = "paused"  # continuous only. "paused": sim waits for the model; "realtime": sim drives on while the model thinks
    control_hz: float = 10.0  # controller calls per sim second
    reaction_s: float = 0.5  # paused clock: delay before a decision takes effect (old controls act meanwhile)
    latency_scale: float = 1.0  # realtime clock: sim seconds per wall second of model latency
    latency_offset_s: float = 0.0  # realtime clock: wall seconds of each request that are not charged (e.g. CLI start-up)
    plan_first_turn: bool = True  # realtime clock: the first decision is pre-race planning (the clock starts when you act)
    max_duration_s: float = 30.0  # continuous drive(): longest duration per command
    check_in_s: float = 20.0  # continuous code: default / bounds for check-ins
    min_check_in_s: float = 2.0
    max_check_in_s: float = 120.0


def nudge(mode: str, control: str = "discrete") -> str:
    if control == "continuous":
        if mode == "code":
            return "Please act now: call set_controller with a controller program, or keep_driving."
        return "Please act now: call the drive tool (speed_mps negative = reverse, 0 = stop)."
    if mode == "code":
        return "Please act now: call run_program with a Python program that drives the vehicle."
    return "Please act now: call the drive tool (negative distance = reverse)."


NUDGE = nudge("tools")


def compact_report(text: str) -> str:
    keep = [ln for ln in text.splitlines() if ln.startswith(("PROGRAM", "RESULT", "WOKE", "MOTION", "EVENTS", "STATUS", "STEERING", "PROGRESS", "VIEW"))]
    return "[older report, abridged] " + " / ".join(keep)


def compact_history(messages: list[Message], keep_full: int, max_abridged: int = 10**9) -> list[Message]:
    tool_idx = [i for i, m in enumerate(messages) if m.role == "tool"]
    old = tool_idx[:-keep_full] if keep_full >= 0 and len(tool_idx) > keep_full else []
    stub = set(old[:-max_abridged]) if len(old) > max_abridged else set()
    old = set(old)
    out = []
    for i, m in enumerate(messages):
        if i in stub:
            out.append(Message(role="tool", text="[old report omitted]", tool_call_id=m.tool_call_id, tool_name=m.tool_name))
        elif i in old:
            out.append(Message(role="tool", text=compact_report(m.text), tool_call_id=m.tool_call_id, tool_name=m.tool_name))
        else:
            out.append(m)
    return out


def program_report(res, final_report: str) -> str:
    head = f"PROGRAM FINISHED: {len(res.commands)} drive command(s), {res.calls} API call(s), {res.wall_s:.1f} s wall time."
    if res.error:
        head = head.replace("PROGRAM FINISHED", "PROGRAM FAILED")
    parts = [head, "OUTPUT:", res.stdout.rstrip() or "(nothing printed)"]
    if res.error:
        parts += ["ERROR:", res.error]
    parts += ["", final_report]
    return "\n".join(parts)


class LLMDriver:
    def __init__(self, adapter: ModelAdapter, config: AgentConfig | None = None, label: str | None = None, log=None,
                 budget: Budget | None = None):
        self.adapter = adapter
        self.cfg = config or AgentConfig()
        self.label = label or adapter.label
        self.log = log  # optional per-turn progress callback
        self.budget = budget
        self.learning: dict | None = None  # leg index -> learning.LegNotes (notes to read / write); None = no notes

    def _exec_call(self, leg: Leg, tc, k: int, call: dict) -> str:
        cfg = self.cfg
        allowed = {t.name for t in tools_for(cfg.obs_level, cfg.mode)}
        if leg.done:
            return f"LEG OVER ({leg.outcome}): this command was not executed."
        if k >= cfg.max_calls_per_turn:
            return f"Not executed: at most {cfg.max_calls_per_turn} tool calls per reply."
        if tc.name not in allowed:
            return f"Unknown tool {tc.name!r}. Available: {', '.join(sorted(allowed))}."
        a = tc.arguments or {}
        t0 = leg.t
        if tc.name == "drive":
            try:
                steer, dist = float(a.get("steer_deg")), float(a.get("distance_m"))
            except (TypeError, ValueError):
                return "Invalid arguments: drive needs numeric steer_deg and distance_m."
            r = leg.drive(steer, dist, a.get("speed_mps"))
            call.update({"command": r.command, "result": r.to_dict()})
            first = None
        elif tc.name == "look":
            r = leg.look(a.get("yaw_deg"))
            if not r["ok"]:
                return f"look refused: {r.get('note')}"
            call.update({"result": r})
            first = f"RESULT: you now look {r['gaze_deg']:+.0f}° (took {r['time_cost']:.1f} s)."
        else:  # run_program
            code = a.get("code")
            if not isinstance(code, str) or not code.strip():
                return "Invalid arguments: run_program needs the program text in `code`."
            leg.spend(cfg.program_start_s, "program")
            if leg.done:
                return f"LEG OVER ({leg.outcome}): the program was not started."
            res = ProgramRunner(leg, cfg.obs_level, cfg.program_wall_s, cfg.max_program_calls).run(code)
            call.update({"t0": round(t0, 2), "t1": round(leg.t, 2), "stdout": res.stdout, "error": res.error, "commands": res.commands,
                         "program": res.to_dict()})
            over = f"{leg.outcome} — {leg.outcome_reason}" if leg.done else None
            return program_report(res, render_report(observe(leg, cfg.obs_level), leg_over=over))
        over = f"{leg.outcome} — {leg.outcome_reason}" if leg.done else None
        call.update({"t0": round(t0, 2), "t1": round(leg.t, 2)})
        rep = render_report(observe(leg, cfg.obs_level), r if tc.name == "drive" else None, leg_over=over)
        return (first + "\n" + rep) if first else rep

    def _charge(self, resp, usage: Usage, cost: dict):
        usage.add(resp.usage)
        if self.budget:
            from .costs import usd
            cost["usd"] += self.budget.charge(self.adapter.spec, asdict(resp.usage))
            cost["api_equiv_usd"] += usd(self.adapter.spec, asdict(resp.usage))[1]

    @staticmethod
    def _finish_usage(usage: Usage, cost: dict) -> dict:
        u = asdict(usage)
        u["cost_usd"] = round(cost["usd"], 6)
        u["api_equiv_usd"] = round(cost["api_equiv_usd"], 6)
        return u

    def run_leg(self, leg: Leg, leg_index: int = 0, n_legs: int = 1, relay_note: str = "", on_update=None) -> dict:
        cfg = self.cfg
        if cfg.control == "continuous":
            return self._run_leg_continuous(leg, leg_index, n_legs, relay_note, on_update)
        system, tools = system_prompt(cfg.obs_level, cfg.mode, task=leg.track.task), tools_for(cfg.obs_level, cfg.mode)
        first = render_report(observe(leg, cfg.obs_level))
        messages: list[Message] = [Message(role="user", text=briefing(leg, leg_index, n_legs, first, relay_note, obs_level=cfg.obs_level))]
        transcript: list[dict] = []
        usage = Usage()
        cost = {"usd": 0.0, "api_equiv_usd": 0.0}
        no_tool = 0
        api_error = None
        turns = 0
        while not leg.done and turns < cfg.max_turns:
            turns += 1
            t_before = leg.t
            try:
                if self.budget:
                    self.budget.check()
                resp = self.adapter.complete(system, compact_history(messages, cfg.keep_full_reports, cfg.max_abridged_reports), tools)
            except (ModelError, BudgetExceeded) as e:
                api_error = str(e)
                leg.abort("budget exhausted" if isinstance(e, BudgetExceeded) else "model API error")
                transcript.append({"turn": turns, "t": round(t_before, 2), "error": api_error})
                break
            self._charge(resp, usage, cost)
            msg = resp.message
            messages.append(msg)
            leg.charge_thinking(resp.usage.output_tokens)
            entry = {"turn": turns, "t": round(t_before, 2), "latency_s": round(resp.latency_s, 2),
                     "usage": asdict(resp.usage), "stop_reason": resp.stop_reason, "text": msg.text, "calls": []}
            if msg.reasoning:
                entry["reasoning"] = msg.reasoning
            if not msg.tool_calls:
                no_tool += 1
                transcript.append(entry)
                if no_tool >= cfg.max_no_tool_turns:
                    leg.abort(f"driver stopped calling the tool ({no_tool} turns)")
                    break
                messages.append(Message(role="user", text=nudge(cfg.mode)))
                entry["nudged"] = True
                if on_update:
                    on_update(transcript)
                continue
            no_tool = 0
            for k, tc in enumerate(msg.tool_calls):
                call = {"id": tc.id, "name": tc.name, "args": tc.arguments}
                out = self._exec_call(leg, tc, k, call)
                call["report"] = out
                entry["calls"].append(call)
                messages.append(Message(role="tool", text=out, tool_call_id=tc.id, tool_name=tc.name))
            transcript.append(entry)
            if on_update:
                on_update(transcript)
            if self.log:
                def brief(c):
                    a = c["args"] or {}
                    if c["name"] == "drive":
                        sp = a.get("speed_mps")
                        return f"({a.get('steer_deg')}, {a.get('distance_m')}{'' if sp is None else f', v={sp}'})"
                    if c["name"] == "look":
                        return f"look({a.get('yaw_deg')})"
                    return f"program[{len(c.get('commands') or [])} cmds{', ERR' if c.get('error') else ''}]"
                cmds = ", ".join(brief(c) for c in entry["calls"])
                self.log(f"    [{self.label} leg {leg_index + 1}] turn {turns:3d} {resp.latency_s:5.1f}s  t={leg.t:6.1f}s "
                         f"{leg.status:15s} hitch {math.degrees(leg.hitch):+5.1f}  {cmds}"
                         + (f"  | {' '.join((msg.text or msg.reasoning).split())[:140]}" if (msg.text or msg.reasoning) else ""))
        if not leg.done:
            leg.abort("turn limit reached")
        return {"transcript": transcript, "usage": self._finish_usage(usage, cost), "api_error": api_error, "turns": turns,
                "messages": [m.to_dict() for m in messages][:1]}  # keep the briefing for inspection

    # ================================================================== continuous control
    def _report(self, eng: Engine, leg: Leg, head: list[str], mark: dict | None = None, label: str = "SINCE YOUR LAST DECISION") -> str:
        lines = list(head)
        if mark is not None:
            lines += eng.describe(mark, label)
        lines.append(eng.motion_line())
        over = f"{leg.outcome} — {leg.outcome_reason}" if leg.done else None
        lines.append(render_report(observe(leg, self.cfg.obs_level), leg_over=over))
        return "\n".join(lines)

    def _exec_continuous(self, eng: Engine, leg: Leg, tc, k: int, call: dict) -> tuple[str, bool]:
        """Run one continuous-mode tool call. Returns (report, woken early)."""
        cfg = self.cfg
        allowed = {t.name for t in tools_for(cfg.obs_level, cfg.mode, "continuous")}
        if leg.done:
            return f"LEG OVER ({leg.outcome}): this command was not executed.", False
        if k >= cfg.max_calls_per_turn:
            return f"Not executed: at most {cfg.max_calls_per_turn} tool calls per reply.", False
        if tc.name not in allowed:
            return f"Unknown tool {tc.name!r}. Available: {', '.join(sorted(allowed))}.", False
        a = tc.arguments or {}

        def num(key, default, lo, hi):
            try:
                v = float(a.get(key, default))
            except (TypeError, ValueError):
                v = default
            if not math.isfinite(v):
                v = default
            return max(lo, min(hi, v))

        t0 = leg.t
        call["t0"] = round(t0, 2)
        head: list[str] = []
        eng.take_wakes()  # anything before this command has already been reported
        if tc.name == "look":
            if leg.looks >= leg.cfg.max_looks:
                return f"look refused: look limit ({leg.cfg.max_looks}) reached", False
            leg.looks += 1
            leg.set_gaze(a.get("yaw_deg"))
            leg._event("look", yaw_deg=round(leg.gaze_target, 1))
            turn = abs((leg.gaze_target - leg.gaze_deg + 180.0) % 360.0 - 180.0)
            mark = eng.mark()
            eng.run(turn / leg.spec.head_rate_deg_per_s + 0.2)
            head.append(f"RESULT: you now look {leg.gaze_deg:+.0f}° (the rig kept driving for {leg.t - t0:.1f} s).")
        else:
            if not leg.count_command():
                return self._report(eng, leg, [f"LEG OVER: {leg.outcome_reason}"]), True
            if tc.name == "drive":
                notes = leg.set_controls(a.get("steer_deg"), a.get("speed_mps"))
                dur = num("duration_s", 3.0, 0.0, cfg.max_duration_s)
                leg._event("command", n=leg.commands, steer_deg=round(math.degrees(leg.target_steer), 2),
                           speed=round(leg.target_speed, 2), duration=round(dur, 2))
                mark = eng.mark()
                ran = eng.run(dur)
                head.append(f"RESULT: steering → {math.degrees(leg.target_steer):+.1f}°, speed → {leg.target_speed:+.2f} m/s; "
                            f"drove {ran:.1f} s of {dur:.1f} s." + (f" Note: {'; '.join(notes)}." if notes else ""))
                call["result"] = {"notes": notes, "duration": dur, "ran_s": round(ran, 2)}
            else:  # set_controller / keep_driving
                check = num("check_in_s", cfg.check_in_s, cfg.min_check_in_s, cfg.max_check_in_s)
                if tc.name == "set_controller":
                    code = a.get("code")
                    if not isinstance(code, str) or not code.strip():
                        return "Invalid arguments: set_controller needs the program text in `code`.", False
                    call["program"] = {"code": code}
                    err = eng.install(code)
                    if err:
                        call["error"] = err
                        mark = eng.mark()
                        head.append("CONTROLLER NOT INSTALLED (it failed to start): "
                                    + ("the previous controller keeps driving." if eng.controller else "the held controls stay in force."))
                        head.append("ERROR:\n" + err)
                        return self._report(eng, leg, head), False
                    head.append(f"RESULT: controller #{eng.controller_n} installed and driving; next check-in after {check:.0f} s.")
                else:
                    head.append(f"RESULT: {'controller #' + str(eng.controller_n) if eng.controller else 'held controls'} "
                                f"kept driving; next check-in after {check:.0f} s.")
                mark = eng.mark()
                eng.run(check)
        wakes = eng.take_wakes()
        early = bool(wakes)
        printed = "".join(o for _, o in eng.output[mark["out"]:]).rstrip()
        if printed:
            call["stdout"] = printed[-1500:]
        err = next((d for k, d in wakes if k == "controller error"), None)
        if err:
            call["error"] = err
        if wakes:
            head.insert(0, "WOKE after {:.1f} s: ".format(leg.t - t0) + "; ".join(f"{k.upper()} — {d}" for k, d in wakes))
        call["t1"] = round(leg.t, 2)
        if wakes:
            call["woke"] = [k for k, _ in wakes]
        return self._report(eng, leg, head, mark, label="DURING THIS COMMAND"), early

    def _run_leg_continuous(self, leg: Leg, leg_index: int, n_legs: int, relay_note: str, on_update) -> dict:
        cfg = self.cfg
        eng = Engine(leg, cfg.obs_level, hz=cfg.control_hz, wake_on_committed=(cfg.mode != "code"), perception=cfg.perception)
        system = system_prompt(cfg.obs_level, cfg.mode, "continuous", cfg.clock, cfg.reaction_s, task=leg.track.task)
        tools = tools_for(cfg.obs_level, cfg.mode, "continuous")
        first = render_report(observe(leg, cfg.obs_level))
        ln = (self.learning or {}).get(leg_index)
        text = briefing(leg, leg_index, n_legs, first, relay_note, control="continuous", obs_level=cfg.obs_level)
        if ln is not None:
            from .learning import notes_section

            text += notes_section(ln)
        messages: list[Message] = [Message(role="user", text=text)]
        transcript: list[dict] = []
        usage, cost = Usage(), {"usd": 0.0, "api_equiv_usd": 0.0}
        no_tool, api_error, turns = 0, None, 0
        try:
            while not leg.done and turns < cfg.max_turns:
                turns += 1
                t_before = leg.t
                mark = eng.mark()
                req = lambda: self.adapter.complete(system, compact_history(messages, cfg.keep_full_reports, cfg.max_abridged_reports), tools)  # noqa: E731
                think_sim = 0.0
                try:
                    if self.budget:
                        self.budget.check()
                    if cfg.clock == "realtime" and not (cfg.plan_first_turn and turns == 1):
                        resp, _wall, think_sim = eng.run_while(req, cfg.latency_scale, cfg.latency_offset_s)
                    else:
                        resp = req()
                except (ModelError, BudgetExceeded) as e:
                    api_error = str(e)
                    leg.abort("budget exhausted" if isinstance(e, BudgetExceeded) else "model API error")
                    transcript.append({"turn": turns, "t": round(t_before, 2), "error": api_error})
                    break
                self._charge(resp, usage, cost)
                msg = resp.message
                messages.append(msg)
                entry = {"turn": turns, "t": round(t_before, 2), "t_reply": round(leg.t, 2), "latency_s": round(resp.latency_s, 2),
                         "think_sim_s": round(think_sim, 2), "usage": asdict(resp.usage), "stop_reason": resp.stop_reason,
                         "text": msg.text, "calls": []}
                if msg.reasoning:
                    entry["reasoning"] = msg.reasoning
                if leg.done:  # the leg ended while the model was thinking (realtime)
                    entry["late"] = True
                    transcript.append(entry)
                    break
                # what happened while the model was thinking (realtime) is reported with the first call
                while_thinking = eng.describe(mark, "WHILE YOU WERE THINKING") if think_sim > 0 else []
                thinking_wakes = eng.take_wakes() if think_sim > 0 else []
                if thinking_wakes:
                    while_thinking.insert(0, "WHILE YOU WERE THINKING: " + "; ".join(f"{k.upper()} — {d}" for k, d in thinking_wakes))
                if not msg.tool_calls:
                    no_tool += 1
                    transcript.append(entry)
                    if no_tool >= cfg.max_no_tool_turns:
                        leg.abort(f"driver stopped calling the tool ({no_tool} turns)")
                        break
                    messages.append(Message(role="user", text="\n".join(while_thinking + [nudge(cfg.mode, "continuous")])))
                    entry["nudged"] = True
                    if on_update:
                        on_update(transcript)
                    continue
                no_tool = 0
                if cfg.clock == "paused" and cfg.reaction_s > 0:
                    eng.run(cfg.reaction_s, interrupt=False)  # the previous controls act during the reaction time
                woken = False
                for k, tc in enumerate(msg.tool_calls):
                    call = {"id": tc.id, "name": tc.name, "args": tc.arguments}
                    if woken and not leg.done:
                        out = "Not executed: you were woken by an event before this command (see the previous report)."
                    else:
                        out, woken = self._exec_continuous(eng, leg, tc, k, call)
                    if k == 0 and while_thinking:
                        out = "\n".join(while_thinking) + "\n" + out
                    call["report"] = out
                    entry["calls"].append(call)
                    messages.append(Message(role="tool", text=out, tool_call_id=tc.id, tool_name=tc.name))
                transcript.append(entry)
                if on_update:
                    on_update(transcript)
                if self.log:
                    def brief(c):
                        a = c["args"] or {}
                        if c["name"] == "drive":
                            return f"({a.get('steer_deg')}, v={a.get('speed_mps')}, {a.get('duration_s')}s)"
                        if c["name"] == "look":
                            return f"look({a.get('yaw_deg')})"
                        if c["name"] == "set_controller":
                            return "controller" + ("[FAILED]" if c.get("error") else "") + f"({a.get('check_in_s')}s)"
                        return f"keep({a.get('check_in_s')}s)"
                    cmds = ", ".join(brief(c) + ("!" + "/".join(c["woke"]) if c.get("woke") else "") for c in entry["calls"])
                    self.log(f"    [{self.label} leg {leg_index + 1}] turn {turns:3d} {resp.latency_s:5.1f}s  t={leg.t:6.1f}s "
                             f"{leg.status:15s} v {leg.v:+.2f} hitch {math.degrees(leg.hitch):+5.1f} prog {leg.metrics()['progress']:.2f}  {cmds}"
                             + (f"  | {' '.join((msg.text or msg.reasoning).split())[:120]}" if (msg.text or msg.reasoning) else ""))
        finally:
            eng.close()
        if not leg.done:
            leg.abort("turn limit reached")
        out = {"transcript": transcript, "api_error": api_error, "turns": turns,
               "messages": [m.to_dict() for m in messages][:1], "control": {"ticks": eng.ticks, "controllers": eng.controller_n}}
        if ln is not None:
            out["notes_in"] = ln.notes_in
            if ln.write and not api_error:
                out["note"] = self._leave_note(leg, system, messages, ln, usage, cost)
        out["usage"] = self._finish_usage(usage, cost)
        return out

    def _leave_note(self, leg: Leg, system: str, messages: list, ln, usage, cost) -> dict:
        """One extra turn after the leg: the driver leaves a note (checked; one rewrite if it breaks the rules)."""
        from .learning import LEAVE_NOTE_TOOL, check_note, leg_summary, note_request

        note = {"text": None, "accepted": False, "rejections": [], "drafts": []}
        messages.append(Message(role="user", text=note_request(ln, leg_summary(leg))))
        for _ in range(1 + ln.policy.retries):
            try:
                resp = self.adapter.complete(system, compact_history(messages, self.cfg.keep_full_reports, self.cfg.max_abridged_reports),
                                             [LEAVE_NOTE_TOOL])
            except ModelError as e:
                note["error"] = str(e)
                break
            self._charge(resp, usage, cost)
            msg = resp.message
            messages.append(msg)
            tc = next((c for c in msg.tool_calls if c.name == "leave_note"), None)
            text = (tc.arguments or {}).get("note") if tc else None
            why = check_note(text, ln.policy) if tc else ["no leave_note call"]
            note["drafts"].append(text if text is not None else (msg.text or ""))
            if not why:
                note.update(text=text.strip(), accepted=True)
                break
            note["rejections"].append(why)
            reply = "NOTE REJECTED: " + "; ".join(why) + ". Rewrite it and call leave_note again."
            if tc:
                messages.append(Message(role="tool", text=reply, tool_call_id=tc.id, tool_name="leave_note"))
            else:
                messages.append(Message(role="user", text=reply))
        return note


class ScriptDriver:
    """Drives a fixed list of commands (tests, demos)."""

    label = "script"

    def __init__(self, commands):
        self.commands = list(commands)

    def run_leg(self, leg: Leg, *a, **k) -> dict:
        for s, d in self.commands:
            if leg.done:
                break
            leg.drive(s, d)
        if not leg.done:
            leg.abort("script exhausted")
        return {"transcript": [], "usage": {}}


def wall_clock():
    return time.time()
