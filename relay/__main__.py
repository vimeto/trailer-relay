"""CLI: python -m relay {seed,run,results,learn,report,serve,site}"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
from pathlib import Path
import socketserver
import sysconfig

from .agent import AgentConfig
from .costs import Budget
from .runner import run_trials, update_index
from .seeder import RELAYS, load_or_generate_relay
from .sim import LegConfig

ROOT = Path(os.environ.get("RELAY_WORKSPACE", Path.cwd())).resolve()


def viewer_directory():
    checkout = Path(__file__).resolve().parent.parent / "viewer"
    return checkout if checkout.is_dir() else Path(sysconfig.get_path("data")) / "share" / "trailer-relay" / "viewer"


def cmd_seed(a):
    relay = load_or_generate_relay(a.relay, a.seed, a.tracks)
    for t in relay["tracks"]:
        print(f"{t.name:32s} len {t.length:6.1f} m  certified={t.meta.get('certified')} widen={t.meta.get('widen')} par={t.par}")


def cmd_run(a):
    leg_cfg = LegConfig(max_commands=a.max_commands, think_s_per_1k_tokens=a.think_cost)
    agent_cfg = AgentConfig(obs_level=a.obs, perception=a.perception, keep_full_reports=a.keep_full, mode=a.mode, control=a.control, clock=a.clock,
                            latency_scale=a.latency_scale, latency_offset_s=a.latency_offset)
    if a.clock == "realtime" and a.control != "continuous":
        raise SystemExit("--clock realtime needs --control continuous")
    budget = Budget(Path(a.runs) / "ledger" / "spend.json", a.budget_usd, a.budget_tag)
    if a.budget_usd is not None:
        print(f"budget '{a.budget_tag}': ${budget.spent():.3f} of ${a.budget_usd:.2f} spent so far", flush=True)
    kw = {}
    if a.reasoning:
        kw["reasoning_effort"] = a.reasoning
    if a.max_output_tokens:
        kw["max_output_tokens"] = a.max_output_tokens
    if a.timeout:
        kw["timeout"] = a.timeout
    paths = run_trials(a.models, a.relay, a.seed, a.trials, a.parallel, a.runs, a.tracks, leg_cfg, agent_cfg, kw,
                       log=lambda m: print(m, flush=True), verbose=a.verbose, parallel_legs=not a.sequential_legs, budget=budget,
                       live=not a.no_live)
    if a.budget_usd is not None:
        print(f"budget '{a.budget_tag}': ${budget.spent():.3f} of ${a.budget_usd:.2f} spent", flush=True)
    print("saved:", *[str(p) for p in paths], sep="\n  ")
    cmd_report(a)
    cmd_results(a)


def cmd_results(a):
    from .results import write

    print(write(a.runs))


def cmd_learn(a):
    from .learning import Study

    cfg = AgentConfig(obs_level="driver", mode="code", control="continuous", clock="paused", perception=a.perception)
    st = Study(a.model, a.condition, level=a.perception, seed=a.seed, transfer_seed=None if a.transfer_seed < 0 else a.transfer_seed,
               members=a.members, steps=a.steps, numbers=a.numbers, relay_name=a.relay, runs_dir=a.runs, tracks_dir=a.tracks,
               agent_cfg=cfg, log=lambda m: print(m, flush=True), verbose=a.verbose)
    print(f"study {st.id}", flush=True)
    st.run()
    update_index(a.runs)
    cmd_results(a)
    print(f"done: {Path(a.runs) / 'studies' / (st.id + '.json')}")


def cmd_report(a):
    idx = json.loads(update_index(a.runs).read_text())
    rows = [r for r in idx["runs"] if not getattr(a, "relay", None) or r["relay"]["name"] == a.relay]
    groups: dict[tuple, dict[str, list]] = {}
    for r in rows:
        if r.get("partial") or r.get("live"):
            continue
        cfg = r.get("config") or {}
        pv, sv, hv = cfg.get("prompt_version", 1), cfg.get("sim_version", 1), cfg.get("harness_version", 1)
        obs = (cfg.get("agent") or {}).get("obs_level", "local")
        mode, route = cfg.get("mode", "tools"), cfg.get("route", "api")
        control = cfg.get("control") or "discrete"
        if control == "continuous":
            mode += f" cont-{cfg.get('clock') or 'paused'}"
        spec = r["driver"]["spec"]
        llm = not (spec == "autopilot" or spec.startswith("scripted"))
        tags = ("" if obs == "local" else " " + obs) + ("" if mode == "tools" else " " + mode) + ("" if route == "api" else " " + route)
        key = r["driver"]["label"] + (f" (p{pv} h{hv}{tags})" if llm else "")
        groups.setdefault((r["relay"]["id"], sv), {}).setdefault(key, []).append(r)
    for (relay_id, sv), by in sorted(groups.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
        print(f"\n== relay {relay_id} · physics v{sv}")
        print(f"{'driver':52s} {'runs':>4s} {'legs✓':>7s} {'progress':>8s} {'adj time':>9s} {'touches':>7s} {'shuttles':>8s} {'jk':>4s} {'cmds':>5s} {'stops':>5s} {'out tok':>8s} {'$ billed':>8s} {'$ API eq':>8s}")
        for k, rs in sorted(by.items(), key=lambda kv: -sum(r["totals"]["relay_progress"] for r in kv[1]) / len(kv[1])):
            n = len(rs)
            avg = lambda f: sum(f(r) for r in rs) / n  # noqa: E731
            stops = f"{avg(lambda r: r['totals']['stops']):5.0f}" if all("stops" in r["totals"] for r in rs) else f"{'-':>5s}"
            print(f"{k:52s} {n:4d} {avg(lambda r: r['totals']['legs_finished']):5.2f}/{rs[0]['totals']['legs']} "
                  f"{avg(lambda r: r['totals']['relay_progress']):8.2f} {avg(lambda r: r['totals']['adjusted_time']):9.1f} "
                  f"{avg(lambda r: r['totals']['touches']):7.1f} {avg(lambda r: r['totals']['shuttles']):8.1f} {avg(lambda r: r['totals']['jackknifes']):4.1f} "
                  f"{avg(lambda r: r['totals']['commands']):5.0f} {stops} {avg(lambda r: r['totals']['usage']['output_tokens']):8.0f} "
                  f"{avg(lambda r: r['totals'].get('cost_usd', 0)):8.3f} {avg(lambda r: r['totals'].get('api_equiv_usd', 0)):8.3f}")


def cmd_serve(a):
    update_index(a.runs)
    class ViewerHandler(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            from urllib.parse import unquote, urlsplit

            urlpath = unquote(urlsplit(path).path)
            if urlpath.startswith("/viewer/"):
                base = viewer_directory().resolve()
                target = (base / urlpath[len("/viewer/"):]).resolve()
            elif urlpath.startswith("/runs/"):
                base = Path(a.runs).resolve()
                target = (base / urlpath[len("/runs/"):]).resolve()
            else:
                return super().translate_path(path)
            return str(target if target.is_relative_to(base) else base / "__not_found__")

    handler = functools.partial(ViewerHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    for port in range(a.port, a.port + 20):  # a busy port (e.g. another viewer) falls through to the next one
        try:
            httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
        except OSError:
            print(f"port {port} is in use, trying {port + 1}")
            continue
        break
    else:
        raise SystemExit(f"no free port in {a.port}-{a.port + 19}; pass --port")
    with httpd:
        print(f"viewer: http://127.0.0.1:{port}/viewer/", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def cmd_site(a):
    from .site import build

    build(a.out, ROOT, a.runs, full_traces=a.full_traces, valid_only=a.valid_only)


def main(argv=None):
    p = argparse.ArgumentParser(prog="relay")
    p.add_argument("--runs", default=str(ROOT / "runs"))
    p.add_argument("--tracks", default=str(ROOT / "tracks"))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed", help="generate + certify a relay's tracks")
    s.add_argument("--relay", default="standard", choices=sorted(RELAYS))
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_seed)
    r = sub.add_parser("run", help="run trials")
    r.add_argument("--models", nargs="+", default=["openai:gpt-6-luna"], help="provider:model specs, or 'autopilot'")
    r.add_argument("--relay", default="standard", choices=sorted(RELAYS))
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--trials", type=int, default=1)
    r.add_argument("--parallel", type=int, default=4)
    r.add_argument("--obs", default="local", choices=["sensors", "local", "sketch", "full", "cone", "cone-sketch", "driver"],
                   help="driver = driver's-seat perception: head, mirrors, reversing camera, human depth judgement (continuous only)")
    r.add_argument("--perception", default="medium", choices=["easy", "medium", "hard", "v1"],
                   help="--obs driver: difficulty (depth-noise scale, camera range, peripheral-vision range); v1 = no periphery")
    r.add_argument("--mode", default="tools", choices=["tools", "code"], help="tools: one call per command; code: the model writes driving programs")
    r.add_argument("--control", default="discrete", choices=["discrete", "continuous"],
                   help="discrete: every command ends standing still; continuous: the vehicle keeps rolling between decisions")
    r.add_argument("--clock", default="paused", choices=["paused", "realtime"],
                   help="continuous only: realtime = the sim keeps driving while the model thinks")
    r.add_argument("--latency-scale", type=float, default=1.0, help="realtime: sim seconds per wall second of model latency")
    r.add_argument("--latency-offset", type=float, default=0.0, help="realtime: wall seconds per request not charged (CLI start-up)")
    r.add_argument("--keep-full", type=int, default=4)
    r.add_argument("--budget-usd", type=float, default=None, help="hard USD limit for this tag (API routes; codex: is $0)")
    r.add_argument("--budget-tag", default="default")
    r.add_argument("--no-live", action="store_true", help="do not write live snapshots for the viewer")
    r.add_argument("--max-commands", type=int, default=80)
    r.add_argument("--think-cost", type=float, default=0.0, help="sim seconds per 1k output tokens")
    r.add_argument("--reasoning", default=None, help="reasoning effort (low/medium/high) if supported")
    r.add_argument("--max-output-tokens", type=int, default=None)
    r.add_argument("--timeout", type=float, default=None, help="per-request read timeout in seconds (default 120)")
    r.add_argument("-v", "--verbose", action="store_true", help="log every model turn")
    r.add_argument("--sequential-legs", action="store_true", help="drive legs one after another (default: concurrently)")
    r.set_defaults(fn=cmd_run)
    rs = sub.add_parser("results", help="write runs/results.json (failure modes, learning studies) for the viewer's Results view")
    rs.set_defaults(fn=cmd_results)
    ln = sub.add_parser("learn", help="a learning study: drivers leave each other constrained notes (relay/learning.py)")
    ln.add_argument("--model", required=True)
    ln.add_argument("--condition", required=True, choices=["self", "chain", "pool"])
    ln.add_argument("--steps", type=int, default=3, help="attempts (self), generations (chain) or rounds (pool)")
    ln.add_argument("--members", type=int, default=2, help="independent drivers (self) or drivers per round (pool)")
    ln.add_argument("--numbers", default="allow", choices=["allow", "none"])
    ln.add_argument("--perception", default="medium", choices=["easy", "medium", "hard", "v1"])
    ln.add_argument("--relay", default="standard", choices=sorted(RELAYS))
    ln.add_argument("--seed", type=int, default=0)
    ln.add_argument("--transfer-seed", type=int, default=1, help="-1 = no transfer attempt")
    ln.add_argument("-v", "--verbose", action="store_true")
    ln.set_defaults(fn=cmd_learn)
    rp = sub.add_parser("report", help="summary table of stored runs")
    rp.add_argument("--relay", default=None)
    rp.set_defaults(fn=cmd_report)
    sv = sub.add_parser("serve", help="serve the 3D viewer")
    sv.add_argument("--port", type=int, default=8765)
    sv.set_defaults(fn=cmd_serve)
    st = sub.add_parser("site", help="static export: story page + explorer (viewer) + the curated runs, for any static host")
    st.add_argument("--out", default=str(ROOT / "site"))
    st.add_argument("--full-traces", action="store_true", help="keep every trace sample (default: thinned to 5 Hz)")
    st.add_argument("--valid-only", action="store_true", help="omit infrastructure-failed runs and invalid learning studies")
    st.set_defaults(fn=cmd_site)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
