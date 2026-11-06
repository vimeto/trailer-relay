"""Learning across attempts: drivers leave each other short, constrained NOTES.

Every leg is a fresh conversation, so whatever a driver understood about its failure is lost - unless it is written
down. Here, after a leg ends, the driver gets one extra turn to call `leave_note`. The note is checked (prose only:
no code, no formulas, a length cap and a cap on numbers - so it carries lessons and priorities, not a controller) and
handed to later drivers of the SAME leg (same vehicle) in their briefing. Three ways of passing notes on:

  self   one driver, the same course again and again; it sees all of ITS OWN earlier notes ("you, attempt k")
  chain  a line of fresh drivers sharing one NOTEBOOK: each reads it, drives, and rewrites it (the rewrite replaces it)
  pool   rounds of drivers driving at the same time; each round reads every note the previous round left

Each study ends with a TRANSFER attempt on a different course (another seed; same vehicles): notes that only memorised
the course ("turn right at 40 %") stop helping there, lessons about driving do not.

    python -m relay learn --model codex:gpt-6.1-sol --condition self --members 2 --attempts 3 --numbers allow

Records: every run carries `learning` {study, condition, role train|transfer, step, member, numbers, ...}; every leg
carries `notes_in` (what its briefing showed) and `note` (what it wrote: text, accepted, rejections). The whole study is
summarised in runs/studies/<study>.json (run ids in order + all notes) for the viewer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime as dt
import json
from pathlib import Path
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from .models.base import ToolSpec

LEARN_PROMPT_VERSION = 2  # 2: the note check accepts ordinary prose (semicolons, 'while', 'return to')
CONDITIONS = ("self", "chain", "pool")


@dataclass
class NotePolicy:
    numbers: str = "allow"  # allow: at most max_numbers numbers; none: no numbers at all
    max_chars: int = 600
    max_numbers: int = 6
    retries: int = 1  # a rejected note may be rewritten this many times; then nothing is passed on

    def describe(self) -> str:
        nums = (f"at most {self.max_numbers} numbers" if self.numbers == "allow"
                else "NO numbers at all (no digits, no number words): describe it in words - 'early', 'close', 'slowly'")
        return (f"plain prose, at most {self.max_chars} characters, {nums}; no code, no pseudo-code, no formulas or equations, "
                f"no variable names - the next driver writes their own controller")


@dataclass
class LegNotes:
    """What one driver of one leg reads and is asked to write."""
    condition: str  # self | chain | pool
    notes_in: list = field(default_factory=list)  # [{"from": "...", "text": "..."}]
    write: bool = True
    transfer: bool = False  # the notes were written on a different course
    policy: NotePolicy = field(default_factory=NotePolicy)


LEAVE_NOTE_TOOL = ToolSpec(
    name="leave_note",
    description="Leave the note (see the instructions). It is checked; if it breaks a rule you are told why and may rewrite it once.",
    parameters={"type": "object", "properties": {"note": {"type": "string", "description": "the note, plain prose"}},
                "required": ["note"]},
)

# ---------------------------------------------------------------------------- the check
_CODE = [  # code-SHAPED only: ordinary prose (semicolons, "while", "return to", arrows) must pass
    (r"```|`[^`]+`", "code formatting"),
    (r"\bdef\s+\w+\s*\(|\bimport\s+\w+|\blambda\b|\breturn\s*[-(\[\d]|\breturn\s+\w+\s*[,)]", "a code keyword"),
    (r"(?m)^\s*(if|elif|else|for|while|try|except)\b[^\n]*:\s*$", "a code block"),
    (r"\bobs\s*\[|\[['\"]\w+['\"]\]|\w+\.\w+\(", "code-style access or call"),
    (r"[{}]|==|!=|<=|>=|\+=|-=|\*\*", "code operators or braces"),
    (r"(?<![\w.])[A-Za-z_]\w*\s*=\s*[-+(\w]", "an assignment"),
    (r"(\w|\))\s*[*×÷^]\s*(\w|\()|\)\s*/|/\s*\(|[a-z]\s*/\s*\d|\d\s*/\s*[a-z]\b(?!/)(?<!m/s)", "a formula"),  # not units (m/s)
    (r"\b(min|max|abs|clamp|atan2?|sin|cos|tan|sqrt|exp|log|deg|rad)\s*\(", "a math function"),
    (r"\b[a-z]\w*_[a-z_]\w*\b", "a variable name"),
]
_NUM = r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?"
_NUM_WORDS = r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|" \
             r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|dozen|half|quarter|third)\b"


def check_note(text: str, policy: NotePolicy) -> list[str]:
    """Reasons the note breaks the rules ([] = accepted)."""
    if not isinstance(text, str) or not text.strip():
        return ["the note is empty"]
    why = []
    if len(text) > policy.max_chars:
        why.append(f"too long ({len(text)} characters; the limit is {policy.max_chars})")
    for pat, what in _CODE:
        m = re.search(pat, text)
        if m:
            why.append(f"it contains {what} ({m.group(0).strip()!r})")
    nums = re.findall(_NUM, text)
    if policy.numbers == "none":
        words = re.findall(_NUM_WORDS, text, flags=re.I)
        if nums or words:
            why.append(f"it contains numbers ({', '.join((nums + words)[:4])}); this note must have none")
    elif len(nums) > policy.max_numbers:
        why.append(f"it contains {len(nums)} numbers; at most {policy.max_numbers} are allowed")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 4 and sum(1 for ln in lines if ln.startswith(("    ", "\t"))) >= 2:
        why.append("indented lines like code")
    return why


# ---------------------------------------------------------------------------- prompt text
def notes_section(ln: LegNotes) -> str:
    """The briefing's NOTES section (empty if there is nothing to show)."""
    if not ln.notes_in:
        return ""
    where = ("They were written on a DIFFERENT course with the same vehicle: lessons about driving carry over, details of "
             "that course do not." if ln.transfer else "They were written on THIS course with this same vehicle.")
    head = {"self": "YOUR OWN NOTES from your earlier attempts (you remember nothing else of them)",
            "chain": "THE NOTEBOOK handed down by the earlier drivers of this leg",
            "pool": "NOTES from the drivers of the previous round (they drove at the same time, independently)"}[ln.condition]
    body = "\n".join(f"--- {n['from']}:\n{n['text']}" for n in ln.notes_in)
    return f"\n\n{head}. {where} Use your own judgement: notes can be wrong.\n{body}\n--- end of notes\n"


def note_request(ln: LegNotes, summary: str, previous: list | None = None) -> str:
    who = {"self": "You will drive this SAME course again with the same vehicle, from the start, with a fresh mind: you will "
                   "remember nothing of this attempt except the note you leave yourself now (your earlier notes stay too).",
           "chain": "A new driver (fresh, like you were) will drive this same course with the same vehicle next. All they get "
                    "is the NOTEBOOK. Rewrite it now: your text REPLACES the old notebook, so keep what still holds, correct "
                    "what proved wrong, and add what you learned.",
           "pool": "Several drivers drove this course at the same time as you, independently. Each leaves a note; the "
                   "drivers of the next round will read all of them."}[ln.condition]
    return (f"LEG OVER. {summary}\n\n{who}\n"
            "Call leave_note with what will help most: what went wrong (or right) and why, what to prioritise, and one tip "
            f"for this vehicle. Rules: {ln.policy.describe()}. Lessons and priorities, not a program.")


def leg_summary(leg) -> str:
    m = leg.metrics()
    contacts = [e for e in leg.events if e["type"] == "contact"]
    parts = [f"Outcome: {m['outcome']} ({m.get('reason') or ''}).".replace(" ().", "."),
             f"Time {m['time']:.0f} s (autopilot par {m.get('par_time') or 0:.0f} s), progress {m['progress'] * 100:.0f}%,",
             f"rail touches {m['touches']}, gear changes {m['shuttles']}, jackknifes {m['jackknifes']}, max hitch {m['max_abs_hitch_deg']:.0f}°."]
    if contacts:
        where = {}
        for e in contacts:
            k = f"{e.get('body')} {e.get('corner') or e.get('edge')}"
            where[k] = where.get(k, 0) + 1
        parts.append("Touches by part: " + ", ".join(f"{k} x{v}" for k, v in sorted(where.items(), key=lambda kv: -kv[1])) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------------------- studies
def _notes_from(note: dict | None, label: str) -> list:
    return [{"from": label, "text": note["text"]}] if note and note.get("accepted") and note.get("text") else []


class Study:
    """Runs one learning study and writes its summary. Steps are relays (all four legs, legs in parallel)."""

    def __init__(self, model: str, condition: str, *, level: str = "medium", seed: int = 0, transfer_seed: int | None = 1,
                 members: int = 1, steps: int = 3, numbers: str = "allow", relay_name: str = "standard",
                 runs_dir="runs", tracks_dir="tracks", agent_cfg=None, log=print, verbose=False, study_id: str | None = None):
        from .agent import AgentConfig

        assert condition in CONDITIONS
        self.model, self.condition, self.level, self.seed, self.transfer_seed = model, condition, level, seed, transfer_seed
        self.members = members if condition != "chain" else 1
        self.steps, self.policy, self.relay_name = steps, NotePolicy(numbers=numbers), relay_name
        self.runs_dir, self.tracks_dir, self.log, self.verbose = Path(runs_dir), tracks_dir, log, verbose
        self.cfg = agent_cfg or AgentConfig(obs_level="driver", mode="code", control="continuous", clock="paused", perception=level)
        short = model.split(":")[-1].replace("gpt-", "").replace(".", "")
        self.id = study_id or f"{dt.datetime.now():%Y%m%d-%H%M}-{condition}-{short}-{numbers}-{uuid.uuid4().hex[:4]}"
        self.runs: list[dict] = []  # summary rows
        self.lock = threading.Lock()

    # one relay with per-leg note contexts
    def _relay(self, relay, ctx: dict, role: str, step: int, member: int) -> dict:
        from .runner import leg_config_for, make_driver, pretty_label, run_relay, save_run

        drv = make_driver(self.model, self.cfg, turn_log=self.log if self.verbose else None)
        drv.learning = ctx
        rec = run_relay(relay, drv, self.model, leg_config_for(self.cfg.mode, None, self.cfg.control), self.cfg,
                        label=pretty_label(self.model), trial=step, log=self.log, parallel_legs=True, live_dir=self.runs_dir)
        rec["learning"] = {"study": self.id, "condition": self.condition, "role": role, "step": step, "member": member,
                           "numbers": self.policy.numbers, "level": self.level, "relay": relay["id"],
                           "prompt_version": LEARN_PROMPT_VERSION, "policy": asdict(self.policy)}
        save_run(rec, self.runs_dir)
        row = {"run_id": rec["run_id"], "role": role, "step": step, "member": member, "relay": relay["id"],
               "legs": [{"index": L["index"], "vehicle": L["vehicle"], "outcome": L["metrics"]["outcome"],
                         "time": L["metrics"]["time"], "progress": L["metrics"]["progress"], "touches": L["metrics"]["touches"],
                         "notes_in": L.get("notes_in") or [], "note": L.get("note")} for L in rec["legs"]]}
        with self.lock:
            self.runs.append(row)
            self.save()
        fin = sum(1 for L in row["legs"] if L["outcome"] == "FINISHED")
        self.log(f"[study {self.id}] {role} step {step} member {member}: {fin}/{len(row['legs'])} parked")
        return row

    def _ctx(self, notes_by_leg: dict, n_legs: int, write=True, transfer=False) -> dict:
        return {i: LegNotes(self.condition, list(notes_by_leg.get(i, [])), write=write, transfer=transfer, policy=self.policy)
                for i in range(n_legs)}

    def run(self) -> dict:
        from .runner import load_or_generate_relay

        relay = load_or_generate_relay(self.relay_name, self.seed, self.tracks_dir)
        n = len(relay["tracks"])
        final_notes: list[dict] = []  # per member: leg -> notes for the transfer attempt
        if self.condition == "self":
            def member(m):
                mine: dict = {}
                for s in range(1, self.steps + 1):
                    row = self._relay(relay, self._ctx(mine, n), "train", s, m)
                    for L in row["legs"]:
                        mine.setdefault(L["index"], []).extend(_notes_from(L["note"], f"you, attempt {s}"))
                return mine
            with ThreadPoolExecutor(self.members) as ex:
                final_notes = list(ex.map(member, range(self.members)))
        elif self.condition == "chain":
            book: dict = {}
            for s in range(1, self.steps + 1):
                row = self._relay(relay, self._ctx(book, n), "train", s, 0)
                for L in row["legs"]:
                    new = _notes_from(L["note"], f"notebook as rewritten by driver {s}")
                    if new:  # a rejected rewrite keeps the old notebook
                        book[L["index"]] = new
            final_notes = [book]
        else:  # pool
            prev: dict = {}
            for s in range(1, self.steps + 1):
                with ThreadPoolExecutor(self.members) as ex:
                    rows = list(ex.map(lambda m: self._relay(relay, self._ctx(prev, n), "train", s, m), range(self.members)))
                prev = {}
                for row in rows:
                    for L in row["legs"]:
                        prev.setdefault(L["index"], []).extend(_notes_from(L["note"], f"round {s}, driver {row['member'] + 1}"))
            final_notes = [prev]
        if self.transfer_seed is not None:
            other = load_or_generate_relay(self.relay_name, self.transfer_seed, self.tracks_dir)
            by_vehicle = {t.vehicle_key: i for i, t in enumerate(relay["tracks"])}
            with ThreadPoolExecutor(len(final_notes)) as ex:
                def transfer(m):
                    notes = {j: final_notes[m].get(by_vehicle.get(t.vehicle_key), []) for j, t in enumerate(other["tracks"])}
                    return self._relay(other, self._ctx(notes, len(other["tracks"]), write=False, transfer=True),
                                       "transfer", self.steps + 1, m)
                list(ex.map(transfer, range(len(final_notes))))
        return self.summary()

    def summary(self) -> dict:
        return {"study": self.id, "model": self.model, "condition": self.condition, "level": self.level, "seed": self.seed,
                "transfer_seed": self.transfer_seed, "members": self.members, "steps": self.steps,
                "policy": asdict(self.policy), "prompt_version": LEARN_PROMPT_VERSION,
                "runs": sorted(self.runs, key=lambda r: (r["role"] != "train", r["step"], r["member"]))}

    def save(self):
        d = self.runs_dir / "studies"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{self.id}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.summary(), indent=1))
        tmp.replace(p)
        idx = sorted(x.name for x in d.glob("*.json") if x.name != "index.json")
        (d / "index.json").write_text(json.dumps(idx))
