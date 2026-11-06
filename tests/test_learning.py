"""Learning studies (relay/learning.py): the note check, notes in the briefing, and the three conditions end to end."""
import json

import pytest

from relay.agent import AgentConfig
from relay.learning import LegNotes, NotePolicy, Study, check_note, notes_section

GOOD = ("What went wrong: I watched the camera through the bend and the car's front-left corner swung into the rail. "
        "Priority: glance ahead before tightening the turn. Tip: start the bend early and slowly.")


@pytest.mark.parametrize("bad, why", [
    ("def control(obs): return 0, -1", "keyword"),
    ("steer = 1.7 * (hitch - desired)", "assignment"),
    ("use 1.7*(h-d) for the steering", "formula"),
    ("read obs['head'] each tick", "access"),
    ("keep hitch_deg small", "variable name"),
    ("set {steer: 10}", "operators"),
    ("def control(o):\n    pass", "keyword"),
    ("if hitch > 30:\n  stop", "code block"),
    ("return -1, 0", "keyword"),
    ("```python\nx\n```", "code formatting"),
    ("x" * 700, "too long"),
    ("turn at 1, 2, 3, 4, 5, 6 and 7 m", "numbers"),
])
def test_check_rejects_code_formulas_and_too_many_numbers(bad, why):
    reasons = check_note(bad, NotePolicy())
    assert reasons and any(why in r for r in reasons), reasons


@pytest.mark.parametrize("prose", [
    "Keep the trailer straight; this short cart responds well to tiny corrections.",
    "Watch both sides while making the turn, then return to the centre.",
    "For each bend in the course, start early. Forward -> reverse costs time.",
    "If the right side closes: pull forward, reset the line, and reverse again.",
])
def test_check_accepts_ordinary_prose(prose):
    assert check_note(prose, NotePolicy()) == []


def test_check_accepts_prose_and_enforces_no_numbers():
    assert check_note(GOOD, NotePolicy()) == []
    assert check_note("Start the turn when the rail is 4 m away, at 0.5 m/s.", NotePolicy()) == []
    assert check_note("Start the turn when the rail is 4 m away.", NotePolicy(numbers="none"))
    assert check_note("Start the turn when the rail is four metres away.", NotePolicy(numbers="none"))
    assert check_note(GOOD, NotePolicy(numbers="none")) == []  # 'one tip' style words are fine; GOOD has no numbers


def test_notes_section_framing():
    ln = LegNotes("self", [{"from": "you, attempt 1", "text": GOOD}])
    s = notes_section(ln)
    assert "YOUR OWN NOTES" in s and "THIS course" in s and GOOD in s
    assert "DIFFERENT course" in notes_section(LegNotes("chain", [{"from": "x", "text": "y"}], transfer=True))
    assert notes_section(LegNotes("pool", [])) == ""


@pytest.mark.parametrize("condition, members, steps, runs", [("self", 2, 2, 2 * 2 + 2), ("chain", 1, 3, 3 + 1), ("pool", 2, 2, 2 * 2 + 1)])
def test_study_end_to_end(tmp_path, condition, members, steps, runs):
    cfg = AgentConfig(obs_level="driver", mode="code", control="continuous", clock="paused", perception="medium")
    st = Study("scripted:ctrl-camera", condition, members=members, steps=steps, runs_dir=tmp_path / "runs",
               tracks_dir="tracks", agent_cfg=cfg, log=lambda m: None, relay_name="warmup")
    summ = st.run()
    assert len(summ["runs"]) == runs
    saved = json.loads((tmp_path / "runs" / "studies" / f"{st.id}.json").read_text())
    assert saved["condition"] == condition and len(saved["runs"]) == runs
    train = [r for r in saved["runs"] if r["role"] == "train"]
    transfer = [r for r in saved["runs"] if r["role"] == "transfer"]
    first, later = [r for r in train if r["step"] == 1], [r for r in train if r["step"] == 2]
    assert all(not L["notes_in"] for r in first for L in r["legs"])  # nothing to read on the first attempt
    assert all(L["note"]["accepted"] for r in train for L in r["legs"])  # every driver left a note
    assert all(L["notes_in"] for r in later for L in r["legs"])
    if condition == "pool":  # the second round reads every note of the first
        assert all(len(L["notes_in"]) == members for r in later for L in r["legs"])
    if condition == "chain":  # one notebook, rewritten
        assert all(len(L["notes_in"]) == 1 for r in later for L in r["legs"])
    assert transfer and all(L["notes_in"] and L.get("note") is None for r in transfer for L in r["legs"])
    rec = json.loads(next((tmp_path / "runs").glob(f"{transfer[0]['run_id']}.json")).read_text())
    assert rec["learning"]["role"] == "transfer" and rec["learning"]["study"] == st.id
    assert "DIFFERENT course" in rec["legs"][0]["briefing"]
