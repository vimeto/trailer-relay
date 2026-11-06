"""Release gates: configuration drift, quota censoring and partial studies."""
from copy import deepcopy

from relay.evaluation import CORE_VERSIONS, cohort_reason, study_reason


def record():
    return {"run_id": "r", "obs": "driver", "relay_name": "standard", "legs": 4, "mode": "code",
            "control": "continuous", "clock": "paused", "versions": CORE_VERSIONS.copy(), "perception_version": 2,
            "perception_prompt_version": 2, "seed": 0, "level": "medium"}


def legs():
    return [{"vehicle": v, "reason": "the leg clock ran out"} for v in ("car", "tractor", "mower", "truck")]


def test_cohort_keeps_bad_driving_and_rejects_api_failure_and_configuration_drift():
    r, ls = record(), legs()
    assert cohort_reason(r, ls) is None
    for key, value in [("clock", "realtime"), ("mode", "tools"), ("seed", 2), ("perception_version", 1)]:
        changed = {**r, key: value}
        assert cohort_reason(changed, ls) is not None
    assert cohort_reason(r, ls[:-1]) == "incomplete relay"
    ls[2]["reason"] = "model API error"
    assert cohort_reason(r, ls) == "model API error"


def test_study_needs_every_member_and_transfer_and_no_api_errors():
    s = {"condition": "self", "members": 2, "steps": 2, "runs": []}
    records, by_run = {}, {}
    for step, role in [(1, "train"), (2, "train"), (3, "transfer")]:
        for member in range(2):
            rid = f"r{step}{member}"
            records[rid] = {**record(), "run_id": rid, "seed": 1 if role == "transfer" else 0}
            by_run[rid] = legs()
            s["runs"].append({"run_id": rid, "role": role, "step": step, "member": member})
    assert study_reason(s, records, by_run) is None
    partial = deepcopy(s)
    partial["runs"].pop()
    assert study_reason(partial, records, by_run) == "incomplete study"
    by_run["r31"][1]["reason"] = "model API error"
    assert study_reason(s, records, by_run) == "model API error"
    del records["r31"]
    assert study_reason(s, records, by_run) == "missing run record"
