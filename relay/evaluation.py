"""Explicit cohort and study-completion rules for the November 2026 evaluation.

Infrastructure failures are recorded in the dataset, but excluded as whole runs
or whole studies. These rules never remove a run because of poor driving.
"""

CORE_VERSIONS = {"sim_version": 3, "harness_version": 4, "prompt_version": 3, "control_prompt_version": 1}


def cohort_reason(run: dict, legs: list, *, allow_ablations: bool = False) -> str | None:
    if any(L.get("reason") == "model API error" for L in legs):
        return "model API error"
    if run.get("obs") != "driver" or run.get("relay_name") != "standard":
        return "different task or observation"
    if run.get("legs") != 4 or len(legs) != 4 or {L["vehicle"] for L in legs} != {"car", "tractor", "mower", "truck"}:
        return "incomplete relay"
    if (run.get("mode"), run.get("control"), run.get("clock")) != ("code", "continuous", "paused"):
        return "different control configuration"
    if run.get("versions") != CORE_VERSIONS:
        return "different simulator or prompt versions"
    v1 = allow_ablations and run.get("level") == "v1"
    expected_version = 1 if v1 else 2
    if (run.get("perception_version"), run.get("perception_prompt_version")) != (expected_version, expected_version):
        return "different perception versions"
    if run.get("seed") not in (0, 1):
        return "outside release course sets"
    levels = ("easy", "medium", "hard", "v1") if allow_ablations else ("medium",)
    if run.get("level") not in levels:
        return "different perception level"
    return None


def study_reason(study: dict, runs: dict, legs_by_run: dict) -> str | None:
    rows = study.get("runs") or []
    if not rows:
        return "empty study"
    for row in rows:
        rid = row["run_id"]
        if rid not in runs:
            return "missing run record"
        reason = cohort_reason(runs[rid], legs_by_run.get(rid, []))
        if reason:
            return reason
        expected_seed = study.get("transfer_seed", 1) if row["role"] == "transfer" else study.get("seed", 0)
        if runs[rid].get("seed") != expected_seed:
            return "different study course set"
    members = study["members"] if study["condition"] in ("self", "pool") else 1
    expected = {("train", step, member) for step in range(1, study["steps"] + 1) for member in range(members)}
    transfer_members = study["members"] if study["condition"] == "self" else 1
    expected |= {("transfer", study["steps"] + 1, member) for member in range(transfer_members)}
    actual = {(r["role"], r["step"], r["member"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        return "incomplete study"
    return None
