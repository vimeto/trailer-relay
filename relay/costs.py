"""API cost accounting and a hard spend limit.

Prices are USD per 1M tokens: (input, cached input, output). Reasoning tokens are billed as output
and are already included in output_tokens. The `codex` route runs on the CLI's own login, so it is
recorded at $0 (with the API-equivalent cost kept separately for reference).

The ledger (runs/ledger/spend.json) is shared by every process and guarded with a file lock. A `Budget`
refuses new model calls once the spend of its tag would pass the limit, so concurrent legs can
overshoot by at most one call each.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path

# USD per 1M tokens: input, cached input, output. Sources (Nov 2026): OpenAI's GPT-6.1 Sol announcement;
# Requesty's pricing posts for GPT-6.1 Terra and GPT-6 Luna (third-party; treat as provisional).
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-sol": (2.00, 0.10, 10.00),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
}


class BudgetExceeded(RuntimeError):
    pass


SUBSCRIPTION_ROUTES = ("codex:", "claude-code:", "agy:")  # CLIs on the user's own logins: billed $0


def model_of(spec: str) -> str:
    return spec.split(":", 1)[-1].split("@")[0]


def usd(spec: str, usage: dict) -> tuple[float, float]:
    """(billed cost, API-equivalent cost) of one usage record for a model spec."""
    p = PRICES.get(model_of(spec))
    if not p:
        return 0.0, 0.0
    inp, cached, out = usage.get("input_tokens", 0), usage.get("cached_tokens", 0), usage.get("output_tokens", 0)
    eq = ((inp - cached) * p[0] + cached * p[1] + out * p[2]) / 1e6
    billed = 0.0 if spec.startswith(SUBSCRIPTION_ROUTES + ("scripted:",)) else eq
    return billed, eq


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"total_usd": 0.0, "by_tag": {}, "by_model": {}}


@dataclass
class Budget:
    """Shared spend ledger with an optional hard limit for one tag (e.g. 'sol-terra')."""

    path: Path
    limit_usd: float | None = None
    tag: str = "default"

    def spent(self, tag: str | None = None) -> float:
        with _locked(self.path):
            d = _read(self.path)
        return d["by_tag"].get(tag or self.tag, 0.0) if (tag or self.tag) else d["total_usd"]

    def check(self):
        if self.limit_usd is not None and self.spent() >= self.limit_usd:
            raise BudgetExceeded(f"budget '{self.tag}' exhausted: ${self.spent():.2f} of ${self.limit_usd:.2f}")

    def charge(self, spec: str, usage: dict) -> float:
        billed, eq = usd(spec, usage)
        with _locked(self.path):
            d = _read(self.path)
            d["total_usd"] = round(d["total_usd"] + billed, 6)
            d["by_tag"][self.tag] = round(d["by_tag"].get(self.tag, 0.0) + billed, 6)
            m = d["by_model"].setdefault(spec, {"usd": 0.0, "api_equiv_usd": 0.0, "calls": 0})
            m["usd"] = round(m["usd"] + billed, 6)
            m["api_equiv_usd"] = round(m["api_equiv_usd"] + eq, 6)
            m["calls"] += 1
            self.path.write_text(json.dumps(d, indent=1))
        return billed
