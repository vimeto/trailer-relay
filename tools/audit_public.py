"""Check a prepared public tree without printing matched secret values."""
from __future__ import annotations
import argparse
import re
from pathlib import Path

PATTERNS = {
    "credential": re.compile(r"(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AIza[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"),
    "local home path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
}
FORBIDDEN = {".env", ".secrets", "SUBMISSION_BRIEF.md", "default.toml"}


def audit(root):
    errors = []
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file() or any(part in {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "dist", "build", "qa"} for part in p.relative_to(root).parts):
            continue
        rel = str(p.relative_to(root))
        if p.name in FORBIDDEN:
            errors.append((rel, "private file"))
        if p.stat().st_size > 95 * 1024 * 1024:
            errors.append((rel, "file exceeds GitHub size budget"))
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".woff", ".woff2", ".gz", ".zip", ".whl"}:
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append((rel, name))
    return errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    args = ap.parse_args()
    problems = audit(args.root.resolve())
    for rel, kind in problems:
        print(f"{kind}: {rel}")
    print(f"Public-tree audit: {len(problems)} findings")
    raise SystemExit(bool(problems))
