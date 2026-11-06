# Controller execution boundary

Model-written Python programs are untrusted code. The simulator remains authoritative and communicates over pipes. Programs run with a scrubbed environment, Python isolated mode, CPU/file-size limits, and process-group wall-time limits.

On macOS, `sandbox-exec` also restricts network access, home-directory reads and writes outside the fresh work directory. Python installation paths are allowed for interpreter startup. On Linux and other POSIX systems these filesystem/network restrictions do not apply automatically: use an isolated container or VM with no credentials, no network and only the required files mounted before running untrusted generated controllers. A subprocess and resource limits alone are not a security sandbox. Windows is not supported in this release.

Subscription model CLIs are another boundary: Codex and Claude Code disable built-in tools; Antigravity uses headless permission denial and an empty temporary workspace. This is not identical tool removal. Never run a CLI route with repository access or extra mounts that change its empty-workspace assumption. Record denied actions and effective settings when extending an adapter.

Do not add keys, provider sessions, `.env` files, spend ledgers or local account configuration to dataset releases. `tools/audit_public.py` and a separate Gitleaks scan are used before publishing. Report vulnerabilities privately through GitHub's security advisory mechanism when available, rather than placing credentials in an issue.
