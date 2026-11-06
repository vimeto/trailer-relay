# Contributing

Install `python -m pip install -e '.[dev]'`. Run `pytest -q -m 'not live and not browser'` for ordinary changes. Browser work also requires `python -m playwright install chromium` and `pytest -q -m browser`. Live API tests are opt-in (`RELAY_LIVE=1`) and use your own credentials and budget.

Preserve the frozen release dataset. Use a new directory for experiments (`python -m relay --runs new-runs ...`). Include the model route, requested/effective effort, CLI version, task/prompt/physics/perception versions and seeds with new results. Do not interpret API quota failures as driving failures.

Bump the appropriate version when changing physics, observations, prompt semantics or harness behavior. Add regression coverage for behavior changes. Changes to the released cohort rules or price sheet should be explicit and accompanied by a reviewed evaluation-manifest update. Do not silently replace historical results.

Use Conventional Commits. Do not commit credentials, account sessions, local checkpoints or article drafts. Read `docs/SECURITY.md` before running model-written code. Keep the vendored three.js license with the browser assets.
