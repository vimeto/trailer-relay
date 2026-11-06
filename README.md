# Trailer Relay

A simulator and evaluation harness for reversing vehicles with trailers. Language-model drivers get partial observations, issue controls or write Python controllers, and can pass short notes to later drivers. Every recorded attempt can be replayed in a three.js viewer.

**[Interactive results and replays](https://trailer.games.vtoivonen.com/)** · [Reproduce the evaluation](docs/REPRODUCIBILITY.md) · [Dataset](docs/DATASET.md) · [Extend the environment](docs/EXTENDING.md)

The November 2026 experiment compares eight model configurations on a car, tractor, ride-on mower and semi truck. Sol parks 16/32 valid core legs; Opus parks 14/28. This is an exploratory comparison through subscription CLIs, with unequal samples, route-specific preambles and default reasoning efforts. It is not a controlled comparison of model weights alone. Notes and repeated attempts are not experimentally separated.

## Install and try it

Python 3.11+ on macOS or Linux. The simulator and replay viewer need no model credentials.

```sh
git clone https://github.com/vimeto/trailer-relay.git
cd trailer-relay
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m relay run --models autopilot --relay mini
python -m relay serve
```

Open the printed viewer URL. `relay` is also installed as a command. By default, tracks and run records are written under the current working directory; `--runs` and `--tracks` go before the subcommand. `RELAY_WORKSPACE` sets a different workspace root.

For a model run, supply your own credentials through environment variables, or use an installed subscription CLI on your own account. Nothing is charged by installing or replaying this dataset.

```sh
export OPENAI_API_KEY=...  # set in your shell, never in a committed file
python -m relay run --models openai:YOUR_MODEL_ID --relay standard \
  --seed 0 --obs driver --perception medium --mode code \
  --control continuous --clock paused --trials 2 --budget-usd 1
```

The API budget is checked between requests; concurrent in-flight calls can overshoot it. Verify current model availability and rates before new experiments. Historical model identifiers in this dataset are the exact requested CLI identifiers; provider-side weight snapshots were not independently verified.

Model-written code runs in a subprocess. **Read [the execution boundary](docs/SECURITY.md) before running untrusted controllers.** The macOS implementation uses `sandbox-exec`; Linux requires an external sandbox/container for equivalent filesystem and network isolation.

## What is included

| Path | Contents |
|---|---|
| `relay/` | Kinematics, collision handling, track generation, perception, controllers, provider adapters, experiment orchestration and analysis |
| `viewer/` | Buildless three.js replay explorer; bundled with the wheel |
| `tracks/` | Seeded, autopilot-certified courses and older physics caches |
| `runs/` | Experiment transcripts, controller programs, metrics, events and traces; archived invalid attempts are retained separately |
| `experiments/` | Frozen evaluation, cohort definition and raw-record SHA-256 checksums |
| `tests/` | Simulator, harness, adapter, analysis and browser tests |
| `examples/` | A direct simulator rollout showing the extension surface |

The article, editorial drafts, design iterations and website deployment live in a separate private repository. They are not needed to run, inspect or extend the simulator.

## Validate

```sh
python tools/reproduce.py --verify  # local analysis only; no API calls
pytest -q -m 'not live and not browser'
python -m playwright install chromium
pytest -q -m browser
python -m build
```

New research runs should use a separate `--runs` directory to preserve the frozen release. See [CONTRIBUTING.md](CONTRIBUTING.md).

MIT for the original software and experiment records; the vendored three.js license is retained in `viewer/vendor/LICENSE`. Provider names belong to their respective owners. Cite the project using [CITATION.cff](CITATION.cff).
