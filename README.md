# Trailer Relay

A simulator and evaluation harness for reversing vehicles with trailers. Language models drive from the driver's seat: they see rail points from wherever their head points, plus mirrors and a reversing camera, and they write Python controllers that steer the rig. They can leave short notes for later drivers. Every recorded attempt can be replayed in a three.js viewer.

**[Interactive results and replays](https://trailer.games.vtoivonen.com/)** · [Reproduce the evaluation](docs/REPRODUCIBILITY.md) · [Dataset](docs/DATASET.md) · [Extend the environment](docs/EXTENDING.md)

## Why trailers

Backing up a trailer means planning for a position you can't see yet. To get round a 90-degree bend, the trailer has to be at the right angle before the bend starts, and the driver has to know where the tow vehicle's own corners are the whole time. People learn this by feel. Language models have probably seen little of it in training, so it tests whether their spatial reasoning generalises.

The task is far from solved. In the November 2026 evaluation of eight models on a car, a tractor, a ride-on mower and a semi truck, GPT-6.1 Sol parks 16 of 32 legs and Claude Opus 5.5 parks 14 of 28. Four models park none. All decisions are still made on a 2D plan, and the courses leave room to spare, so the benchmark can get much harder: courses at each rig's turning limit, decisions in 3D, wider or asymmetric trailers, and training rather than only evaluation.

Read the results as exploratory. Models run through their vendors' subscription CLIs at default reasoning effort, with different preambles and unequal sample sizes, so this is not a controlled comparison of model weights. Notes and repeated attempts always come together, so the effect of notes alone is not measured.

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

Open the printed viewer URL. Installing also adds a `relay` command. Tracks and run records go under the current working directory unless you pass `--runs` and `--tracks` before the subcommand or set `RELAY_WORKSPACE`.

To run a model, supply your own credentials through environment variables or use a subscription CLI installed on your own account. Installing the package and replaying the dataset cost nothing.

```sh
export OPENAI_API_KEY=...  # set in your shell, never in a committed file
python -m relay run --models openai:YOUR_MODEL_ID --relay standard \
  --seed 0 --obs driver --perception medium --mode code \
  --control continuous --clock paused --trials 2 --budget-usd 1
```

The harness checks the API budget between requests, so calls already in flight can overshoot it. Check current model availability and rates before new experiments. Model identifiers in the dataset are the exact identifiers sent to each CLI. Nothing verifies which weight snapshot the provider served.

Model-written code runs in a subprocess. **Read [the execution boundary](docs/SECURITY.md) before running untrusted controllers.** On macOS the harness uses `sandbox-exec`. On Linux you need your own container or VM for the same filesystem and network isolation.

## What is included

| Path | Contents |
|---|---|
| `relay/` | Kinematics, collision handling, track generation, perception, controllers, provider adapters, experiment orchestration and analysis |
| `viewer/` | Buildless three.js replay explorer; bundled with the wheel |
| `tracks/` | Seeded, autopilot-certified courses and older physics caches |
| `runs/` | Experiment transcripts, controller programs, metrics, events and traces; archived invalid attempts are retained separately |
| `experiments/` | Frozen evaluation, cohort definition and raw-record SHA-256 checksums |
| `tests/` | Simulator, harness, adapter, analysis and browser tests |
| `examples/` | A direct simulator rollout, a starting point for extensions |

The article and its drafts live in a separate repository. You don't need them to run, inspect or extend the simulator.

## Validate

```sh
python tools/reproduce.py --verify  # local analysis only; no API calls
pytest -q -m 'not live and not browser'
python -m playwright install chromium
pytest -q -m browser
python -m build
```

Put new runs in a separate `--runs` directory so the frozen release stays intact. See [CONTRIBUTING.md](CONTRIBUTING.md).

The software and experiment records are MIT licensed. The vendored three.js license is in `viewer/vendor/LICENSE`. Provider names belong to their owners. To cite the project, use [CITATION.cff](CITATION.cff).
