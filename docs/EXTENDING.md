# Extending Trailer Relay, including reinforcement learning

The environment is independent of model APIs. `Leg(track, config=LegConfig(...))` owns physics, collisions, timers, penalties and terminal conditions. `control_step(dt)` advances held targets without stopping between decisions. `observe(leg, level)` returns the permitted observation; `metrics()` exposes evaluator information. The harness separates the authoritative simulator from generated controller code.

Start with `python examples/direct_rollout.py`. It drives the simulator directly, makes no API requests and writes a small JSON summary. It is an integration example, not a competitive policy.

## Surfaces

- Vehicles: add a `VehicleSpec` in `relay/vehicles.py`, then supply perception geometry in `relay/perception.py`.
- Courses: add a template/relay in `relay/seeder.py`; certify it with the privileged autopilot and cache its seed.
- Models: implement `ModelAdapter` in `relay/models/base.py` and register it in `relay/models/registry.py`.
- Policy interfaces: discrete `drive` commands, continuous held controls, or a persistent `control(obs)` process at 10 Hz.
- Evaluation: keep the released cohort immutable; create a new experiment manifest for a changed task or prompt.

## RL roadmap

No Gymnasium wrapper or RL trainer is claimed in v0.1.0. A useful next contribution is a wrapper with `reset(seed)`, `step(action)`, action spaces for steering/signed speed/gaze, and a documented fixed observation schema (current rail-point lists are variable length). Create a new `Leg` and attach seeded perception on every reset. Keep `info`/privileged state separate from policy input.

Use a fixed simulated step and define reward explicitly: progress change, contact penalties and terminal completion are candidate components. Treat time/command limits as truncations and successful parking or mechanical disqualification as terminations according to the selected task contract. Record the reward definition and version. Avoid training on the release course seeds if reporting generalization to them; create held-out seeds and compare to the privileged autopilot and blind/hitch baselines. Batch kinematics exists in `relay/kinematics.py`; a vectorized environment still needs collision/perception/reset orchestration.

Changing reward, observation schema, physics, terminal rules, action frequency or train/test seeds defines a new benchmark variant. Keep those versions and trained checkpoint provenance in the experiment record.
