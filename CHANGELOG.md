# Changelog

## 0.1.0, November 6, 2026

First research release. It includes four articulated vehicles, seeded courses that a scripted autopilot has certified as drivable, partial driver's-seat observations, discrete and continuous controls, a model harness that works with provider APIs and subscription CLIs, sandboxed Python controllers, note-passing studies, the recorded experiments and a 3D replay explorer that needs no build step.

Preparing the release added explicit cohort and version checks, whole-study exclusion when a study hit an API error, checks that every study member is complete, a corrected Claude Sonnet 5.5 cache-read rate, links to official price sources, frozen evaluation manifests, reproducibility tooling and packaging. The raw historical records are kept as they were.

Known limits: two core course sets, unequal samples per model, prompts and reasoning effort that depend on the CLI route, exploratory statistics that ignore shared courses, and no retry-without-notes control. A reinforcement learning interface is described in `docs/EXTENDING.md` but not implemented.
