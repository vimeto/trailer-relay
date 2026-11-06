# Dataset and provenance

`runs/*.json` contains the recorded experiments from November 5–6, 2026. `runs/studies/*.json` holds note-passing study membership and notes. `runs/_archive/` retains earlier physics/control versions, aborted probes, quota failures and server-capacity failures. Presence in this repository does not imply eligibility for the publication.

Each run records the requested model identifier and route, task/courses, configuration versions, vehicle specifications, metrics, model text, tool calls/controller source, token counts, simulated and wall times, trace samples and events. Studies identify role, step, member and transfer seed. The code and exact CLI request construction are in `relay/models/`; historical responses are not guaranteed to repeat.

`experiments/evaluation.json` is the frozen published analysis, with core run IDs, per-model sample/seed counts, valid studies and exclusion reasons. `experiments/SHA256SUMS.json` covers run and study records, including archives; generated `index.json` and `results.json` are omitted from checksums because they are rebuilt. To verify and recompute, use `python tools/reproduce.py --verify`.

Raw records retain API failures for auditing. The hosted explorer exports only valid studies and runs without API errors, and thins traces to 5 Hz while preserving events. The public research records retain the original trace cadence. Neither folder contains live checkpoints, spend ledgers, personal Claude session transcripts, credentials or article drafts. Machine-specific home and Claude temporary paths are redacted; model text and programs otherwise remain as recorded. The private workspace preserves the original files.

Dataset license: MIT for the original project records. Model/service names and third-party software remain subject to their respective ownership and licenses. No model weights are distributed.
