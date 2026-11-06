# Reproducing the November 2026 evaluation

Install the repository in editable mode, then run `python tools/reproduce.py --verify`. This verifies checksums of the frozen run/study JSON files, rebuilds the index and results digest, and compares all published aggregate numbers, cohort IDs, costs and exclusions with `experiments/evaluation.json`. It makes no model calls. Byte-for-byte regeneration of provider responses is not expected.

## Core configuration

Standard relay, seeds 0 and 1; four vehicles in each run; driver's-seat observation, medium perception; code mode; continuous controls at 10 Hz; paused thinking clock. Simulator v3, harness v4, system prompt v3, continuous-control prompt v1, perception v2 and perception prompt v2. The exact inclusion rules are in `relay/evaluation.py`.

Medium samples: Sol 32 legs (seed 0/1 runs 5/3), Terra 24 (3/3), Luna 20 (3/2), Opus 28 (4/3), Sonnet 32 (4/4), Haiku 32 (4/4), Flash 16 (2/2), GPT-OSS 32 (4/4). There are only two independent course sets. All four legs of each run share a trial. Report these imbalances alongside scores.

A run with any `model API error` leg is excluded whole. A study is eligible only when all expected training members and transfer members have complete run records, every run matches the core configuration and no leg has an API error. A poor driving result is retained. Studies have unequal coverage after exclusions; Opus has team rounds only, Haiku has retries and team rounds, and Sol, Terra and Sonnet have all four conditions.

The difficulty sweep and first perception version are separate ablations. Other tasks and control modes remain inspectable in the raw dataset but do not enter the core scoreboard. `experiments/evaluation.json` lists run IDs, seed counts and exclusions.

## New experiments

Run separately for each seed; choose your own model and budget:

```sh
python -m relay --runs new-runs run --models PROVIDER:MODEL \
  --relay standard --seed 0 --trials 4 --obs driver --perception medium \
  --mode code --control continuous --clock paused --reasoning medium
python -m relay --runs new-runs learn --model PROVIDER:MODEL \
  --condition self --members 2 --steps 3 --numbers allow --seed 0 --transfer-seed 1
```

Repeat seed 1 for the no-notes baseline. Learning conditions in the release use self: 3 steps × 2 members (with and without numbers), chain: 6 generations, pool: 2 rounds × 3 members, followed by transfer. Excluded Opus chain runs used only 3 generations. Model support for explicit effort varies; log the effective setting and CLI version in future work. A CLI alias does not pin a provider-side weight snapshot.

## Interpretation

Wilson intervals treat legs as independent binomial trials; shared vehicles, runs and courses violate that assumption. Fisher tests are exploratory, one-sided, uncorrected and unclustered. There is no matched retry-without-notes control. The data therefore support descriptive comparisons and hypotheses, not causal conclusions about notes or definitive rankings. Transfer uses only one held-out course set.

Cost estimates apply a frozen November 6, 2026 rate sheet to successful reported token usage. They exclude subscription fees, cache storage and some failed calls. The 50% cache-read view is a hypothetical sensitivity scenario. See `PRICE_SOURCES.md`.
