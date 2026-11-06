# Frozen price sheet, checked November 6, 2026

USD per million tokens. These are hypothetical standard token charges for recorded subscription-CLI traffic, not invoices. Cache usage is route-specific; a 50% read share is a scenario, not a measured normalization of actual compute.

| Configuration | Input | Cached read | Output | Official source |
|---|---:|---:|---:|---|
| GPT-6.1 Sol | 2.00 | 0.10 | 10.00 | [OpenAI pricing](https://developers.openai.com/api/docs/pricing?tab=suite) |
| GPT-6.1 Terra | 1.00 | 0.05 | 5.00 | [OpenAI pricing](https://developers.openai.com/api/docs/pricing?tab=suite) |
| GPT-6 Luna | 0.10 | 0.01 | 0.50 | [OpenAI pricing](https://developers.openai.com/api/docs/pricing?tab=suite) |
| Claude Opus 5.5 | 4.00 | 0.20 | 20.00 | [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Claude Sonnet 5.5 | 2.00 | 0.10 | 10.00 | [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Claude Haiku 5.5, prompts ≤100K | 0.10 | 0.01 | 0.50 | [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Gemini 3.8 Flash, global intro | 0.75 | 0.075 | 3.75 | [Google Cloud pricing](https://cloud.google.com/model-garden/pricing) |
| GPT-OSS 120B, Fireworks | 0.15 | 0.015 | 0.60 | [Fireworks model page](https://fireworks.ai/models/fireworks/gpt-oss-120b) |

Flash's introductory period ends December 31, 2026. Fireworks is a reference host, not evidence that Antigravity used that host. Rates can change; preserve the release sheet when reproducing historical results.

Cache-write counts and TTLs were not retained separately. The analysis reports an additional sensitivity estimate that bills all uncached Anthropic input as 5-minute writes (1.25× plain input). OpenAI's separately priced explicit cache writes, Google cache storage, long-context premiums and retry charges are not included in the main read/fresh estimate. Haiku successful recorded turns are below 100K prompt tokens; this would need request-level tier accounting for longer prompts.

Antigravity excludes cache reads from its input-token field, so the analysis adds them back. Other routes include reads in input. Reasoning tokens already count in output and are not billed twice. CLIs add different preambles and compact histories differently. All cost-per-parked-leg estimates depend on small, noisy success counts; a zero-success sample has no finite empirical ratio.
