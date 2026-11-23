# Frozen v14 Protocol: OpenEQA-HM3D Confirmation

Date frozen: 2026-09-13

## Question

Does the already selected GeoGate configuration transfer from ScanNet-based
development evidence to the official OpenEQA-HM3D domain without material
loss in answer quality?

This protocol confirms a fixed method on a new scene domain. It does not tune
the pruning layer, token budget, attention scorer, prompt, or acceptance
margin.

## Fixed data and model

- Data: all 557 official `hm3d-v0` questions in official JSON order,
  covering 63 episodes/scenes.
- Model: the same downloaded LLaVA-3D-7B checkpoint used by the paper.
- Conditions: native LLaVA-3D and GeoGate at fixed L14 with keep ratio 0.10.
- Score: last-prompt attention, with no stable, coverage, or adaptive variant.
- Prompt: `<image>\n{official question}` rendered by `llava_v1`, with no
  answer suffix.
- Decoding: greedy, one beam, at most 512 new tokens.
- Seed: 20260913.
- Condition order: native, then GeoGate, for every question.

No layer, budget, threshold, prompt, or selector may change after a smoke or
after generated answers are inspected.

## Deterministic multimodal input

The upstream `mode='random'` OpenEQA path is prohibited. Each episode uses the
official extracted RGB-D, pose, and intrinsic files with:

1. lexicographically sorted `*-rgb.png` frames;
2. deterministic uniform indices
   `floor(i * available_frames / target_frames)`;
3. exactly the model processor's frozen frame count;
4. repeated indices only when an episode has fewer frames than required;
5. paired `*-depth.png` and pose `.txt` validation;
6. HM3D depth conversion by `uint16 / (65535 / 10 m)`;
7. identity axis alignment for Habitat world poses;
8. SHA256 binding of indices, selected files, source files, and resulting
   RGB/depth/pose/intrinsic tensors.

For the audited checkpoint the frozen frame count is 20. The runner aborts if
the loaded processor reports any other value.

Native and GeoGate reuse the same tensors. If the internal 2560-token visual
cap triggers, both conditions must use the same deterministic scene-level
indices and hash. Each question is multimodally prepared exactly once;
native and GeoGate generation must reuse the identical prepared-embedding
object, preserve its SHA256 after both calls, and record two audited prefill
cache hits.

## Staged execution

Execution remains blocked until authorized assets exist.

1. CPU preflight validates all 63 episode directories, frame tuples,
   intrinsics, finite poses, and selected-frame depth ranges. Its entry point
   is `analysis/preflight_openeqa_hm3d_v14.py`.
2. A one-episode construction smoke runs one question twice per condition
   and requires exact frame, tensor, cap, output-token, and trace hashes.
3. A two-episode smoke covers one short and one long episode and checks peak
   memory, complete generation, and output formatting.
4. Only after both smokes pass may the full 557-question paired run start.
5. Raw JSONL and the manifest are pulled back and structurally audited before
   any answer score is computed.

An incomplete question pair is never scored. Existing outputs are never
overwritten. The structural audit entry point is
`analysis/audit_openeqa_hm3d_v14.py`; only a full audit with
`scoring_authorized=true` can proceed to the official evaluator.

## Metrics and uncertainty

Primary metric: the official OpenEQA LLM-Match score on the 0--100 scale,
using the official prompt and a scorer model/version frozen and availability-
checked before scoring begins.

Primary effect:

```text
Delta = LLM-Match(GeoGate) - LLM-Match(native)
```

The primary 95% confidence interval uses 10000 episode-clustered bootstrap
resamples with seed 20260913. Every resampled episode contributes all of its
questions. Question-level paired intervals are secondary.

The non-inferiority margin is frozen at -2.0 LLM-Match points. This is an
operational tolerance selected before generation and scoring, not an estimate
of benchmark measurement noise.

Secondary reporting:

- exact normalized answer match;
- token-F1 against official answer and extra answers;
- per-category paired deltas with exploratory episode bootstrap intervals;
- wins/ties/losses under the official integer LLM-Match grade;
- answer length, generation latency, selected token count, and cap/onset
  traces.

Lexical metrics are diagnostics and cannot replace official LLM-Match.
Because the official evaluator is model-based, its exact API model, seed,
temperature, prompt hashes, retry outcomes, and raw grades must be retained.
After those grades exist, the API-free paired summarizer is
`analysis/score_openeqa_hm3d_v14.py`.

## Decision rule

The independent-domain quality claim is promoted only if all correctness
gates pass and the lower bound of the primary episode-bootstrap 95% interval
is at least -2.0 points.

- If the lower bound is above 0, report directional improvement.
- If it is in `[-2.0, 0]`, report non-inferior transfer under the frozen
  margin, not improvement.
- If it is below -2.0, retain the result as a failed confirmation and do not
  add an independent-quality claim to the paper.
- If the official scorer cannot be frozen or audited, report generated
  predictions as unscored and do not make a quality claim.

The checkpoint-manifest limitation remains in every outcome.

## Paper gate

`docs/paper_cvpr.tex` may be changed only after:

1. authorized asset provenance is recorded;
2. construction and resource smokes pass;
3. all 557 native/GeoGate pairs complete;
4. structural and hash audits pass;
5. official scoring and the frozen bootstrap complete; and
6. the decision rule above passes.
