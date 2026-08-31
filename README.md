# SupportOrbit-MuSiQue

An auditable Qwen3-4B post-training project for evidence-aware multi-hop RAG.
It turns each MuSiQue training example into a three-state evidence orbit:
answerable context (`C`), answerable context with replaced non-support evidence
(`D`), and the official paired unanswerable context (`M`).

## Public release boundary

This repository publishes the project-owned data builder, SFT and HopPAIR
objectives, evaluator, tests, frozen protocol, compact comparison artifact, and
result ledger. It intentionally excludes MuSiQue rows and derived JSONL data,
Qwen weights, LoRA adapters, predictions, and complete run directories.

The reported pilot was executed with a fail-closed local harness whose absolute
paths and SHA-256 identities remain in the frozen source and protocol records.
Those records are evidence of the completed run, not a portable one-command
launcher. Reproduction requires separately obtained MuSiQue train data and
Qwen3-4B-Instruct-2507 weights, followed by regeneration of a new local launch
receipt.

## Seed-17 pilot outcome

The CONTROL SupportOrbit-SFT arm succeeded on the frozen 400-orbit internal dev
split: C/D Answer F1 rose from 33.75/34.51% to 61.05/63.73%, missing-evidence
refusal rose from 31.75% to 86.00%, and parse validity rose from 88.17% to
99.92%. A proposed HopPAIR relational auxiliary loss failed the preregistered
continuation gate because it over-sharpened the answer/refuse boundary; it was
stopped before additional seeds, sealed shadow, or official evaluation.

See [RESULTS_SEED17.md](RESULTS_SEED17.md) for the full evidence ledger,
negative-ablation diagnosis, statistical caveats, and claim boundaries.

This is a one-seed result on an internal held-out split derived only from the
official MuSiQue training member. It is not an official-dev/test, external
generalization, leaderboard, or SOTA claim.

The research package constructs deterministic C/D/M evidence orbits from the separately
extracted official MuSiQue **training** member. Production code accepts only the
frozen path and SHA256 recorded in `SOURCE_PROVENANCE.md`; archive, dev, test,
network, and GPU access are outside the builder boundary.

## Inspect the released code

```bash
python -m pip install -e .
python -m pytest -q
```

The builder accepts only the source path and digest frozen for the original
pilot. See `SOURCE_PROVENANCE.md` and
`protocols/support_orbit_pilot_v2.md` before adapting it to a new local checkout.

The current v2 release contains 1,920 train, 400 dev, and 400 sealed-shadow orbits
(three records per orbit). Each orbit has:

- `C`: the official answerable context and answer;
- `D`: C with exactly two non-support slots replaced by deterministic,
  same-split donors; and
- `M`: the paired official unanswerable context.

Every record follows the evaluator contract `id`, `orbit_id`, `source_id`,
`state`, `answer`, `answer_aliases`, `gold_support_idxs`, and `answerable`, with
the fixed 20-paragraph registry and canonical training fields added.

## Safety gates

`prepared_data_v2/manifest.json` is the release authority. `READY` requires:

1. whole-component split isolation on decomposition IDs, C-support text hashes,
   and normalized answers, aliases, and subanswers;
2. exact D construction invariants;
3. grouped five-fold surface separability AUC at or below 0.60 for C–D, D–M,
   and C–M;
4. semantic C–D separability AUC at or below 0.60; and
5. byte-identical canonical formatter integration with zero train/dev examples
   over 6,144 actual Qwen tokens.

Do not open `prepared_data_v2/shadow.sealed.jsonl` for training, model or prompt
selection, threshold tuning, or learned diagnostics. Use only its sealed marker
and writer-produced digest until the frozen protocol explicitly authorizes a
one-time evaluation.

`prepared_data_v1` is withdrawn and retained only for forensic reproducibility;
see its independent `INVALID_AUDIT_RATIO.md` notice. It must not be used for
training or protocol freeze.

## License

Project-owned code is MIT licensed. MuSiQue attribution and the dataset boundary
are recorded in `NOTICE` and `SOURCE_PROVENANCE.md`.
