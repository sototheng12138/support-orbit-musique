# Support-Orbit MuSiQue pilot v1

## Current decision

The machine protocol is `FROZEN_PRE_GPU`, while the launch state remains
`STOP_BEFORE_GPU`. The exact `prepared_data_v1/manifest.json` has SHA256
`9d5e13b893e7c456872111da223487c526ec25cafb9c88eb549cbb1e2cb44ca3`; independent
validation passed every construction, learned-probe, and tokenizer gate without opening the
sealed-shadow body. Freezing the protocol does not authorize training: a separate fail-closed
launch receipt must still bind the protocol, data, model, tokenizer, code, schedule,
environment, and tests.

The pilot asks whether support-preserving paired consistency can make a 4B language model
retain the same multi-hop answer and evidence under replacement of irrelevant paragraphs,
while preserving ordinary answering and refusing when evidence is incomplete. It is a
parameter-efficient post-training experiment, not a workflow-only RAG project.

## Source and access boundary

The source is official MuSiQue full v1.0 under CC BY 4.0, repository commit
`922ac98f19a201998dbdae6d7f2887a5258dbdeb`. The archive SHA256 is
`98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd`; the separately
extracted train member contains 39,876 rows/19,938 pairs and has SHA256
`b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034`. The license SHA256
is `cce5d01fa4a83b794271bd2c28cffdf99afd43c803e6ddefddae39b591ea7448`.

Before training, only that exact train file may be read by the builder. Training may later
read only `prepared_data_v1/train.jsonl`; internal evaluation may read only
`prepared_data_v1/dev.jsonl`. Public manifests, audits, and hash sidecars are allowlisted.
The sealed shadow body may not be opened until three frozen-seed dev comparisons pass. All
official MuSiQue dev/test members and derivatives remain unextracted, unread, and outside
this pilot. Tests use synthetic fixtures and public audit metadata; they may not inspect
shadow or official dev/test.

## C/D/M support orbit

Each state exposes exactly 20 paragraphs in slots P00–P19.

- `C` is the official answerable row. Its answer and supporting-paragraph indices are gold.
- `M` is its official paired unanswerable row. Eligibility additionally requires that M omit
  at least one raw C-support text hash and at least one decomposition-subanswer token
  sequence. Its exact target is
  `U | evidence=[] | answer=INSUFFICIENT_EVIDENCE`.
- `D` copies C and replaces exactly two non-support slots using non-support donors from other
  orbits in the same split. Every C support paragraph, support slot, answer, and target is
  unchanged. Donors must be leakage-filtered, have token-length ratio at most 1.25, and be
  matched for surface and lexical characteristics.

The supported target is exactly
`S | evidence=[P00,P01] | answer=<answer>`. The parser accepts only the full canonical S/U
grammar, P00–P19, and ascending unique evidence IDs; it never repairs outputs.

Whole leakage components are assigned to the 1,920/400/400 train/dev/shadow split.
Components connect on shared decomposition IDs, raw C-support text hashes, or normalized
answers, aliases, and subanswers. Every train/dev/shadow leakage-key intersection must be
zero.

## CPU data release gates

All gates are conjunctive:

1. All 2,720 selected orbits must preserve the exact C/D target and all C support paragraphs,
   replace exactly two non-support slots, and use same-split donors with zero length-ratio
   violations.
2. An orbit-grouped deterministic 5-fold out-of-fold logistic probe uses 18 state-blind
   document surface features. C–D, D–M, and C–M separability
   `max(AUC, 1−AUC)` must each be at most 0.60. Shadow is excluded.
3. A second diagnostic adds question-to-document lexical-overlap features. C–D separability
   must be at most 0.60; D–M and C–M are diagnostic because missing support is a real semantic
   distinction. No answer-derived feature is allowed.
4. The pinned Qwen tokenizer audit covers train and dev only. Every canonical sequence must
   fit within 6,144 tokens with zero truncation, zero prepared/canonical rendering mismatch,
   native `enable_thinking=false` chat formatting, completion-only labels, and supervised EOS.

The observed surface separability AUCs are 0.53753084/0.55966205/0.56091353 for
C–D/D–M/C–M. Semantic C–D is 0.54053563; D–M/C–M are diagnostic at
0.63325821/0.63463566. All 6,960 canonical train+dev records pass, with maximum length
4,974/6,144, zero over-limit rows, zero rendering mismatches, and zero truncation. The train
and dev SHA256 values are respectively
`e450639b573521be2675c053f8f96f1ce396402bdb350a192d01b98e3e8d23fc` and
`0bfb83a98d057cf612c3236ed419a370bbf0229a4778c521520c516688b5ec69`.
The shadow digest
`eaae9d5ec128b3f1deececddb72cc5589a5dfbdbb7b312bc08e3c79ebbdc4eb5` is bound only from
the builder's streaming writer, marker, and manifest; validation did not reopen its body.

The freeze validator reads train/dev and public audit artifacts but deliberately never opens
the shadow JSONL. It binds the shadow digest transitively through the builder-emitted sealed
marker and manifest.

## Matched training arms

Both arms see the identical atomic sequence `[C,D,M]` and run all three rows in one forward.
The state-balanced SFT term is the mean completion-token CE within each state, then the
unweighted mean across C, D, and M.

- `CONTROL`: `L = L_sft`.
- `HopPAIR`:
  `L = L_sft + 0.1 KL(stopgrad(p_C) || p_D) + 0.2 L_flip`, where KL is averaged over
  shared C/D completion ordinals and
  `L_flip = 0.5[softplus(2−(z_C−z_M)) + softplus(2−(z_D−z_M))]`, with
  `z = logit(S)−logit(U)` at the first assistant-token prediction.

The control isolates the auxiliary objective: data, targets, one-forward exposure, orbit
order, initialization, optimizer, and final checkpoint are otherwise identical.

Each arm completes one exact epoch: atomic microbatch 1 orbit, accumulation 8 orbits,
effective batch 8 orbits, 1,920/8 = 240 optimizer steps. The first seed is 17; seeds 29 and
43 run only if the previous frozen dev comparison is GO. Other frozen settings are Qwen3-4B
Instruct 2507, LoRA r16/alpha32/dropout0.05 across all transformer linear projections,
BF16, SDPA, gradient checkpointing, fused AdamW, learning rate 2e-4, cosine schedule,
warmup ratio 0.03/eight steps, weight decay 0, and max gradient norm 1.0. Exact model and
tokenizer hashes belong in the separate launch receipt.

Any truncation, dropped/duplicate orbit, non-finite loss/gradient/parameter/optimizer state,
skipped optimizer step, resume, or unauthorized data read is a hard stop. Only step 240 may
be evaluated.

## Frozen dev metrics and gates

The strict evaluator uses official MuSiQue answer normalization/alias maximization and
support-set F1. It reports state metrics plus orbit-level minimum C/D F1 and answer/evidence
sufficiency scores that are nonzero only when C and D answer while M refuses. Deltas are
`HopPAIR − CONTROL`. Confidence intervals use 10,000 orbit-paired percentile-bootstrap
resamples with seed 20260814. Base, CONTROL, and HopPAIR all use deterministic decoding with
exactly `max_new_tokens=128`; budget-exhausted rows are recorded. V1 forbids an
output-dependent extension or a 256-token rerun that replaces or selects the frozen result.

Every seed must pass every gate:

| Gate | Frozen rule |
| --- | ---: |
| min(C,D) answer F1 | ≥ +4.00 pp and CI lower bound > 0 |
| orbit answer-sufficiency F1 | ≥ +4.00 pp and CI lower bound > 0 |
| D answer F1 | ≥ +4.00 pp |
| C answer F1 | ≥ −2.00 pp |
| C/D false-refusal rate | ≤ +2.00 pp |
| M refusal rate | delta ≥ 0 and HopPAIR ≥ 80% |
| orbit support-sufficiency F1 | ≥ −2.00 pp |
| strict parse rate | HopPAIR ≥ 99% |
| run integrity | every binding/check true |

Seed 17 failure ends the route. A seed-17 GO permits seeds 29 and 43 under separate exact
versioned receipts, code locks, and output roots; later seeds may not overwrite seed 17.
Before GPU use, the receipt binds the base hash, seed, LoRA and initialization-code hashes,
schedule, and no-resume policy. After both arms train but before either trained arm generates
on dev, their recorded `initial_trainable_parameters_sha256` values must match exactly.
Shadow remains sealed unless all 3/3 seed comparisons are GO. Official dev/test remain
outside this protocol even after shadow success and require a later protocol.

## Literature and novelty boundary

[RAFT (COLM 2024)](https://openreview.net/forum?id=rzQGHXNReU) establishes supervised
open-book adaptation with oracle evidence, distractors, and evidence-citing answers; it does
not establish HopPAIR's paired orbit or objective.

[Trust-Align (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4c88827decab6c046b881a2c3a99c76f-Abstract-Conference.html)
motivates joint attention to answers, citations, and refusal through a preference-alignment
route. This pilot uses neither Trust-Data nor DPO and cannot claim Trust-Score results.

[CORD (NAACL 2025 Short Papers)](https://aclanthology.org/2025.naacl-short.66/) establishes
consistency regularization across position perturbations and warns that blind consistency
may conflict with rank priors. KL consistency itself is therefore not novel here. HopPAIR's
narrow contribution is the controlled combination of support-preserving non-support
replacement, a one-way clean anchor, and an answer-versus-refuse margin on official
multi-hop pairs.

[GRACE (arXiv:2601.04525)](https://arxiv.org/abs/2601.04525) is a 2026 preprint describing an
RL route for evidence grounding and abstention with gated rewards. It is relevant context,
not peer-reviewed authority for this method; this pilot makes no RL, reward, or state-of-the-art
claim.

The strongest allowed project statement is: “a controlled post-training pilot of
support-preserving paired consistency and evidence-sufficiency margin learning.” D is not a
real retriever output or verified adversarial hard negative. M is an official unanswerable
pair passing explicit missing-evidence filters, not proof that parametric memory cannot answer.
Without a separately gated shadow and later official protocol, no official MuSiQue,
state-of-the-art, retriever, production-generalization, preference-training, or RL claim is
allowed.
