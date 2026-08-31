# Support-Orbit MuSiQue pilot v2

## Decision and supersession

V2 is `FROZEN_PRE_GPU`; launch remains `STOP_BEFORE_GPU`. No receipt or GPU run is
authorized by this document.

V1 is permanently withdrawn because its audit mixed incompatible donor-length counts and a
global donor pool could reuse a paragraph that was supporting evidence for another selected
orbit; it also serialized the union of C/M aliases into every state. The immutable withdrawn
v1 protocol SHA256 is
`8c8c9358274890eb8b1539a9e61e1c79b245214d6393004aeaa916cf053b1c8b`; its manifest SHA256
is `9d5e13b893e7c456872111da223487c526ec25cafb9c88eb549cbb1e2cb44ca3`. Neither v1 data nor
its shadow may be used. They remain audit evidence only.

V2 changes data construction and auditing only. The CONTROL/HopPAIR method, hyperparameters,
evaluation metrics, continuation gates, three-seed policy, literature boundary, and claim
boundary are unchanged.

## V2 data lock

The canonical manifest is `prepared_data_v2/manifest.json`, schema
`support-orbit-musique/v2`, builder 2.0.0, SHA256
`84e97c0eb189d664149822c1244436f0b97e69e3cbca7e16be8186df52a7dfc1`.

| Artifact | SHA256 |
| --- | --- |
| train.jsonl | `641bae95c4eb410229fb8c87a52b24e58628c376c636632d242115ca7e4d5c12` |
| dev.jsonl | `84acfa99def02660aa74d836f07ff4219c0c838c29e19d7d335a126c7840b909` |
| shadow writer digest | `1eed3236bb17e79518ae5c3787f3171a37eaea1bfec8f6f46766a59ce2e1d293` |
| audit.json | `462acaada470b7b49b5fbe74c990810714b6a8b9bcaa1b0cd6b97b801a957a6e` |
| SHADOW_SEALED.json | `5e6288ef3d7e35ebd9c6c5d3812895f0a812106913b5fb357ae4e96052af5bd5` |

The split remains 1,920/400/400 orbits and 5,760/1,200/1,200 state records. The validator
recomputed train/dev and public audit hashes, but never opened the shadow body. Its digest is
bound only through the builder's streaming writer, manifest, checksum sidecar, and sealed
marker.

All 2,720 orbits pass exact two-slot D replacement, C-support byte preservation, C/D target
identity, same-split donors, global exclusion of every selected C-support text from donor
use, and state-specific official alias serialization. Donor length-ratio maximum is 1.25
with zero violations.

Surface separability AUC is 0.53968806/0.55929697/0.56091353 for C–D/D–M/C–M, all below
0.60. Semantic C–D is 0.54311794, below 0.60; semantic D–M/C–M are diagnostic at
0.63300405/0.63463566. The tokenizer audit covers 6,960 train+dev records, maximum
4,974/6,144 tokens, zero over-limit rows, zero prepared/canonical mismatches, and zero
truncation.

## Unchanged method and schedule

Both arms see identical atomic `[C,D,M]` groups in one forward.

- CONTROL: `L = L_sft`.
- HopPAIR:
  `L = L_sft + 0.1 KL(stopgrad(p_C)||p_D) + 0.2 × 0.5[softplus(2−(z_C−z_M)) + softplus(2−(z_D−z_M))]`,
  where `z = logit(S)−logit(U)` at the first assistant-token prediction.

SFT is state-balanced completion-token CE. The C→D KL is token-mean over shared completion
ordinals. Both arms use Qwen3-4B-Instruct-2507, LoRA r16/alpha32/dropout0.05 on all
transformer linear projections, BF16, SDPA, gradient checkpointing, fused AdamW, learning
rate 2e-4, cosine scheduling, eight warmup steps, no resume, one orbit per microbatch, eight
orbit gradient accumulation, one exact epoch, and 240 optimizer steps. Only the final step is
eligible.

Seed 17 runs first. Only a frozen dev GO permits separately locked seeds 29 and 43. V2
shadow stays sealed until all 3/3 dev comparisons are GO. Official MuSiQue dev/test are
unextracted, unread, outside this protocol, and cannot be opened by v2.

## Unchanged evaluation gates

Generation is deterministic with `max_new_tokens=128`. Deltas are HopPAIR−CONTROL; paired
bootstrap uses 10,000 orbit resamples and seed 20260814. Every gate is required per seed:

| Gate | Rule |
| --- | ---: |
| min(C,D) answer F1 | ≥ +4.00 pp and CI lower bound > 0 |
| orbit answer-sufficiency F1 | ≥ +4.00 pp and CI lower bound > 0 |
| D answer F1 | ≥ +4.00 pp |
| C answer F1 | ≥ −2.00 pp |
| C/D false-refusal rate | ≤ +2.00 pp |
| M refusal rate | delta ≥ 0 and HopPAIR ≥ 80% |
| orbit support-sufficiency F1 | ≥ −2.00 pp |
| strict parse rate | HopPAIR ≥ 99% |
| run integrity | every binding and runtime check true |

Any failure produces `STOP_NO_SHADOW_NO_OFFICIAL_NO_DPO`.

## Access and claim boundary

After a separately approved v2 launch receipt, training may read only
`prepared_data_v2/train.jsonl`. Dev may first be read only after both arms finish, their
240-step finite ledgers validate, and their recorded initial-trainable-parameter hashes match.
Prepared v1, v2 shadow, official dev/test, and network-fetched evaluation data are forbidden.

The literature and novelty boundary remains the v1 boundary covering RAFT (COLM 2024),
Trust-Align (ICLR 2025), CORD (NAACL 2025 Short Papers), and GRACE
(arXiv:2601.04525, preprint). The allowed description is a controlled post-training pilot of
support-preserving paired consistency and evidence-sufficiency margin learning. V2 cannot
claim official MuSiQue improvement, state of the art, retriever gains, RL/preference training,
or production generalization.
