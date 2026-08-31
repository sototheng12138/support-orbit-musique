"""Manual FP32 objectives for CONTROL and HopPAIR training arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F


Arm = Literal["CONTROL", "HopPAIR"]
STATE_NAMES = ("C", "D", "M")
S_TOKEN_ID = 50
U_TOKEN_ID = 52
KL_WEIGHT = 0.1
FLIP_WEIGHT = 0.2
FLIP_MARGIN = 2.0


@dataclass(frozen=True)
class LossBreakdown:
    loss: torch.Tensor
    sft: torch.Tensor
    kl: torch.Tensor
    flip: torch.Tensor
    state_ce: dict[str, torch.Tensor]
    state_token_counts: dict[str, int]


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise FloatingPointError(f"non-finite {name}")


def _validate_batch(logits: torch.Tensor, batch: dict[str, Any]) -> None:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [rows, kept_positions, vocabulary]")
    required = {
        "target_ids",
        "target_mask",
        "prediction_union_indices",
        "state_ids",
        "logits_to_keep",
    }
    missing = required - set(batch)
    if missing:
        raise ValueError(f"objective batch is missing {sorted(missing)}")
    rows, kept, vocabulary = logits.shape
    if rows == 0 or rows % 3 != 0:
        raise ValueError("rows must be a non-empty orbit-major multiple of three")
    if kept != int(batch["logits_to_keep"].numel()):
        raise ValueError("returned logits do not match logits_to_keep")
    if vocabulary <= max(S_TOKEN_ID, U_TOKEN_ID):
        raise ValueError("vocabulary does not contain frozen S/U action IDs")
    targets = batch["target_ids"]
    mask = batch["target_mask"]
    mapping = batch["prediction_union_indices"]
    if targets.shape != mask.shape or mapping.shape != mask.shape or targets.shape[0] != rows:
        raise ValueError("target tensors have incompatible shapes")
    if batch["state_ids"].tolist() != [0, 1, 2] * (rows // 3):
        raise ValueError("state rows are not orbit-major C/D/M")
    if not bool(mask[:, 0].all().item()):
        raise ValueError("every state must expose a first action prediction")
    first_targets = targets[:, 0].tolist()
    if first_targets != [S_TOKEN_ID, S_TOKEN_ID, U_TOKEN_ID] * (rows // 3):
        raise ValueError("first target IDs must be frozen S/S/U actions for C/D/M")
    if bool((mapping[mask] < 0).any().item()) or bool((mapping[mask] >= kept).any().item()):
        raise ValueError("prediction-union index is out of bounds")
    _require_finite("kept logits", logits)


def _row_logits(
    logits: torch.Tensor,
    batch: dict[str, Any],
    row: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = batch["target_mask"][row]
    indices = batch["prediction_union_indices"][row, valid]
    targets = batch["target_ids"][row, valid]
    return logits[row, indices].float(), targets


def token_normalized_sft(
    logits: torch.Tensor,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int]]:
    """Mean CE within each state, then an unweighted mean over C/D/M."""

    _validate_batch(logits, batch)
    state_sums = [logits.new_zeros((), dtype=torch.float32) for _ in STATE_NAMES]
    state_counts = [0, 0, 0]
    for row in range(logits.shape[0]):
        row_logits, targets = _row_logits(logits, batch, row)
        state = int(batch["state_ids"][row])
        state_sums[state] = state_sums[state] + F.cross_entropy(
            row_logits, targets, reduction="sum"
        )
        state_counts[state] += int(targets.numel())
    if any(count <= 0 for count in state_counts):
        raise ValueError("each state must contain supervised completion tokens")
    state_ce = {
        name: state_sums[index] / state_counts[index]
        for index, name in enumerate(STATE_NAMES)
    }
    sft = torch.stack(tuple(state_ce.values())).mean()
    _require_finite("SFT loss", sft)
    return sft, state_ce, dict(zip(STATE_NAMES, state_counts, strict=True))


def anchored_completion_kl(logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    """Token-mean KL(p_C.stopgrad || p_D) on shared completion ordinals."""

    _validate_batch(logits, batch)
    total = logits.new_zeros((), dtype=torch.float32)
    tokens = 0
    for base in range(0, logits.shape[0], 3):
        complete_logits, complete_targets = _row_logits(logits, batch, base)
        distractor_logits, distractor_targets = _row_logits(logits, batch, base + 1)
        if not torch.equal(complete_targets, distractor_targets):
            raise ValueError("C/D completion ordinals or token IDs differ")
        teacher = F.softmax(complete_logits.detach(), dim=-1)
        student_log = F.log_softmax(distractor_logits, dim=-1)
        total = total + F.kl_div(student_log, teacher, reduction="sum")
        tokens += int(complete_targets.numel())
    if tokens <= 0:
        raise ValueError("KL has no shared completion tokens")
    value = total / tokens
    _require_finite("anchored KL", value)
    return value


def action_flip_loss(logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    """Encourage C and D to flip from U toward S relative to matched M.

    For each state ``z = logit(S) - logit(U)`` at the first assistant-token
    prediction.  Each orbit contributes
    ``.5 * [softplus(2-(z_C-z_M)) + softplus(2-(z_D-z_M))]``.
    """

    _validate_batch(logits, batch)
    terms: list[torch.Tensor] = []
    mapping = batch["prediction_union_indices"]
    for base in range(0, logits.shape[0], 3):
        complete_index = mapping[base, 0]
        distractor_index = mapping[base + 1, 0]
        missing_index = mapping[base + 2, 0]
        complete = logits[base, complete_index].float()
        distractor = logits[base + 1, distractor_index].float()
        missing = logits[base + 2, missing_index].float()
        z_c = complete[S_TOKEN_ID] - complete[U_TOKEN_ID]
        z_d = distractor[S_TOKEN_ID] - distractor[U_TOKEN_ID]
        z_m = missing[S_TOKEN_ID] - missing[U_TOKEN_ID]
        margin = logits.new_tensor(FLIP_MARGIN, dtype=torch.float32)
        terms.append(
            0.5
            * (
                F.softplus(margin - (z_c - z_m))
                + F.softplus(margin - (z_d - z_m))
            )
        )
    value = torch.stack(terms).mean()
    _require_finite("action-flip loss", value)
    return value


def compute_objective(
    logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    arm: Arm,
) -> LossBreakdown:
    """Compute the frozen arm objective without using model-native loss."""

    if arm not in ("CONTROL", "HopPAIR"):
        raise ValueError("arm must be CONTROL or HopPAIR")
    sft, state_ce, counts = token_normalized_sft(logits, batch)
    if arm == "CONTROL":
        zero = sft.new_zeros(())
        loss = sft
        kl = zero
        flip = zero
    else:
        kl = anchored_completion_kl(logits, batch)
        flip = action_flip_loss(logits, batch)
        loss = sft + KL_WEIGHT * kl + FLIP_WEIGHT * flip
    _require_finite("total objective", loss)
    return LossBreakdown(
        loss=loss,
        sft=sft,
        kl=kl,
        flip=flip,
        state_ce=state_ce,
        state_token_counts=counts,
    )
