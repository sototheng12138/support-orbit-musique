"""Single-forward training core and fail-closed finite guards."""

from __future__ import annotations

from typing import Any

import torch

from .losses import Arm, LossBreakdown, compute_objective


def _extract_logits(outputs: Any) -> torch.Tensor:
    logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model output must expose a logits tensor")
    return logits


def one_forward_objective(
    model: Any,
    batch: dict[str, Any],
    *,
    arm: Arm,
) -> LossBreakdown:
    """Run C/D/M together once, then apply the manual arm objective."""

    required = ("input_ids", "attention_mask", "logits_to_keep")
    if any(name not in batch for name in required):
        raise ValueError("batch lacks model-forward tensors")
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    keep = batch["logits_to_keep"].to(device=input_ids.device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=keep,
        use_cache=False,
        return_dict=True,
    )
    logits = _extract_logits(outputs)
    return compute_objective(logits, batch, arm=arm)


def assert_finite_gradients(model: Any) -> None:
    invalid = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad.detach()).all().item())
    ]
    if invalid:
        raise FloatingPointError(f"non-finite gradients: {invalid[:8]}")


def assert_finite_trainable_parameters(model: Any) -> None:
    invalid = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not bool(torch.isfinite(parameter.detach()).all().item())
    ]
    if invalid:
        raise FloatingPointError(f"non-finite trainable parameters: {invalid[:8]}")


def assert_finite_optimizer_state(optimizer: Any) -> None:
    invalid: list[str] = []
    for parameter_index, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and not bool(
                torch.isfinite(value.detach()).all().item()
            ):
                invalid.append(f"parameter[{parameter_index}].{key}")
    if invalid:
        raise FloatingPointError(f"non-finite optimizer state: {invalid[:8]}")
