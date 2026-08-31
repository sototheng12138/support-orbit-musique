"""Train one frozen Support-Orbit CONTROL or HopPAIR arm on one BF16 GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backend import (
    CANONICAL_MODEL_PATH,
    CANONICAL_RECEIPT,
    LoraSpec,
    artifact_manifest,
    assert_no_symlink_components,
    atomic_json,
    build_lora_model,
    ensure_bf16_single_cuda,
    exact_absolute_path,
    load_tokenizer,
    optimizer_state_sha256,
    set_seed,
    sha256_file,
    trainable_parameter_summary,
    trainable_parameters_sha256,
    validate_launch_receipt,
    validate_static_model_identity,
    verified_jsonl_objects,
)
from .losses import FLIP_MARGIN, FLIP_WEIGHT, KL_WEIGHT
from .sft import ACTION_TOKEN_IDS, OrbitMajorCollator, PreparedOrbitDataset
from .trainer_core import (
    assert_finite_gradients,
    assert_finite_optimizer_state,
    assert_finite_trainable_parameters,
    one_forward_objective,
)


RUN_SCHEMA = "support-orbit-musique.train-run.v1"
TRAIN_ORBITS = 1_920
TRAIN_RECORDS = TRAIN_ORBITS * 3
OPTIMIZER_STEPS = 240
GRADIENT_ACCUMULATION = 8
SEED = 17
MAX_LENGTH = 6_144
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
WARMUP_STEPS = 8
MAX_GRAD_NORM = 1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def frozen_orbit_schedule(orbit_ids: Sequence[str], *, seed: int = SEED) -> tuple[str, ...]:
    """Return the cross-platform, arm-independent one-epoch orbit order."""

    if len(orbit_ids) != TRAIN_ORBITS or len(set(orbit_ids)) != TRAIN_ORBITS:
        raise ValueError(f"schedule requires {TRAIN_ORBITS} unique orbit IDs")
    if any(not isinstance(value, str) or not value for value in orbit_ids):
        raise ValueError("schedule orbit IDs must be nonempty strings")
    return tuple(
        sorted(
            orbit_ids,
            key=lambda value: (
                hashlib.sha256(f"{seed}\n{value}".encode()).hexdigest(),
                value,
            ),
        )
    )


def schedule_sha256(schedule: Sequence[str]) -> str:
    if len(schedule) != TRAIN_ORBITS or len(set(schedule)) != TRAIN_ORBITS:
        raise ValueError("cannot hash an incomplete or duplicate orbit schedule")
    return hashlib.sha256("".join(f"{value}\n" for value in schedule).encode()).hexdigest()


def expected_learning_rate(scheduler_step: int) -> float:
    """Frozen HF cosine-with-warmup LR at scheduler state ``scheduler_step``."""

    if not 0 <= scheduler_step <= OPTIMIZER_STEPS:
        raise ValueError("scheduler_step is outside the frozen 0..240 range")
    if scheduler_step < WARMUP_STEPS:
        factor = scheduler_step / max(1, WARMUP_STEPS)
    else:
        progress = (scheduler_step - WARMUP_STEPS) / max(
            1, OPTIMIZER_STEPS - WARMUP_STEPS
        )
        factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LEARNING_RATE * factor


def assert_all_trainable_gradients_present(model: Any) -> None:
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise RuntimeError(f"trainable parameters without gradients: {missing[:8]}")


def _protected_path(path: Path) -> bool:
    lowered = str(path).casefold().replace("\\", "/")
    forbidden = (
        "prepared_data_v1",
        "shadow",
        "official_dev",
        "official-dev",
        "official_test",
        "official-test",
        "musique_full_v1.0_dev",
        "musique_full_v1.0_test",
    )
    return any(token in lowered for token in forbidden)


def load_train_records(
    path: str | Path,
    *,
    expected_orbits: int = TRAIN_ORBITS,
    expected_sha256: str | None = None,
) -> list[dict[str, Any]]:
    source = exact_absolute_path(path, description="training data path")
    if _protected_path(source):
        raise ValueError(f"training refuses a protected path before open: {source}")
    expected_records = expected_orbits * 3
    records = verified_jsonl_objects(source, expected_sha256=expected_sha256)
    seen_ids: set[str] = set()
    for line_number, value in enumerate(records, 1):
        record_id = value.get("id")
        if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
            raise ValueError(f"{source}:{line_number}: invalid or duplicate id")
        seen_ids.add(record_id)
    if len(records) != expected_records:
        raise ValueError(f"training requires {expected_records} records, found {len(records)}")
    orbit_ids: set[str] = set()
    for start in range(0, len(records), 3):
        group = records[start : start + 3]
        states = tuple(record.get("state") for record in group)
        ids = tuple(record.get("orbit_id") for record in group)
        if states != ("C", "D", "M") or not isinstance(ids[0], str) or len(set(ids)) != 1:
            raise ValueError(f"records {start}:{start + 3} are not one canonical C/D/M orbit")
        if ids[0] in orbit_ids:
            raise ValueError(f"duplicate train orbit: {ids[0]}")
        orbit_ids.add(ids[0])
        for record in group:
            if (
                record.get("split") != "train"
                or record.get("training_read_allowed") is not True
                or record.get("sealed") is not False
            ):
                raise ValueError(f"{ids[0]} contains a non-trainable or sealed record")
    if len(orbit_ids) != expected_orbits:
        raise ValueError(f"training requires {expected_orbits} exact C/D/M orbits")
    return records


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("CONTROL", "HopPAIR"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--launch-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def parser() -> argparse.ArgumentParser:
    return _argument_parser()


def _validate_cli_bindings(args: argparse.Namespace, receipt: dict[str, Any]) -> tuple[Path, Path]:
    bindings = receipt["exact_bindings"]
    supplied = {
        "model": exact_absolute_path(args.model_path, description="--model-path"),
        "train": exact_absolute_path(args.train_file, description="--train-file"),
        "manifest": exact_absolute_path(
            args.prepared_manifest, description="--prepared-manifest"
        ),
        "protocol": exact_absolute_path(args.protocol, description="--protocol"),
        "receipt": exact_absolute_path(
            args.launch_receipt, description="--launch-receipt"
        ),
        "output": exact_absolute_path(args.output_dir, description="--output-dir"),
    }
    expected_output_key = "control_train" if args.arm == "CONTROL" else "hoppair_train"
    expected = {
        "model": exact_absolute_path(bindings["base_model"]["path"]),
        "train": exact_absolute_path(bindings["train"]["path"]),
        "manifest": exact_absolute_path(bindings["prepared_manifest"]["path"]),
        "protocol": exact_absolute_path(bindings["protocol"]["path"]),
        "receipt": CANONICAL_RECEIPT,
        "output": exact_absolute_path(receipt["output_roots"][expected_output_key]),
    }
    if supplied != expected:
        mismatch = {key: (str(supplied[key]), str(expected[key])) for key in supplied if supplied[key] != expected[key]}
        raise ValueError(f"CLI paths differ from launch receipt: {mismatch}")
    if supplied["model"] != CANONICAL_MODEL_PATH:
        raise ValueError("training model is not the canonical Qwen3-4B path")
    return supplied["train"], supplied["output"]


class _ForwardCapture:
    """Capture the logits from trainer_core without performing a second forward."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.logits: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self.model(*args, **kwargs)
        self.logits = output.get("logits") if isinstance(output, Mapping) else output.logits
        return output


def _action_margins(logits: Any, batch: dict[str, Any]) -> dict[str, float]:
    import torch

    if not isinstance(logits, torch.Tensor) or logits.shape[0] != 3:
        raise ValueError("margin logging requires one three-row C/D/M orbit")
    mapping = batch["prediction_union_indices"]
    values: list[torch.Tensor] = []
    for row in range(3):
        action_logits = logits[row, mapping[row, 0]].detach().float()
        values.append(action_logits[ACTION_TOKEN_IDS["S"]] - action_logits[ACTION_TOKEN_IDS["U"]])
    z_c, z_d, z_m = (float(value.item()) for value in values)
    result = {
        "z_c": z_c,
        "z_d": z_d,
        "z_m": z_m,
        "c_minus_m": z_c - z_m,
        "d_minus_m": z_d - z_m,
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"non-finite action margin: {result}")
    return result


def _to_cuda(batch: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {
        key: value.to("cuda:0", non_blocking=False) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _mean_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = rows[0]
    result = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"non-finite accumulated metrics: {result}")
    return result


def _write_ledger_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _revalidate_training_identities(
    receipt: dict[str, Any],
    model_path: str | Path,
    train_path: Path,
) -> dict[str, Any]:
    refreshed = validate_launch_receipt(CANONICAL_RECEIPT, purpose="metadata")
    if refreshed["_validated"]["sha256"] != receipt["_validated"]["sha256"]:
        raise ValueError("launch receipt changed during training")
    identity = validate_static_model_identity(refreshed, model_path)
    if sha256_file(train_path) != receipt["exact_bindings"]["train"]["sha256"]:
        raise ValueError("train artifact changed during training")
    return identity


def _run(args: argparse.Namespace) -> int:
    if (KL_WEIGHT, FLIP_WEIGHT, FLIP_MARGIN) != (0.1, 0.2, 2.0):
        raise ValueError("compiled HopPAIR objective constants drifted from the frozen receipt")
    receipt = validate_launch_receipt(args.launch_receipt, purpose="train")
    train_path, output = _validate_cli_bindings(args, receipt)
    static_identity = validate_static_model_identity(receipt, args.model_path)
    records = load_train_records(
        train_path,
        expected_sha256=receipt["exact_bindings"]["train"]["sha256"],
    )
    raw_orbit_ids = [str(records[start]["orbit_id"]) for start in range(0, len(records), 3)]
    schedule = frozen_orbit_schedule(raw_orbit_ids)
    actual_schedule_sha = schedule_sha256(schedule)
    if actual_schedule_sha != receipt["exact_bindings"]["schedule_sha256"]:
        raise ValueError("runtime orbit schedule differs from the launch receipt")

    tokenizer = load_tokenizer(args.model_path, receipt)
    dataset = PreparedOrbitDataset(records, tokenizer, max_length=MAX_LENGTH)
    if len(dataset) != TRAIN_ORBITS or dataset.stats["truncated"] != 0:
        raise ValueError("encoded dataset violates the frozen orbit/zero-truncation contract")
    index_by_id = {dataset[index].orbit_id: index for index in range(len(dataset))}
    if set(index_by_id) != set(schedule):
        raise ValueError("encoded orbit IDs differ from the frozen schedule")
    ordered = tuple(dataset[index_by_id[orbit_id]] for orbit_id in schedule)

    # Do not consume the one-shot canonical output root unless the frozen UUID
    # is still the intended, idle physical device immediately before CUDA use.
    hardware_identity = ensure_bf16_single_cuda(receipt, args.arm)
    assert_no_symlink_components(
        output,
        description="training output root before creation",
        allow_missing_tail=True,
    )
    try:
        output.mkdir(parents=True)
    except FileExistsError as exc:
        raise FileExistsError(f"no-resume run refuses existing output directory: {output}") from exc
    assert_no_symlink_components(output, description="training output root after creation")
    manifest_path = output / "run_manifest.json"
    ledger_path = output / "step_ledger.jsonl"
    started_at = utc_now()
    immutable = {
        "arm": args.arm,
        "receipt": receipt["_validated"],
        "protocol": receipt["exact_bindings"]["protocol"],
        "prepared_manifest": receipt["exact_bindings"]["prepared_manifest"],
        "train": receipt["exact_bindings"]["train"],
        "schedule": {
            "algorithm": receipt["runtime"]["schedule_algorithm"],
            "sha256": actual_schedule_sha,
            "orbits": len(schedule),
        },
        "static_identity": static_identity,
        "tokenizer_semantics": receipt["exact_bindings"]["tokenizer"],
        "runtime": receipt["runtime"],
        "lora": LoraSpec().to_dict(),
        "objective": receipt["objectives"][args.arm],
        "dataset_stats": dataset.stats,
    }
    atomic_json(
        manifest_path,
        {
            "schema_version": RUN_SCHEMA,
            "status": "running",
            "started_at": started_at,
            "immutable": immutable,
        },
    )

    try:
        import torch
        from transformers import get_cosine_schedule_with_warmup

        set_seed(SEED)
        pre_load_identity = _revalidate_training_identities(
            receipt, args.model_path, train_path
        )
        if pre_load_identity != static_identity:
            raise ValueError("model/tokenizer identity changed before model load")
        model = build_lora_model(args.model_path, LoraSpec())
        model.to("cuda:0")
        model.train()
        post_load_identity = _revalidate_training_identities(receipt, args.model_path, train_path)
        if post_load_identity != static_identity:
            raise ValueError("model/tokenizer identity changed across model load")
        initial_parameters_sha = trainable_parameters_sha256(model)
        parameter_summary = trainable_parameter_summary(model)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            fused=True,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=OPTIMIZER_STEPS,
        )
        collator = OrbitMajorCollator(tokenizer.pad_token_id, pad_to_multiple_of=8)
        capture = _ForwardCapture(model)
        global_step = 0
        consumed = 0
        start_time = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with ledger_path.open("x", encoding="utf-8") as ledger:
            for step_start in range(0, TRAIN_ORBITS, GRADIENT_ACCUMULATION):
                micro_rows: list[dict[str, float]] = []
                orbit_batch = schedule[step_start : step_start + GRADIENT_ACCUMULATION]
                applied_lr = float(optimizer.param_groups[0]["lr"])
                if not math.isclose(
                    applied_lr,
                    expected_learning_rate(global_step),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError("optimizer LR differs from the frozen cosine schedule")
                step_clock = time.monotonic()
                for orbit in ordered[step_start : step_start + GRADIENT_ACCUMULATION]:
                    batch = _to_cuda(collator([orbit]))
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        breakdown = one_forward_objective(capture, batch, arm=args.arm)
                    margins = _action_margins(capture.logits, batch)
                    values = {
                        "loss": float(breakdown.loss.detach().item()),
                        "sft": float(breakdown.sft.detach().item()),
                        "kl": float(breakdown.kl.detach().item()),
                        "flip": float(breakdown.flip.detach().item()),
                        "ce_c": float(breakdown.state_ce["C"].detach().item()),
                        "ce_d": float(breakdown.state_ce["D"].detach().item()),
                        "ce_m": float(breakdown.state_ce["M"].detach().item()),
                        **margins,
                    }
                    if args.arm == "CONTROL" and (values["kl"] != 0.0 or values["flip"] != 0.0):
                        raise AssertionError("CONTROL unexpectedly received auxiliary loss")
                    (breakdown.loss / GRADIENT_ACCUMULATION).backward()
                    micro_rows.append(values)
                    capture.logits = None
                    consumed += 1
                assert_all_trainable_gradients_present(model)
                assert_finite_gradients(model)
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    MAX_GRAD_NORM,
                    error_if_nonfinite=True,
                )
                grad_norm = float(grad_norm_tensor.detach().item())
                if not math.isfinite(grad_norm):
                    raise FloatingPointError("non-finite pre-clip gradient norm")
                assert_finite_gradients(model)
                optimizer.step()
                scheduler.step()
                next_lr = float(optimizer.param_groups[0]["lr"])
                if not math.isclose(
                    next_lr,
                    expected_learning_rate(global_step + 1),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError("scheduler next LR differs from the frozen schedule")
                optimizer.zero_grad(set_to_none=True)
                assert_finite_trainable_parameters(model)
                assert_finite_optimizer_state(optimizer)
                global_step += 1
                aggregate = _mean_metrics(micro_rows)
                _write_ledger_row(
                    ledger,
                    {
                        "optimizer_step": global_step,
                        "microsteps": GRADIENT_ACCUMULATION,
                        "consumed_orbits": consumed,
                        "orbit_batch_sha256": hashlib.sha256(
                            "".join(f"{value}\n" for value in orbit_batch).encode()
                        ).hexdigest(),
                        "learning_rate_applied": applied_lr,
                        "learning_rate_next": next_lr,
                        "grad_norm_preclip": grad_norm,
                        "max_grad_norm": MAX_GRAD_NORM,
                        "metrics": aggregate,
                        "finite_loss": True,
                        "finite_gradients": True,
                        "all_trainable_gradients_present": True,
                        "finite_parameters": True,
                        "finite_optimizer_state": True,
                        "step_seconds": time.monotonic() - step_clock,
                    },
                )
        if global_step != OPTIMIZER_STEPS or consumed != TRAIN_ORBITS:
            raise RuntimeError(
                f"training stopped at step/orbit {global_step}/{consumed}, expected "
                f"{OPTIMIZER_STEPS}/{TRAIN_ORBITS}"
            )
        post_training_identity = _revalidate_training_identities(
            receipt, args.model_path, train_path
        )
        if post_training_identity != static_identity:
            raise ValueError("model/tokenizer identity changed during training")
        final_parameters_sha = trainable_parameters_sha256(model)
        final_optimizer_sha = optimizer_state_sha256(optimizer, model)
        adapter_dir = output / "final_adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        checkpoint = artifact_manifest(adapter_dir, require_adapter=True)
        step_ledger = {
            "path": str(ledger_path),
            "sha256": sha256_file(ledger_path),
            "rows": OPTIMIZER_STEPS,
        }
        final = {
            "schema_version": RUN_SCHEMA,
            "status": "completed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "elapsed_seconds": time.monotonic() - start_time,
            "immutable": immutable,
            "parameters": parameter_summary,
            "hardware_identity": hardware_identity,
            "initial_trainable_parameters_sha256": initial_parameters_sha,
            "final_trainable_parameters_sha256": final_parameters_sha,
            "final_optimizer_state_sha256": final_optimizer_sha,
            "global_step": global_step,
            "consumed_orbits": consumed,
            "epochs_completed": 1,
            "optimizer_steps_skipped": 0,
            "step_ledger": step_ledger,
            "checkpoint": checkpoint,
        }
        atomic_json(manifest_path, final)
        return 0
    except BaseException as exc:
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            if current.get("status") == "running":
                current.update(
                    {
                        "status": "failed",
                        "failed_at": utc_now(),
                        "failure": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
                atomic_json(manifest_path, current)
        except BaseException:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GRADIENT_ACCUMULATION",
    "MAX_LENGTH",
    "OPTIMIZER_STEPS",
    "RUN_SCHEMA",
    "SEED",
    "TRAIN_ORBITS",
    "assert_all_trainable_gradients_present",
    "expected_learning_rate",
    "frozen_orbit_schedule",
    "load_train_records",
    "main",
    "parser",
    "schedule_sha256",
]
