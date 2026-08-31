"""Greedy, receipt-bound generation for the frozen 400-orbit development set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backend import (
    CANONICAL_RECEIPT,
    artifact_manifest,
    assert_no_symlink_components,
    atomic_json,
    ensure_bf16_single_cuda,
    exact_absolute_path,
    load_tokenizer,
    read_json_object,
    sha256_file,
    validate_launch_receipt,
    validate_static_model_identity,
    verified_json_object,
    verified_jsonl_objects,
)
from .formatting import render_prepared_record
from .train import (
    GRADIENT_ACCUMULATION,
    OPTIMIZER_STEPS,
    RUN_SCHEMA,
    TRAIN_ORBITS,
    expected_learning_rate,
    frozen_orbit_schedule,
    load_train_records,
    schedule_sha256,
)


GENERATION_SCHEMA = "support-orbit-musique.generation.v1"
DEV_ORBITS = 400
DEV_RECORDS = DEV_ORBITS * 3


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


def load_dev_records(
    path: str | Path,
    *,
    expected_orbits: int = DEV_ORBITS,
    expected_sha256: str | None = None,
) -> list[dict[str, Any]]:
    source = exact_absolute_path(path, description="development data path")
    if _protected_path(source):
        raise ValueError(f"generation refuses a protected path before open: {source}")
    records = verified_jsonl_objects(source, expected_sha256=expected_sha256)
    seen: set[str] = set()
    for line_number, value in enumerate(records, 1):
        record_id = value.get("id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError(f"{source}:{line_number}: invalid or duplicate id")
        seen.add(record_id)
    if len(records) != expected_orbits * 3:
        raise ValueError(
            f"development generation requires {expected_orbits * 3} records, found {len(records)}"
        )
    state_counts = Counter(record.get("state") for record in records)
    if state_counts != {"C": expected_orbits, "D": expected_orbits, "M": expected_orbits}:
        raise ValueError(f"development state counts drifted: {state_counts}")
    seen_orbits: set[str] = set()
    for start in range(0, len(records), 3):
        group = records[start : start + 3]
        states = tuple(record.get("state") for record in group)
        orbit_ids = tuple(record.get("orbit_id") for record in group)
        if states != ("C", "D", "M") or not isinstance(orbit_ids[0], str) or len(set(orbit_ids)) != 1:
            raise ValueError(f"records {start}:{start + 3} are not one canonical C/D/M orbit")
        if orbit_ids[0] in seen_orbits:
            raise ValueError(f"duplicate development orbit: {orbit_ids[0]}")
        seen_orbits.add(orbit_ids[0])
        if any(
            record.get("split") != "dev"
            or record.get("training_read_allowed") is not True
            or record.get("sealed") is not False
            for record in group
        ):
            raise ValueError(f"{orbit_ids[0]} contains a sealed or non-development record")
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("BASE", "CONTROL", "HopPAIR"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--control-run-manifest", required=True)
    parser.add_argument("--hoppair-run-manifest", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--launch-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int, choices=(128,))
    return parser


def parser() -> argparse.ArgumentParser:
    return _parser()


def _validate_cli(args: argparse.Namespace, receipt: dict[str, Any]) -> tuple[Path, Path]:
    bindings = receipt["exact_bindings"]
    input_path = exact_absolute_path(args.input_file, description="--input-file")
    output = exact_absolute_path(args.output_dir, description="--output-dir")
    output_key = {
        "BASE": "base_dev_generation",
        "CONTROL": "control_dev_generation",
        "HopPAIR": "hoppair_dev_generation",
    }[args.arm]
    supplied = {
        "model": exact_absolute_path(args.model_path, description="--model-path"),
        "input": input_path,
        "manifest": exact_absolute_path(
            args.prepared_manifest, description="--prepared-manifest"
        ),
        "protocol": exact_absolute_path(args.protocol, description="--protocol"),
        "receipt": exact_absolute_path(
            args.launch_receipt, description="--launch-receipt"
        ),
        "output": output,
    }
    expected = {
        "model": exact_absolute_path(bindings["base_model"]["path"]),
        "input": exact_absolute_path(bindings["dev"]["path"]),
        "manifest": exact_absolute_path(bindings["prepared_manifest"]["path"]),
        "protocol": exact_absolute_path(bindings["protocol"]["path"]),
        "receipt": CANONICAL_RECEIPT,
        "output": exact_absolute_path(receipt["output_roots"][output_key]),
    }
    mismatches = [
        (name, str(supplied[name]), str(expected[name]))
        for name in supplied
        if supplied[name] != expected[name]
    ]
    if mismatches:
        raise ValueError(f"generation CLI paths differ from launch receipt: {mismatches}")
    generation = receipt["runtime"]["generation"]
    if args.batch_size is not None and args.batch_size != generation["batch_size"]:
        raise ValueError("--batch-size differs from the frozen receipt")
    if args.max_new_tokens is not None and args.max_new_tokens != generation["max_new_tokens"]:
        raise ValueError("--max-new-tokens differs from the frozen receipt")
    return input_path, output


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_step_ledger(
    path: Path,
    *,
    expected_arm: str | None = None,
    expected_schedule: tuple[str, ...] | None = None,
    expected_sha256: str | None = None,
) -> list[dict[str, Any]]:
    if expected_schedule is not None and (
        len(expected_schedule) != TRAIN_ORBITS
        or len(set(expected_schedule)) != TRAIN_ORBITS
    ):
        raise ValueError("step-ledger verification requires the exact 1920-orbit schedule")
    rows = verified_jsonl_objects(path, expected_sha256=expected_sha256)
    if len(rows) != OPTIMIZER_STEPS:
        raise ValueError(f"step ledger has {len(rows)} rows, expected {OPTIMIZER_STEPS}")
    required_metrics = {
        "loss",
        "sft",
        "kl",
        "flip",
        "ce_c",
        "ce_d",
        "ce_m",
        "z_c",
        "z_d",
        "z_m",
        "c_minus_m",
        "d_minus_m",
    }
    for expected_step, row in enumerate(rows, 1):
        metrics = row.get("metrics")
        schedule_start = (expected_step - 1) * GRADIENT_ACCUMULATION
        expected_batch_sha = (
            hashlib.sha256(
                "".join(
                    f"{value}\n"
                    for value in expected_schedule[
                        schedule_start : schedule_start + GRADIENT_ACCUMULATION
                    ]
                ).encode()
            ).hexdigest()
            if expected_schedule is not None
            else None
        )
        numeric = [
            row.get("learning_rate_applied"),
            row.get("learning_rate_next"),
            row.get("grad_norm_preclip"),
            row.get("step_seconds"),
            *(metrics.values() if isinstance(metrics, dict) else []),
        ]
        if (
            row.get("optimizer_step") != expected_step
            or row.get("microsteps") != 8
            or row.get("consumed_orbits") != expected_step * 8
            or not _is_sha256(row.get("orbit_batch_sha256"))
            or (
                expected_batch_sha is not None
                and row.get("orbit_batch_sha256") != expected_batch_sha
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in numeric
            )
            or not math.isclose(
                float(row.get("learning_rate_applied", float("nan"))),
                expected_learning_rate(expected_step - 1),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(row.get("learning_rate_next", float("nan"))),
                expected_learning_rate(expected_step),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or row.get("max_grad_norm") != 1.0
            or not isinstance(metrics, dict)
            or set(metrics) != required_metrics
            or any(row.get(key) is not True for key in (
                "finite_loss",
                "finite_gradients",
                "finite_parameters",
                "finite_optimizer_state",
                "all_trainable_gradients_present",
            ))
        ):
            raise ValueError(f"step ledger row {expected_step} violates the finite contract")
        if expected_arm == "CONTROL" and (metrics["kl"] != 0.0 or metrics["flip"] != 0.0):
            raise ValueError("CONTROL step ledger contains nonzero auxiliary losses")
    return rows


def validate_completed_run(
    path: str | Path,
    *,
    expected_arm: str,
    receipt: dict[str, Any],
    expected_schedule: tuple[str, ...],
    expected_static_identity: dict[str, Any],
) -> dict[str, Any]:
    root_key = "control_train" if expected_arm == "CONTROL" else "hoppair_train"
    expected_path = exact_absolute_path(receipt["output_roots"][root_key]) / "run_manifest.json"
    manifest_path = exact_absolute_path(path, description=f"{expected_arm} run manifest")
    if manifest_path != expected_path:
        raise ValueError(f"{expected_arm} run manifest path differs from the frozen output root")
    manifest, manifest_sha256 = verified_json_object(
        manifest_path,
        description=f"{expected_arm} run manifest",
    )
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("status") != "completed"
        or manifest.get("global_step") != OPTIMIZER_STEPS
        or manifest.get("consumed_orbits") != TRAIN_ORBITS
        or manifest.get("epochs_completed") != 1
        or manifest.get("optimizer_steps_skipped") != 0
        or manifest.get("hardware_identity") != receipt["hardware_lock"][expected_arm]
    ):
        raise ValueError(f"{expected_arm} did not complete the exact training contract")
    immutable = manifest.get("immutable")
    expected_schedule_binding = {
        "algorithm": receipt["runtime"]["schedule_algorithm"],
        "sha256": receipt["exact_bindings"]["schedule_sha256"],
        "orbits": TRAIN_ORBITS,
    }
    if (
        not isinstance(immutable, dict)
        or immutable.get("arm") != expected_arm
        or immutable.get("receipt") != receipt["_validated"]
        or immutable.get("protocol") != receipt["exact_bindings"]["protocol"]
        or immutable.get("prepared_manifest")
        != receipt["exact_bindings"]["prepared_manifest"]
        or immutable.get("schedule") != expected_schedule_binding
        or immutable.get("train") != receipt["exact_bindings"]["train"]
        or immutable.get("static_identity") != expected_static_identity
        or immutable.get("tokenizer_semantics") != receipt["exact_bindings"]["tokenizer"]
        or immutable.get("runtime") != receipt["runtime"]
        or immutable.get("lora") != receipt["lora"]
        or immutable.get("objective") != receipt["objectives"][expected_arm]
        or immutable.get("dataset_stats", {}).get("orbits") != TRAIN_ORBITS
        or immutable.get("dataset_stats", {}).get("states") != TRAIN_ORBITS * 3
        or immutable.get("dataset_stats", {}).get("truncated") != 0
    ):
        raise ValueError(f"{expected_arm} immutable run binding drift")
    for key in (
        "initial_trainable_parameters_sha256",
        "final_trainable_parameters_sha256",
        "final_optimizer_state_sha256",
    ):
        if not _is_sha256(manifest.get(key)):
            raise ValueError(f"{expected_arm} lacks valid {key}")
    ledger = manifest.get("step_ledger")
    expected_ledger = manifest_path.parent / "step_ledger.jsonl"
    if (
        not isinstance(ledger, dict)
        or exact_absolute_path(
            str(ledger.get("path", "")), description=f"{expected_arm} step ledger"
        )
        != expected_ledger
        or ledger.get("rows") != OPTIMIZER_STEPS
        or not _is_sha256(ledger.get("sha256"))
    ):
        raise ValueError(f"{expected_arm} step-ledger binding mismatch")
    _read_step_ledger(
        expected_ledger,
        expected_arm=expected_arm,
        expected_schedule=expected_schedule,
        expected_sha256=ledger["sha256"],
    )
    adapter_path = manifest_path.parent / "final_adapter"
    actual_checkpoint = artifact_manifest(adapter_path, require_adapter=True)
    recorded = manifest.get("checkpoint")
    if (
        not isinstance(recorded, dict)
        or exact_absolute_path(
            str(recorded.get("path", "")), description=f"{expected_arm} checkpoint"
        )
        != adapter_path
        or recorded.get("sha256") != actual_checkpoint["sha256"]
    ):
        raise ValueError(f"{expected_arm} checkpoint differs from its completed manifest")
    manifest["_validated"] = {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "adapter": actual_checkpoint,
    }
    return manifest


def validate_paired_fairness(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    arms = {first["immutable"]["arm"], second["immutable"]["arm"]}
    if arms != {"CONTROL", "HopPAIR"}:
        raise ValueError("paired fairness requires one CONTROL and one HopPAIR run")
    checks = {
        "initial_trainable_parameters_exact": first["initial_trainable_parameters_sha256"]
        == second["initial_trainable_parameters_sha256"],
        "schedule_exact": first["immutable"]["schedule"] == second["immutable"]["schedule"],
        "train_artifact_exact": first["immutable"]["train"] == second["immutable"]["train"],
        "static_model_exact": first["immutable"]["static_identity"]
        == second["immutable"]["static_identity"],
        "tokenizer_exact": first["immutable"]["tokenizer_semantics"]
        == second["immutable"]["tokenizer_semantics"],
        "runtime_exact": first["immutable"]["runtime"] == second["immutable"]["runtime"],
        "lora_exact": first["immutable"]["lora"] == second["immutable"]["lora"],
    }
    if not all(checks.values()):
        raise ValueError(f"paired training fairness failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "diagnostics": {
            # Descriptive only: equal final checkpoints are suspicious and
            # reportable, but are not an integrity failure when initialization,
            # data, schedule, and executable objectives are all exact.
            "checkpoints_distinct": first["checkpoint"]["sha256"]
            != second["checkpoint"]["sha256"],
        },
        "initial_trainable_parameters_sha256": first[
            "initial_trainable_parameters_sha256"
        ],
        "schedule_sha256": first["immutable"]["schedule"]["sha256"],
        "control_run_sha256": (
            first if first["immutable"]["arm"] == "CONTROL" else second
        )["_validated"]["sha256"],
        "hoppair_run_sha256": (
            first if first["immutable"]["arm"] == "HopPAIR" else second
        )["_validated"]["sha256"],
    }


def _load_model(
    model_path: Path,
    adapter_path: Path | None,
    *,
    base_identity: dict[str, Any],
    adapter_identity: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    identity: dict[str, Any] = {"base_model": base_identity, "adapter": None}
    if adapter_path is not None:
        from peft import PeftModel

        if adapter_identity is None:
            raise ValueError("trained model load lacks its verified adapter identity")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        identity["adapter"] = adapter_identity
    model.to("cuda:0")
    model.eval()
    model.config.use_cache = True
    return model, identity


def _revalidate_generation_identities(
    receipt: dict[str, Any],
    model_path: Path,
    adapter_path: Path | None,
    expected_adapter: dict[str, Any] | None,
) -> dict[str, Any]:
    refreshed = validate_launch_receipt(CANONICAL_RECEIPT, purpose="metadata")
    if refreshed["_validated"]["sha256"] != receipt["_validated"]["sha256"]:
        raise ValueError("launch receipt changed during generation")
    static = validate_static_model_identity(refreshed, model_path)
    for binding_name in ("train", "dev"):
        entry = receipt["exact_bindings"][binding_name]
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError(f"{binding_name} artifact changed during generation")
    adapter = None
    if adapter_path is not None:
        adapter = artifact_manifest(adapter_path, require_adapter=True)
        if expected_adapter is None or adapter != expected_adapter:
            raise ValueError("adapter identity changed during generation")
    elif expected_adapter is not None:
        raise ValueError("BASE identity unexpectedly includes an adapter")
    return {"base_model": static["base_model"], "adapter": adapter}


def _run(args: argparse.Namespace) -> int:
    receipt = validate_launch_receipt(args.launch_receipt, purpose="generation")
    input_path, output = _validate_cli(args, receipt)
    initial_static_identity = validate_static_model_identity(receipt, args.model_path)

    # This is deliberately before the first dev-body hash/read.  BASE is also
    # blocked until both arms complete, so no development output can influence
    # training or paired-initialization verification.
    train_records = load_train_records(
        receipt["exact_bindings"]["train"]["path"],
        expected_sha256=receipt["exact_bindings"]["train"]["sha256"],
    )
    train_orbit_ids = [
        str(train_records[start]["orbit_id"]) for start in range(0, len(train_records), 3)
    ]
    expected_schedule = frozen_orbit_schedule(train_orbit_ids)
    if schedule_sha256(expected_schedule) != receipt["exact_bindings"]["schedule_sha256"]:
        raise ValueError("independently recomputed train schedule differs from receipt")

    control_run = validate_completed_run(
        args.control_run_manifest,
        expected_arm="CONTROL",
        receipt=receipt,
        expected_schedule=expected_schedule,
        expected_static_identity=initial_static_identity,
    )
    hoppair_run = validate_completed_run(
        args.hoppair_run_manifest,
        expected_arm="HopPAIR",
        receipt=receipt,
        expected_schedule=expected_schedule,
        expected_static_identity=initial_static_identity,
    )
    fairness = validate_paired_fairness(control_run, hoppair_run)

    own_run: dict[str, Any] | None = None
    adapter_path: Path | None = None
    if args.arm == "BASE":
        if args.adapter_path is not None:
            raise ValueError("BASE generation must not provide an adapter")
    else:
        if args.adapter_path is None:
            raise ValueError("trained generation requires its completed adapter")
        own_run = control_run if args.arm == "CONTROL" else hoppair_run
        adapter_path = exact_absolute_path(args.adapter_path, description="--adapter-path")
        expected_adapter = exact_absolute_path(own_run["checkpoint"]["path"])
        if adapter_path != expected_adapter:
            raise ValueError("adapter path differs from the completed own-arm run")

    expected_dev_sha = receipt["exact_bindings"]["dev"]["sha256"]
    records = load_dev_records(input_path, expected_sha256=expected_dev_sha)

    tokenizer = load_tokenizer(args.model_path, receipt)
    tokenizer.padding_side = "left"
    prompts: list[str] = []
    prompt_hashes: list[str] = []
    prompt_lengths: list[int] = []
    for record in records:
        rendered = render_prepared_record(tokenizer, record)
        prompts.append(rendered.prompt)
        prompt_hashes.append(rendered.prompt_sha256)
        encoded = tokenizer(
            rendered.prompt,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        prompt_lengths.append(len(encoded))
    if max(prompt_lengths) > receipt["runtime"]["max_length"]:
        raise ValueError("development prompt exceeds the frozen no-truncation limit")

    hardware_identity = ensure_bf16_single_cuda(receipt, args.arm)
    import torch

    assert_no_symlink_components(
        output,
        description="generation output root before creation",
        allow_missing_tail=True,
    )
    try:
        output.mkdir(parents=True)
    except FileExistsError as exc:
        raise FileExistsError(f"generation refuses existing output directory: {output}") from exc
    assert_no_symlink_components(output, description="generation output root after creation")
    generation_started_at = datetime.now(timezone.utc).isoformat()
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": GENERATION_SCHEMA,
            "status": "running",
            "started_at": generation_started_at,
            "arm": args.arm,
            "receipt": receipt["_validated"],
            "input": receipt["exact_bindings"]["dev"],
        },
    )
    expected_adapter_identity = (
        own_run["_validated"]["adapter"] if own_run is not None else None
    )
    model_path = exact_absolute_path(args.model_path, description="--model-path")
    pre_load_identity = _revalidate_generation_identities(
        receipt,
        model_path,
        adapter_path,
        expected_adapter_identity,
    )
    if pre_load_identity != {
        "base_model": initial_static_identity["base_model"],
        "adapter": expected_adapter_identity,
    }:
        raise ValueError("model/adapter identity changed before inference load")
    model, model_identity = _load_model(
        model_path,
        adapter_path,
        base_identity=pre_load_identity["base_model"],
        adapter_identity=expected_adapter_identity,
    )
    post_load_identity = _revalidate_generation_identities(
        receipt,
        model_path,
        adapter_path,
        expected_adapter_identity,
    )
    if post_load_identity != model_identity:
        raise ValueError("model/adapter identity changed across inference load")
    generation = receipt["runtime"]["generation"]
    batch_size = int(generation["batch_size"])
    max_new_tokens = int(generation["max_new_tokens"])
    predictions: list[dict[str, Any]] = []
    token_counts: list[int] = []
    budget_exhausted = 0
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        ).to("cuda:0")
        prompt_width = int(encoded.input_ids.shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        suffix = generated[:, prompt_width:]
        for offset, (record, token_ids) in enumerate(zip(batch_records, suffix, strict=True)):
            ids = token_ids.tolist()
            saw_eos = tokenizer.eos_token_id in ids
            if saw_eos:
                ids = ids[: ids.index(tokenizer.eos_token_id)]
            while ids and ids[-1] == tokenizer.pad_token_id:
                ids.pop()
            if not saw_eos and len(ids) >= max_new_tokens:
                budget_exhausted += 1
            # Remove only the explicitly handled EOS/PAD terminators above.
            # Other generated control tokens remain visible and therefore score
            # invalid instead of being silently repaired into a valid record.
            prediction = tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            token_counts.append(len(ids))
            predictions.append(
                {
                    "id": record["id"],
                    "prediction": prediction,
                    "arm_id": args.arm,
                    "dataset_manifest_sha256": receipt["exact_bindings"][
                        "prepared_manifest"
                    ]["sha256"],
                }
            )
            expected_hash = prompt_hashes[start + offset]
            actual_hash = hashlib.sha256(batch_prompts[offset].encode()).hexdigest()
            if actual_hash != expected_hash:
                raise AssertionError("prompt changed between whitelist rendering and generation")
    output_ids = [str(row["id"]) for row in predictions]
    if len(output_ids) != DEV_RECORDS or len(set(output_ids)) != DEV_RECORDS:
        raise RuntimeError("generation did not produce exactly one output per unique development id")
    if output_ids != [str(record["id"]) for record in records]:
        raise RuntimeError("generation output order differs from the bound development input")

    post_generation_identity = _revalidate_generation_identities(
        receipt,
        model_path,
        adapter_path,
        expected_adapter_identity,
    )
    if post_generation_identity != model_identity:
        raise ValueError("model/adapter/source identity changed during generation")

    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    prediction_sha = sha256_file(prediction_path)
    checkpoint_sha = (
        receipt["exact_bindings"]["base_model"]["sha256"]
        if own_run is None
        else own_run["checkpoint"]["sha256"]
    )
    binding = {
        "arm_id": args.arm,
        "dataset_manifest_sha256": receipt["exact_bindings"]["prepared_manifest"]["sha256"],
        "split_artifact_sha256": receipt["exact_bindings"]["dev"]["sha256"],
        "predictions_sha256": prediction_sha,
        "protocol_sha256": receipt["exact_bindings"]["protocol"]["sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "launch_receipt_sha256": receipt["_validated"]["sha256"],
        "schedule_sha256": receipt["exact_bindings"]["schedule_sha256"],
    }
    if own_run is not None:
        binding["initialization_sha256"] = own_run[
            "initial_trainable_parameters_sha256"
        ]
    binding_path = output / "binding.json"
    atomic_json(binding_path, binding)
    manifest = {
        "schema_version": GENERATION_SCHEMA,
        "status": "completed",
        "started_at": generation_started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "receipt": receipt["_validated"],
        "input": {
            **receipt["exact_bindings"]["dev"],
            "prompt_tokens_max": max(prompt_lengths),
            "prompt_tokens_total": sum(prompt_lengths),
        },
        "model_identity": model_identity,
        "hardware_identity": hardware_identity,
        "training_run": own_run["_validated"] if own_run is not None else None,
        "paired_training_fairness": fairness,
        "decoding": {
            "do_sample": False,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
            "num_return_sequences": 1,
            "length_preflight_status": generation["length_preflight_status"],
            "budget_exhausted_rows": budget_exhausted,
        },
        "predictions": {
            "path": str(prediction_path),
            "sha256": prediction_sha,
            "rows": len(predictions),
            "unique_ids": len(set(output_ids)),
            "generated_tokens_total": sum(token_counts),
            "generated_tokens_max": max(token_counts),
        },
        "binding": {"path": str(binding_path), "sha256": sha256_file(binding_path)},
    }
    atomic_json(output / "manifest.json", manifest)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except BaseException as exc:
        manifest_path = Path(args.output_dir).expanduser().resolve() / "manifest.json"
        if manifest_path.is_file():
            try:
                current = read_json_object(manifest_path, "generation manifest")
                if current.get("status") == "running":
                    current.update(
                        {
                            "status": "failed",
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                            "failure": {"type": type(exc).__name__, "message": str(exc)},
                        }
                    )
                    atomic_json(manifest_path, current)
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEV_ORBITS",
    "DEV_RECORDS",
    "GENERATION_SCHEMA",
    "load_dev_records",
    "main",
    "parser",
    "validate_completed_run",
    "validate_paired_fairness",
]
