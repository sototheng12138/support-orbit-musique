from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from support_orbit_musique.backend import (
    CANONICAL_PREPARED_MANIFEST,
    CANONICAL_OUTPUT_ROOTS,
    CANONICAL_PROTOCOL,
    CANONICAL_RECEIPT,
    EXPECTED_OUTPUT_ROOT_KEYS,
    RECEIPT_SCHEMA,
    LoraSpec,
    REQUIRED_SOURCE_LOCK,
    _validate_hardware_lock,
    artifact_manifest,
    assert_gpu_uuid_idle,
    exact_absolute_path,
    sha256_file,
    tensor_collection_sha256,
    validate_launch_receipt,
    validate_output_roots,
    validate_runtime_cuda_versions,
    verified_json_object,
    verified_jsonl_objects,
)
from support_orbit_musique.generate import (
    _read_step_ledger,
    load_dev_records,
    parser as generation_parser,
    validate_paired_fairness,
)
from support_orbit_musique.train import (
    OPTIMIZER_STEPS,
    TRAIN_ORBITS,
    assert_all_trainable_gradients_present,
    expected_learning_rate,
    frozen_orbit_schedule,
    load_train_records,
    parser as training_parser,
    schedule_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _orbit_rows(split: str, orbit_id: str = "o") -> list[dict[str, object]]:
    return [
        {
            "id": f"{orbit_id}::{state}",
            "orbit_id": orbit_id,
            "state": state,
            "split": split,
            "training_read_allowed": True,
            "sealed": False,
        }
        for state in ("C", "D", "M")
    ]


def test_lora_spec_is_the_frozen_all_linear_contract() -> None:
    spec = LoraSpec().to_dict()
    assert spec["r"] == 16
    assert spec["alpha"] == 32
    assert spec["dropout"] == 0.05
    assert spec["coverage"] == "all_transformer_linear"
    assert spec["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    with pytest.raises(ValueError, match="target modules"):
        LoraSpec(target_modules=("q_proj",))


def test_schedule_is_deterministic_unique_and_exactly_240_updates() -> None:
    orbit_ids = [f"orbit-{index:04d}" for index in range(TRAIN_ORBITS)]
    first = frozen_orbit_schedule(orbit_ids)
    second = frozen_orbit_schedule(list(reversed(orbit_ids)))
    assert first == second
    assert len(first) == len(set(first)) == TRAIN_ORBITS
    assert len(first) // 8 == OPTIMIZER_STEPS
    assert schedule_sha256(first) == schedule_sha256(second)
    with pytest.raises(ValueError, match="1920"):
        frozen_orbit_schedule(orbit_ids[:-1])


def test_train_and_dev_loaders_validate_orbit_atomicity_and_split(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    _write_jsonl(train, _orbit_rows("train"))
    _write_jsonl(dev, _orbit_rows("dev"))
    assert len(load_train_records(train, expected_orbits=1)) == 3
    assert len(load_dev_records(dev, expected_orbits=1)) == 3

    broken = tmp_path / "broken.jsonl"
    rows = _orbit_rows("train")
    rows[1]["state"] = "C"
    _write_jsonl(broken, rows)
    with pytest.raises(ValueError, match="canonical C/D/M"):
        load_train_records(broken, expected_orbits=1)


def test_protected_paths_fail_before_open() -> None:
    with pytest.raises(ValueError, match="protected"):
        load_train_records("/does/not/exist/shadow.sealed.jsonl")
    with pytest.raises(ValueError, match="protected"):
        load_dev_records("/does/not/exist/musique_full_v1.0_test.jsonl")
    with pytest.raises(ValueError, match="protected"):
        load_train_records("/does/not/exist/prepared_data_v1/train.jsonl")
    with pytest.raises(ValueError, match="protected"):
        load_dev_records("/does/not/exist/prepared_data_v1/dev.jsonl")


def test_cli_surfaces_only_receipt_bound_inputs() -> None:
    train_args = training_parser().parse_args(
        [
            "--arm",
            "HopPAIR",
            "--model-path",
            "/model",
            "--train-file",
            "/train",
            "--prepared-manifest",
            "/manifest",
            "--protocol",
            "/protocol",
            "--launch-receipt",
            "/receipt",
            "--output-dir",
            "/output",
        ]
    )
    assert train_args.arm == "HopPAIR"
    assert not hasattr(train_args, "resume_from_checkpoint")

    generation_args = generation_parser().parse_args(
        [
            "--arm",
            "BASE",
            "--model-path",
            "/model",
            "--control-run-manifest",
            "/control-run",
            "--hoppair-run-manifest",
            "/hoppair-run",
            "--input-file",
            "/dev",
            "--prepared-manifest",
            "/manifest",
            "--protocol",
            "/protocol",
            "--launch-receipt",
            "/receipt",
            "--output-dir",
            "/output",
        ]
    )
    assert generation_args.batch_size is None
    assert generation_args.max_new_tokens is None


def test_noncanonical_or_absent_receipt_fails_closed_before_model_load(tmp_path: Path) -> None:
    fake = tmp_path / "receipt.json"
    fake.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical path"):
        validate_launch_receipt(fake)
    assert CANONICAL_RECEIPT.name == "pilot_v2_launch_receipt.json"
    assert CANONICAL_PROTOCOL.name == "support_orbit_pilot_v2.json"
    assert CANONICAL_PREPARED_MANIFEST.parent.name == "prepared_data_v2"
    assert RECEIPT_SCHEMA == "support-orbit-musique.launch-receipt.v2"
    with pytest.raises(ValueError, match="canonical path"):
        validate_launch_receipt(PROJECT_ROOT / "protocols" / "pilot_v1_launch_receipt.json")
    if not CANONICAL_RECEIPT.exists():
        with pytest.raises(FileNotFoundError, match="future receipt"):
            validate_launch_receipt(CANONICAL_RECEIPT)


def test_exact_paths_and_verified_readers_reject_aliases_and_tampering(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="absolute"):
        exact_absolute_path("relative.json")

    payload = tmp_path / "payload.json"
    payload.write_text('{"ok":true}\n', encoding="utf-8")
    expected_sha = sha256_file(payload)
    value, actual_sha = verified_json_object(payload, expected_sha256=expected_sha)
    assert value == {"ok": True}
    assert actual_sha == expected_sha
    with pytest.raises(ValueError, match="hash mismatch"):
        verified_json_object(payload, expected_sha256="0" * 64)

    alias = tmp_path / "alias.json"
    alias.symlink_to(payload)
    with pytest.raises(ValueError, match="symlink component"):
        verified_json_object(alias)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        verified_json_object(duplicate)

    nonfinite = tmp_path / "nonfinite.jsonl"
    nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        verified_jsonl_objects(nonfinite)


def test_hardware_and_output_root_bindings_fail_closed() -> None:
    receipt: dict[str, object] = {
        "runtime": {
            "gpu_by_arm": {
                arm: f"GPU-unit-{index}"
                for index, arm in enumerate(("CONTROL", "HopPAIR", "BASE"))
            }
        },
        "hardware_lock": {
            arm: {
                "cuda_visible_devices": f"GPU-unit-{index}",
                "uuid": f"GPU-unit-{index}",
                "name": "NVIDIA A800-SXM4-80GB",
                "driver_version": "unit-driver",
                "driver_cuda_version": "12.4",
                "torch_cuda_version": "12.8",
                "compute_capability": [8, 0],
            }
            for index, arm in enumerate(("CONTROL", "HopPAIR", "BASE"))
        },
        "hardware_preflight": {
            "observations": {
                arm: {
                    "index": str(index),
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                    "compute_processes": 0,
                    "idle": True,
                }
                for index, arm in enumerate(("CONTROL", "HopPAIR", "BASE"))
            },
            "same_model": True,
            "idle_memory_ceiling_mib": 1_024,
            "idle_utilization_ceiling_percent": 0,
            "excluded_gpu": {
                "index": "3",
                "uuid": "GPU-unit-3",
                "present": True,
                "assigned": False,
            },
            "model_loaded": False,
            "cuda_initialized_by_freezer": False,
        },
        "output_roots": {
            key: value for key, value in CANONICAL_OUTPUT_ROOTS.items()
        },
    }
    _validate_hardware_lock(receipt)  # type: ignore[arg-type]
    roots = validate_output_roots(receipt)  # type: ignore[arg-type]
    assert set(roots) == EXPECTED_OUTPUT_ROOT_KEYS

    mismatched_gpu = copy.deepcopy(receipt)
    mismatched_gpu["hardware_lock"]["CONTROL"]["uuid"] = "not-a-gpu-uuid"  # type: ignore[index]
    with pytest.raises(ValueError, match="invalid identity"):
        _validate_hardware_lock(mismatched_gpu)  # type: ignore[arg-type]

    ordinal_bound = copy.deepcopy(receipt)
    ordinal_bound["runtime"]["gpu_by_arm"]["CONTROL"] = "0"  # type: ignore[index]
    ordinal_bound["hardware_lock"]["CONTROL"]["cuda_visible_devices"] = "0"  # type: ignore[index]
    with pytest.raises(ValueError, match="invalid identity"):
        _validate_hardware_lock(ordinal_bound)  # type: ignore[arg-type]

    duplicate_root = copy.deepcopy(receipt)
    duplicate_root["output_roots"]["control_train"] = duplicate_root["output_roots"][  # type: ignore[index]
        "hoppair_train"
    ]
    with pytest.raises(ValueError, match="canonical seed-17"):
        validate_output_roots(duplicate_root)  # type: ignore[arg-type]

    nested_root = copy.deepcopy(receipt)
    nested_root["output_roots"]["control_train"] = (  # type: ignore[index]
        str(nested_root["output_roots"]["hoppair_train"]) + "/nested"  # type: ignore[index]
    )
    with pytest.raises(ValueError, match="canonical seed-17"):
        validate_output_roots(nested_root)  # type: ignore[arg-type]

    alternate_root = copy.deepcopy(receipt)
    alternate_root["output_roots"]["control_train"] = str(  # type: ignore[index]
        PROJECT_ROOT / "runs" / "pilot_v2_seed17_alternate" / "control_train"
    )
    with pytest.raises(ValueError, match="sole canonical seed-17"):
        validate_output_roots(alternate_root)  # type: ignore[arg-type]


def test_runtime_uuid_idle_recheck_fails_on_new_compute_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = __import__("subprocess").CompletedProcess
    results = iter(
        [
            completed([], 0, "GPU-unit-0, NVIDIA A800-SXM4-80GB, 570.00, 0, 0\n", ""),
            completed([], 0, "", ""),
            completed([], 0, "CUDA Version: 12.4\n", ""),
        ]
    )
    monkeypatch.setattr(
        "support_orbit_musique.backend.subprocess.run",
        lambda *args, **kwargs: next(results),
    )
    identity = assert_gpu_uuid_idle("GPU-unit-0")
    assert identity["uuid"] == "GPU-unit-0"

    busy_results = iter(
        [
            completed([], 0, "GPU-unit-0, NVIDIA A800-SXM4-80GB, 570.00, 0, 0\n", ""),
            completed([], 0, "1234, python, 500\n", ""),
            completed([], 0, "CUDA Version: 12.4\n", ""),
        ]
    )
    monkeypatch.setattr(
        "support_orbit_musique.backend.subprocess.run",
        lambda *args, **kwargs: next(busy_results),
    )
    with pytest.raises(RuntimeError, match="no longer idle"):
        assert_gpu_uuid_idle("GPU-unit-0")


def test_driver_and_torch_cuda_versions_are_independent_exact_locks() -> None:
    expected = {"driver_cuda_version": "12.4", "torch_cuda_version": "12.8"}
    validate_runtime_cuda_versions(
        expected,
        observed_driver_cuda_version="12.4",
        observed_torch_cuda_version="12.8",
    )
    with pytest.raises(RuntimeError, match="CUDA identity"):
        validate_runtime_cuda_versions(
            expected,
            observed_driver_cuda_version="12.5",
            observed_torch_cuda_version="12.8",
        )
    with pytest.raises(RuntimeError, match="CUDA identity"):
        validate_runtime_cuda_versions(
            expected,
            observed_driver_cuda_version="12.4",
            observed_torch_cuda_version="12.9",
        )


def test_required_source_lock_is_complete_on_disk() -> None:
    missing = [relative for relative in REQUIRED_SOURCE_LOCK if not (PROJECT_ROOT / relative).is_file()]
    assert missing == []


def test_artifact_and_tensor_hashes_are_content_sensitive(tmp_path: Path) -> None:
    artifact = tmp_path / "adapter"
    artifact.mkdir()
    (artifact / "adapter_config.json").write_text("{}", encoding="utf-8")
    (artifact / "adapter_model.safetensors").write_bytes(b"weights")
    first = artifact_manifest(artifact, require_adapter=True)
    (artifact / "adapter_model.safetensors").write_bytes(b"changed")
    second = artifact_manifest(artifact, require_adapter=True)
    assert first["sha256"] != second["sha256"]

    tensors_a = [("b", torch.tensor([2.0])), ("a", torch.tensor([1.0]))]
    tensors_b = list(reversed(tensors_a))
    assert tensor_collection_sha256(tensors_a) == tensor_collection_sha256(tensors_b)
    assert tensor_collection_sha256(tensors_a) != tensor_collection_sha256(
        [("a", torch.tensor([9.0])), ("b", torch.tensor([2.0]))]
    )
    assert len(tensor_collection_sha256([("scalar_step", torch.tensor(1.0))])) == 64


def _fake_run(arm: str, checkpoint: str, initial: str = "a" * 64) -> dict[str, object]:
    return {
        "immutable": {
            "arm": arm,
            "schedule": {"sha256": "b" * 64},
            "train": {"sha256": "c" * 64},
            "static_identity": {"sha256": "d" * 64},
            "tokenizer_semantics": {"sha256": "e" * 64},
            "runtime": {"seed": 17},
            "lora": LoraSpec().to_dict(),
        },
        "initial_trainable_parameters_sha256": initial,
        "checkpoint": {"sha256": checkpoint * 64},
        "_validated": {"sha256": ("f" if arm == "CONTROL" else "0") * 64},
    }


def test_paired_fairness_requires_same_real_initialization_and_schedule() -> None:
    control = _fake_run("CONTROL", "1")
    treatment = _fake_run("HopPAIR", "2")
    result = validate_paired_fairness(control, treatment)  # type: ignore[arg-type]
    assert result["passed"]
    treatment["initial_trainable_parameters_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="fairness failed"):
        validate_paired_fairness(control, treatment)  # type: ignore[arg-type]


def test_identical_final_checkpoints_are_descriptive_not_a_fairness_failure() -> None:
    control = _fake_run("CONTROL", "1")
    treatment = _fake_run("HopPAIR", "1")
    result = validate_paired_fairness(control, treatment)  # type: ignore[arg-type]
    assert result["passed"] is True
    assert result["diagnostics"] == {"checkpoints_distinct": False}


def test_frozen_learning_rate_and_all_gradient_contract() -> None:
    assert expected_learning_rate(0) == 0.0
    assert expected_learning_rate(1) == pytest.approx(2e-4 / 8)
    assert expected_learning_rate(8) == pytest.approx(2e-4)
    assert expected_learning_rate(OPTIMIZER_STEPS) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError, match="0..240"):
        expected_learning_rate(-1)
    with pytest.raises(ValueError, match="0..240"):
        expected_learning_rate(OPTIMIZER_STEPS + 1)

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    with pytest.raises(RuntimeError, match="without gradients"):
        assert_all_trainable_gradients_present(model)
    model(torch.ones(1, 2)).sum().backward()
    assert_all_trainable_gradients_present(model)


def test_step_ledger_requires_240_finite_complete_rows(tmp_path: Path) -> None:
    ledger = tmp_path / "step_ledger.jsonl"
    schedule = frozen_orbit_schedule(
        [f"ledger-orbit-{index:04d}" for index in range(TRAIN_ORBITS)]
    )
    rows = []
    for step in range(1, OPTIMIZER_STEPS + 1):
        start = (step - 1) * 8
        orbit_batch_sha256 = hashlib.sha256(
            "".join(f"{orbit}\n" for orbit in schedule[start : start + 8]).encode()
        ).hexdigest()
        rows.append(
            {
                "optimizer_step": step,
                "microsteps": 8,
                "consumed_orbits": step * 8,
                "orbit_batch_sha256": orbit_batch_sha256,
                "learning_rate_applied": expected_learning_rate(step - 1),
                "learning_rate_next": expected_learning_rate(step),
                "grad_norm_preclip": 1.0,
                "max_grad_norm": 1.0,
                "step_seconds": 0.1,
                "metrics": {
                    "loss": 1.0,
                    "sft": 1.0,
                    "kl": 0.0,
                    "flip": 0.0,
                    "ce_c": 1.0,
                    "ce_d": 1.0,
                    "ce_m": 1.0,
                    "z_c": 1.0,
                    "z_d": 1.0,
                    "z_m": -1.0,
                    "c_minus_m": 2.0,
                    "d_minus_m": 2.0,
                },
                "finite_loss": True,
                "finite_gradients": True,
                "all_trainable_gradients_present": True,
                "finite_parameters": True,
                "finite_optimizer_state": True,
            }
        )
    _write_jsonl(ledger, rows)
    ledger_sha = sha256_file(ledger)
    assert (
        len(
            _read_step_ledger(
                ledger,
                expected_arm="CONTROL",
                expected_schedule=schedule,
                expected_sha256=ledger_sha,
            )
        )
        == OPTIMIZER_STEPS
    )

    rows[-1]["all_trainable_gradients_present"] = False
    _write_jsonl(ledger, rows)
    with pytest.raises(ValueError, match="finite contract"):
        _read_step_ledger(ledger, expected_schedule=schedule)

    rows[-1]["all_trainable_gradients_present"] = True
    rows[0]["learning_rate_applied"] = 1e-4
    _write_jsonl(ledger, rows)
    with pytest.raises(ValueError, match="finite contract"):
        _read_step_ledger(ledger, expected_schedule=schedule)

    rows[0]["learning_rate_applied"] = expected_learning_rate(0)
    rows[0]["orbit_batch_sha256"] = "a" * 64
    _write_jsonl(ledger, rows)
    with pytest.raises(ValueError, match="finite contract"):
        _read_step_ledger(ledger, expected_schedule=schedule)

    with pytest.raises(ValueError, match="hash mismatch"):
        _read_step_ledger(ledger, expected_sha256=ledger_sha)
