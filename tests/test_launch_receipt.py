from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import freeze_launch
import support_orbit_musique.backend as backend
from support_orbit_musique.backend import (
    CANONICAL_RECEIPT,
    EXPECTED_OUTPUT_ROOT_KEYS,
    EXPECTED_RUNTIME,
    RECEIPT_SCHEMA,
    RECEIPT_STATUS,
    REQUIRED_SOURCE_LOCK,
    _validate_environment_lock,
    _validate_independent_redteam_signoff,
    assert_no_symlink_components,
    canonical_json_sha256,
)


def _completed(stdout: bytes, returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=b"")


def _hardware() -> dict[str, object]:
    lock = {}
    observations = {}
    for arm, index in freeze_launch.GPU_BY_ARM.items():
        lock[arm] = {
            "cuda_visible_devices": f"GPU-unit-{index}",
            "uuid": f"GPU-unit-{index}",
            "name": "NVIDIA A800-SXM4-80GB",
            "driver_version": "570.00",
            "driver_cuda_version": "12.4",
            "torch_cuda_version": "12.8",
            "compute_capability": [8, 0],
        }
        observations[arm] = {
            "index": index,
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "compute_processes": 0,
            "idle": True,
        }
    return {
        "hardware_lock": lock,
        "observations": observations,
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
    }


def _data() -> dict[str, object]:
    return {
        "protocol": {
            "path": str(freeze_launch.CANONICAL_PROTOCOL),
            "sha256": freeze_launch.PROTOCOL_SHA256,
        },
        "prepared_manifest": {
            "path": str(freeze_launch.CANONICAL_PREPARED_MANIFEST),
            "sha256": freeze_launch.MANIFEST_SHA256,
        },
        "train": {
            "path": str(freeze_launch.TRAIN_PATH),
            "sha256": freeze_launch.TRAIN_SHA256,
            "orbits": 1_920,
            "records": 5_760,
        },
        "dev": {
            "path": str(freeze_launch.DEV_PATH),
            "sha256": freeze_launch.DEV_SHA256,
            "orbits": 400,
            "records": 1_200,
        },
        "schedule_sha256": freeze_launch.SCHEDULE_SHA256,
        "release_validation": {"passed": True, "shadow_body_opened": False},
        "train_sidecar": {"sha256": "1" * 64},
        "dev_sidecar": {"sha256": "2" * 64},
        "sealed_shadow_metadata": {
            "body_path": str(freeze_launch.SHADOW_BODY_PATH),
            "body_opened": False,
            "writer_sha256": freeze_launch.SHADOW_WRITER_SHA256,
        },
    }


def _model() -> dict[str, object]:
    return {
        "base_model_binding": {
            "path": str(freeze_launch.CANONICAL_MODEL_PATH),
            "sha256": "a" * 64,
        },
        "tokenizer_binding": {
            "path": str(freeze_launch.CANONICAL_MODEL_PATH),
            "sha256": "b" * 64,
            "chat_template_sha256": "c" * 64,
            "eos_token_id": 151645,
            "pad_token_id": 151643,
        },
        "base_model_manifest": {"path": "model", "files": [], "sha256": "a" * 64},
        "tokenizer_manifest": {"path": "model", "files": [], "sha256": "b" * 64},
        "model_loaded": False,
        "cuda_initialized": False,
    }


def _tests() -> dict[str, object]:
    return {
        "passed": True,
        "returncode": 0,
        "cuda_visible_devices": "",
        "offline": True,
        "commands": [],
    }


def _environment() -> dict[str, object]:
    return {
        "python": {
            "executable": str(freeze_launch.CANONICAL_PYTHON),
            "version": "3.13.9",
        },
        "packages": {
            "torch": "2.9.1",
            "transformers": "4.57.3",
            "peft": "0.20.0",
            "accelerate": "1.14.0",
            "numpy": "2.4.6",
            "tokenizers": "0.22.1",
            "safetensors": "0.6.2",
        },
    }


def _redteam(source_lock: dict[str, str]) -> dict[str, str]:
    return {
        "path": str(freeze_launch.CANONICAL_REDTEAM_SIGNOFF),
        "sha256": "d" * 64,
        "reviewer_id": "independent-unit-reviewer",
        "source_lock_sha256": canonical_json_sha256(source_lock),
    }


def test_freezer_defaults_to_dry_run_and_constants_are_exact() -> None:
    args = freeze_launch.parser().parse_args([])
    assert args.write_receipt is False
    assert CANONICAL_RECEIPT == freeze_launch.CANONICAL_RECEIPT
    assert RECEIPT_SCHEMA == "support-orbit-musique.launch-receipt.v2"
    assert freeze_launch.PROTOCOL_SHA256 == (
        "1a888808d4a09983d0e98d31f1668b2f5faf844dde3b44f4d5e863e555e0c33f"
    )
    assert freeze_launch.MANIFEST_SHA256 == (
        "84e97c0eb189d664149822c1244436f0b97e69e3cbca7e16be8186df52a7dfc1"
    )
    assert freeze_launch.TRAIN_SHA256 == (
        "641bae95c4eb410229fb8c87a52b24e58628c376c636632d242115ca7e4d5c12"
    )
    assert freeze_launch.DEV_SHA256 == (
        "84acfa99def02660aa74d836f07ff4219c0c838c29e19d7d335a126c7840b909"
    )
    assert freeze_launch.SCHEDULE_SHA256 == (
        "027c7d3eba760be6956060a487836088911ea5d7746e4fb5586f811c2c7c6ac4"
    )
    assert freeze_launch.GPU_BY_ARM == {"CONTROL": "0", "HopPAIR": "1", "BASE": "2"}
    assert freeze_launch.EXCLUDED_GPU_INDEX == "3"


def test_complete_source_lock_includes_freezer_and_its_tests() -> None:
    assert "freeze_launch.py" in REQUIRED_SOURCE_LOCK
    assert "tests/test_launch_receipt.py" in REQUIRED_SOURCE_LOCK


def test_receipt_shape_exactly_matches_backend_contract() -> None:
    source_lock = {relative: "e" * 64 for relative in sorted(REQUIRED_SOURCE_LOCK)}
    receipt = freeze_launch._compose_receipt(
        data=_data(),
        model=_model(),
        source_lock=source_lock,
        test_evidence=_tests(),
        hardware=_hardware(),
        output_roots=freeze_launch.OUTPUT_ROOTS,
        redteam=_redteam(source_lock),
        environment_lock=_environment(),
    )
    assert receipt["schema_version"] == RECEIPT_SCHEMA
    assert receipt["status"] == RECEIPT_STATUS
    assert set(receipt["exact_bindings"]) == {
        "protocol",
        "prepared_manifest",
        "train",
        "dev",
        "schedule_sha256",
        "base_model",
        "tokenizer",
        "source_lock_sha256",
    }
    assert {key: receipt["runtime"][key] for key in EXPECTED_RUNTIME} == EXPECTED_RUNTIME
    assert receipt["runtime"]["generation"] == {
        "batch_size": 1,
        "max_new_tokens": 128,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "length_preflight_status": "PASS",
    }
    assert receipt["runtime"]["gpu_by_arm"] == {
        "CONTROL": "GPU-unit-0",
        "HopPAIR": "GPU-unit-1",
        "BASE": "GPU-unit-2",
    }
    assert set(receipt["output_roots"]) == EXPECTED_OUTPUT_ROOT_KEYS
    assert receipt["forbidden_access"] == {
        "withdrawn_v1": True,
        "shadow": True,
        "official_dev": True,
        "official_test": True,
    }
    assert all(gate["status"] == "PASS" for gate in receipt["pre_gpu_gates"])
    expected_evidence = canonical_json_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key not in {"created_at", "evidence_sha256"}
        }
    )
    assert receipt["evidence_sha256"] == expected_evidence


def test_dry_candidate_is_not_gpu_authority() -> None:
    source_lock = {relative: "e" * 64 for relative in sorted(REQUIRED_SOURCE_LOCK)}
    receipt = freeze_launch._compose_receipt(
        data=_data(),
        model=_model(),
        source_lock=source_lock,
        test_evidence=_tests(),
        hardware=_hardware(),
        output_roots=freeze_launch.OUTPUT_ROOTS,
        redteam=None,
        environment_lock=_environment(),
    )
    assert receipt["status"] == "STOP_PENDING_INDEPENDENT_REDTEAM"
    assert receipt["independent_redteam_signoff"] is None
    assert receipt["pre_gpu_gates"][-1]["status"] == "PENDING"


def test_exclusive_receipt_pair_never_overwrites(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    payload = {"schema_version": RECEIPT_SCHEMA, "status": RECEIPT_STATUS}
    digest = freeze_launch._write_exclusive_receipt_pair(receipt_path, payload)
    raw = receipt_path.read_bytes()
    assert digest == hashlib.sha256(raw).hexdigest()
    assert Path(f"{receipt_path}.sha256").read_text(encoding="ascii") == (
        f"{digest}  {receipt_path.name}\n"
    )
    original = raw
    with pytest.raises(FileExistsError, match="overwrite"):
        freeze_launch._write_exclusive_receipt_pair(receipt_path, {"changed": True})
    assert receipt_path.read_bytes() == original


def test_path_guards_reject_symlink_and_ancestor_overlap(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        freeze_launch._assert_no_symlink_components(alias / "future", description="test")
    with pytest.raises(ValueError, match="symlink"):
        assert_no_symlink_components(alias / "future", description="runtime test")

    parent = tmp_path / "parent"
    child = parent / "child"
    with pytest.raises(ValueError, match="overlap"):
        freeze_launch._assert_disjoint({"parent": parent, "child": child}, "test")


def test_source_lock_comparison_fails_on_any_change() -> None:
    first = {"a": "1" * 64, "b": "2" * 64}
    freeze_launch._require_same_source_lock(first, dict(first), "test")
    changed = dict(first)
    changed["b"] = "3" * 64
    with pytest.raises(RuntimeError, match="b"):
        freeze_launch._require_same_source_lock(first, changed, "test")


def test_shadow_inspection_never_opens_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = tmp_path / "shadow.sealed.jsonl"
    body.write_bytes(b"sealed-body-must-not-open")
    sidecar = tmp_path / "shadow.sealed.jsonl.sha256"
    sidecar.write_text(
        f"{freeze_launch.SHADOW_WRITER_SHA256}  {body.name}\n",
        encoding="ascii",
    )
    marker = {
        "sealed": True,
        "training_read_allowed": False,
        "artifact": body.name,
        "orbit_count": 400,
        "record_count": 1_200,
        "bytes": body.stat().st_size,
        "sha256": freeze_launch.SHADOW_WRITER_SHA256,
    }
    sidecar_sha = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    real_open = os.open
    opened: list[Path] = []

    def guarded_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        candidate = Path(os.fspath(path))
        opened.append(candidate)
        if candidate == body:
            raise AssertionError("sealed shadow body was opened")
        return real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", guarded_open)
    result = freeze_launch._inspect_shadow_metadata(
        body_path=body,
        marker=marker,
        sidecar_path=sidecar,
        sidecar_expected_sha256=sidecar_sha,
    )
    assert result["body_opened"] is False
    assert body not in opened
    assert sidecar in opened


def test_v2_freeze_validates_metadata_without_opening_any_split_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not freeze_launch.TRAIN_PATH.exists() or not freeze_launch.DEV_PATH.exists():
        pytest.skip("public release omits MuSiQue-derived train/dev JSONL bodies")
    protected_bodies = {
        freeze_launch.TRAIN_PATH,
        freeze_launch.DEV_PATH,
        freeze_launch.SHADOW_BODY_PATH,
    }
    real_open = os.open
    opened: list[Path] = []

    def guarded_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        candidate = Path(os.path.abspath(os.fspath(path)))
        opened.append(candidate)
        if candidate in protected_bodies:
            raise AssertionError(f"pre-receipt split body was opened: {candidate}")
        return real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", guarded_open)
    result = freeze_launch.validate_frozen_inputs()
    assert result["body_access"] == {
        "train_opened": False,
        "dev_opened": False,
        "shadow_opened": False,
    }
    assert protected_bodies.isdisjoint(opened)


def test_hardware_inventory_binds_idle_same_model_012_and_excludes_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = "\n".join(
        f"{index}, GPU-unit-{index}, NVIDIA A800-SXM4-80GB, 570.00, 8.0, 0, 0"
        for index in range(4)
    ).encode()
    overview = b"NVIDIA-SMI 570.00 Driver Version: 570.00 CUDA Version: 12.4\n"
    results = iter(
        [_completed(inventory), _completed(b""), _completed(overview), _completed(b"12.8\n")]
    )
    monkeypatch.setattr(freeze_launch, "_run_command", lambda *args, **kwargs: next(results))
    hardware = freeze_launch.validate_hardware()
    assert set(hardware["hardware_lock"]) == {"CONTROL", "HopPAIR", "BASE"}
    assert hardware["hardware_lock"]["CONTROL"]["cuda_visible_devices"] == "GPU-unit-0"
    assert hardware["hardware_lock"]["HopPAIR"]["cuda_visible_devices"] == "GPU-unit-1"
    assert hardware["hardware_lock"]["BASE"]["cuda_visible_devices"] == "GPU-unit-2"
    assert hardware["excluded_gpu"] == {
        "index": "3",
        "uuid": "GPU-unit-3",
        "present": True,
        "assigned": False,
    }


def test_receipt_gpu_runtime_is_uuid_stable_when_ordinals_reorder() -> None:
    source_lock = {relative: "e" * 64 for relative in sorted(REQUIRED_SOURCE_LOCK)}
    hardware = _hardware()
    receipt = freeze_launch._compose_receipt(
        data=_data(),
        model=_model(),
        source_lock=source_lock,
        test_evidence=_tests(),
        hardware=hardware,
        output_roots=freeze_launch.OUTPUT_ROOTS,
        redteam=_redteam(source_lock),
        environment_lock=_environment(),
    )
    # Frozen runtime values are UUIDs, so a later CUDA ordinal permutation
    # cannot silently select a different physical device.
    assert set(receipt["runtime"]["gpu_by_arm"].values()) == {
        "GPU-unit-0",
        "GPU-unit-1",
        "GPU-unit-2",
    }
    assert all(
        entry["cuda_visible_devices"] == entry["uuid"]
        for entry in receipt["hardware_lock"].values()
    )


@pytest.mark.parametrize(
    ("inventory_change", "processes", "message"),
    [
        ({"index": 2, "name": "NVIDIA H100"}, b"", "same model"),
        ({"index": 1, "utilization": 1}, b"", "not idle"),
        ({}, b"GPU-unit-0, 123, python, 100\n", "active compute"),
    ],
)
def test_hardware_preflight_fails_closed_when_not_matched_or_idle(
    monkeypatch: pytest.MonkeyPatch,
    inventory_change: dict[str, object],
    processes: bytes,
    message: str,
) -> None:
    rows = []
    for index in range(4):
        name = "NVIDIA A800-SXM4-80GB"
        utilization = 0
        if inventory_change.get("index") == index:
            name = str(inventory_change.get("name", name))
            utilization = int(inventory_change.get("utilization", utilization))
        rows.append(f"{index}, GPU-unit-{index}, {name}, 570.00, 8.0, 0, {utilization}")
    inventory = "\n".join(rows).encode()
    overview = b"CUDA Version: 12.4\n"
    results = iter(
        [
            _completed(inventory),
            _completed(processes),
            _completed(overview),
            _completed(b"12.8\n"),
        ]
    )
    monkeypatch.setattr(freeze_launch, "_run_command", lambda *args, **kwargs: next(results))
    with pytest.raises(ValueError, match=message):
        freeze_launch.validate_hardware()


def _write_bound_json(path: Path, payload: dict[str, object]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def _signoff_payload(source_lock_sha: str) -> dict[str, object]:
    return {
        "schema_version": freeze_launch.REDTEAM_SCHEMA,
        "status": freeze_launch.REDTEAM_STATUS,
        "reviewer_kind": "independent_redteam",
        "reviewer_id": "unit-independent-reviewer",
        "created_at": "2026-08-14T00:00:00+00:00",
        **freeze_launch._redteam_expected(source_lock_sha),
        "findings": {"P0": 0, "P1": 0},
        "access_boundary": {
            "withdrawn_v1_data_body_opened": False,
            "shadow_body_opened": False,
            "official_dev_opened": False,
            "official_test_opened": False,
            "official_archive_opened": False,
            "gpu_model_loaded": False,
        },
    }


def test_independent_redteam_signoff_is_sidecar_and_source_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signoff_path = tmp_path / "pilot_v2_redteam_signoff.json"
    source_lock_sha = "f" * 64
    _write_bound_json(signoff_path, _signoff_payload(source_lock_sha))
    monkeypatch.setattr(freeze_launch, "CANONICAL_REDTEAM_SIGNOFF", signoff_path)
    result = freeze_launch.validate_redteam_signoff(
        source_lock_sha256=source_lock_sha,
        required=True,
    )
    assert result is not None
    assert result["reviewer_id"] == "unit-independent-reviewer"

    tampered = _signoff_payload(source_lock_sha)
    tampered["findings"] = {"P0": 0, "P1": 1}
    _write_bound_json(signoff_path, tampered)
    with pytest.raises(ValueError, match="approval contract"):
        freeze_launch.validate_redteam_signoff(
            source_lock_sha256=source_lock_sha,
            required=True,
        )


def test_backend_rejects_missing_forged_or_drifted_redteam_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signoff_path = tmp_path / "pilot_v2_redteam_signoff.json"
    source_lock_sha = "f" * 64
    payload = _signoff_payload(source_lock_sha)
    signoff_sha = _write_bound_json(signoff_path, payload)
    monkeypatch.setattr(backend, "CANONICAL_REDTEAM_SIGNOFF", signoff_path)
    bindings = {
        "protocol": {"sha256": freeze_launch.PROTOCOL_SHA256},
        "prepared_manifest": {"sha256": freeze_launch.MANIFEST_SHA256},
        "train": {"sha256": freeze_launch.TRAIN_SHA256},
        "dev": {"sha256": freeze_launch.DEV_SHA256},
        "schedule_sha256": freeze_launch.SCHEDULE_SHA256,
        "source_lock_sha256": source_lock_sha,
    }
    valid_receipt = {
        "independent_redteam_signoff": {
            "path": str(signoff_path),
            "sha256": signoff_sha,
            "reviewer_id": payload["reviewer_id"],
        }
    }
    _validate_independent_redteam_signoff(valid_receipt, bindings)

    forged = {"pre_gpu_gates": [{"name": "forged", "status": "PASS"}]}
    with pytest.raises(ValueError, match="lacks"):
        _validate_independent_redteam_signoff(forged, bindings)

    wrong_receipt_hash = copy.deepcopy(valid_receipt)
    wrong_receipt_hash["independent_redteam_signoff"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash differs"):
        _validate_independent_redteam_signoff(wrong_receipt_hash, bindings)

    drifted_payload = _signoff_payload(source_lock_sha)
    drifted_payload["reviewed_schedule_sha256"] = "0" * 64
    drifted_sha = _write_bound_json(signoff_path, drifted_payload)
    drifted_receipt = copy.deepcopy(valid_receipt)
    drifted_receipt["independent_redteam_signoff"]["sha256"] = drifted_sha
    with pytest.raises(ValueError, match="schedule"):
        _validate_independent_redteam_signoff(drifted_receipt, bindings)


def test_redteam_absence_is_pending_for_dry_run_but_blocks_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "absent.json"
    monkeypatch.setattr(freeze_launch, "CANONICAL_REDTEAM_SIGNOFF", path)
    assert (
        freeze_launch.validate_redteam_signoff(source_lock_sha256="f" * 64, required=False)
        is None
    )
    with pytest.raises(FileNotFoundError, match="requires"):
        freeze_launch.validate_redteam_signoff(source_lock_sha256="f" * 64, required=True)


def test_offline_check_environment_hides_cuda_and_forces_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, str]] = []

    def fake_success(argv: object, *, environment: dict[str, str]) -> dict[str, object]:
        observed.append(copy.deepcopy(environment))
        return {
            "argv": list(argv),  # type: ignore[arg-type]
            "returncode": 0,
            "stdout_bytes": 0,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_bytes": 0,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        }

    monkeypatch.setattr(freeze_launch, "_successful_command", fake_success)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    evidence = freeze_launch.run_offline_checks()
    assert evidence["passed"] is True
    assert len(observed) == 4
    for environment in observed:
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert environment["TRANSFORMERS_OFFLINE"] == "1"


def test_runtime_environment_rejects_wrong_interpreter_or_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(freeze_launch.sys, "executable", "/wrong/python")
    with pytest.raises(RuntimeError, match="canonical project interpreter"):
        freeze_launch.validate_runtime_environment()

    monkeypatch.setattr(freeze_launch.sys, "executable", str(freeze_launch.CANONICAL_PYTHON))
    incomplete = _environment()
    incomplete["packages"]["peft"] = None  # type: ignore[index]
    monkeypatch.setattr(freeze_launch, "package_versions", lambda: incomplete)
    with pytest.raises(RuntimeError, match="required package"):
        freeze_launch.validate_runtime_environment()


def test_backend_environment_lock_requires_all_seven_packages() -> None:
    observed = backend.package_versions()
    assert set(observed["packages"]) == {
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "numpy",
        "tokenizers",
        "safetensors",
    }
    _validate_environment_lock({"environment_lock": observed})
    incomplete = copy.deepcopy(observed)
    incomplete["packages"].pop("safetensors")
    with pytest.raises(ValueError, match="differ"):
        _validate_environment_lock({"environment_lock": incomplete})
