#!/usr/bin/env python3
"""Fail-closed CPU/static preflight and one-shot MuSiQue v2 launch receipt writer.

The default mode is a dry run.  It never loads a model, initializes CUDA, reads
the sealed-shadow JSONL body, or touches official MuSiQue dev/test/archive
content.  ``--write-receipt`` additionally requires a sidecar-bound independent
red-team approval and creates the canonical receipt and checksum with exclusive
``O_EXCL`` writes.  Existing receipt artifacts are never replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from support_orbit_musique.backend import (
    CANONICAL_MODEL_PATH,
    CANONICAL_OUTPUT_ROOTS,
    CANONICAL_PYTHON,
    CANONICAL_PREPARED_MANIFEST,
    CANONICAL_PROTOCOL,
    CANONICAL_REDTEAM_SIGNOFF,
    CANONICAL_RECEIPT,
    EXPECTED_OUTPUT_ROOT_KEYS,
    EXPECTED_RUNTIME,
    PROJECT_ROOT,
    RECEIPT_SCHEMA,
    RECEIPT_STATUS,
    REDTEAM_SIGNOFF_SCHEMA,
    REDTEAM_SIGNOFF_STATUS,
    REQUIRED_SOURCE_LOCK,
    LoraSpec,
    base_model_manifest,
    canonical_json_sha256,
    package_versions,
    sha256_file,
    tokenizer_manifest,
    validate_launch_receipt,
    validate_output_roots,
    verified_bytes,
    verified_json_object,
)
from support_orbit_musique.protocol import validate_data_release


PROTOCOL_SHA256 = "1a888808d4a09983d0e98d31f1668b2f5faf844dde3b44f4d5e863e555e0c33f"
MANIFEST_SHA256 = "84e97c0eb189d664149822c1244436f0b97e69e3cbca7e16be8186df52a7dfc1"
TRAIN_SHA256 = "641bae95c4eb410229fb8c87a52b24e58628c376c636632d242115ca7e4d5c12"
DEV_SHA256 = "84acfa99def02660aa74d836f07ff4219c0c838c29e19d7d335a126c7840b909"
SCHEDULE_SHA256 = "027c7d3eba760be6956060a487836088911ea5d7746e4fb5586f811c2c7c6ac4"
AUDIT_SHA256 = "462acaada470b7b49b5fbe74c990810714b6a8b9bcaa1b0cd6b97b801a957a6e"
SHADOW_MARKER_SHA256 = "5e6288ef3d7e35ebd9c6c5d3812895f0a812106913b5fb357ae4e96052af5bd5"
SHADOW_WRITER_SHA256 = "1eed3236bb17e79518ae5c3787f3171a37eaea1bfec8f6f46766a59ce2e1d293"
SHADOW_SIDECAR_SHA256 = "81570ec99e9f01f55c61050e46eb3a155d8579574fcbc72733457de0feefecb7"
TRAIN_SIDECAR_SHA256 = "fe158e0fa881d10673c311cdc68acef7f3705a208077ad1cc26f73502c95d81d"
DEV_SIDECAR_SHA256 = "0ec2f1ed612cfaa9566eebe6cc76adad099840a5d8b30304ef071bf6cbdc9d0e"
MODEL_CONFIG_SHA256 = "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba"
TOKENIZER_JSON_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
TOKENIZER_CONFIG_SHA256 = "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3"
CHAT_TEMPLATE_SHA256 = "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326"

REDTEAM_SCHEMA = REDTEAM_SIGNOFF_SCHEMA
REDTEAM_STATUS = REDTEAM_SIGNOFF_STATUS
RUFF_EXECUTABLE = Path("/home/hesong/.local/bin/ruff")

PREPARED_ROOT = PROJECT_ROOT / "prepared_data_v2"
TRAIN_PATH = PREPARED_ROOT / "train.jsonl"
DEV_PATH = PREPARED_ROOT / "dev.jsonl"
AUDIT_PATH = PREPARED_ROOT / "audit.json"
SHADOW_MARKER_PATH = PREPARED_ROOT / "SHADOW_SEALED.json"
SHADOW_BODY_PATH = PREPARED_ROOT / "shadow.sealed.jsonl"
SHADOW_SIDECAR_PATH = PREPARED_ROOT / "shadow.sealed.jsonl.sha256"

GPU_BY_ARM = {"CONTROL": "0", "HopPAIR": "1", "BASE": "2"}
EXCLUDED_GPU_INDEX = "3"
IDLE_MEMORY_CEILING_MIB = 1_024
IDLE_UTILIZATION_CEILING_PERCENT = 0

OUTPUT_ROOTS = CANONICAL_OUTPUT_ROOTS

REQUIRED_REDTEAM_KEYS = {
    "schema_version",
    "status",
    "reviewer_kind",
    "reviewer_id",
    "created_at",
    "reviewed_protocol_sha256",
    "reviewed_manifest_sha256",
    "reviewed_train_sha256",
    "reviewed_dev_sha256",
    "reviewed_schedule_sha256",
    "reviewed_source_lock_sha256",
    "findings",
    "access_boundary",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--write-receipt",
        action="store_true",
        help="exclusively create the canonical receipt after independent sign-off",
    )
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_sha(value: object, description: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{description} must be a lowercase SHA256")
    return str(value)


def _strict_absolute(path: str | Path, description: str) -> Path:
    raw = os.fspath(path)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{description} must be an absolute normalized path")
    if os.path.normpath(str(candidate)) != str(candidate):
        raise ValueError(f"{description} contains a noncanonical spelling")
    return candidate


def _assert_no_symlink_components(
    path: str | Path,
    *,
    description: str,
    allow_missing_tail: bool = False,
) -> Path:
    """Reject a symlink in any existing component without resolving the leaf."""

    candidate = _strict_absolute(path, description)
    current = Path(candidate.anchor)
    missing_seen = False
    for component in candidate.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if not allow_missing_tail:
                raise
            missing_seen = True
            continue
        if missing_seen:
            raise RuntimeError(f"{description} changed during path inspection: {candidate}")
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{description} refuses symlink component: {current}")
    return candidate


def _assert_regular_single_link(path: Path, description: str) -> os.stat_result:
    candidate = _assert_no_symlink_components(path, description=description)
    metadata = os.lstat(candidate)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{description} must be a regular single-link file: {candidate}")
    return metadata


def _assert_disjoint(paths: Mapping[str, Path], description: str) -> None:
    rows = sorted(paths.items())
    for index, (first_name, first) in enumerate(rows):
        for second_name, second in rows[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError(
                    f"{description} paths overlap: {first_name}={first}, "
                    f"{second_name}={second}"
                )


def _canonical_sidecar(
    artifact: Path,
    *,
    expected_artifact_sha256: str,
    expected_sidecar_sha256: str | None = None,
) -> dict[str, Any]:
    sidecar = Path(f"{artifact}.sha256")
    _assert_regular_single_link(sidecar, f"{artifact.name} checksum sidecar")
    payload, sidecar_sha = verified_bytes(sidecar)
    expected = f"{expected_artifact_sha256}  {artifact.name}\n".encode("ascii")
    if payload != expected:
        raise ValueError(f"noncanonical checksum sidecar: {sidecar}")
    if expected_sidecar_sha256 is not None and sidecar_sha != expected_sidecar_sha256:
        raise ValueError(f"checksum sidecar hash mismatch: {sidecar}")
    return {"path": str(sidecar), "sha256": sidecar_sha, "target_sha256": expected_artifact_sha256}


def _load_sidecar_bound_json(
    path: Path,
    *,
    expected_sha256: str,
    description: str,
) -> dict[str, Any]:
    _assert_regular_single_link(path, description)
    _canonical_sidecar(path, expected_artifact_sha256=expected_sha256)
    value, digest = verified_json_object(
        path,
        expected_sha256=expected_sha256,
        description=description,
    )
    if digest != expected_sha256:
        raise AssertionError(f"{description} digest drift")
    return value


def _production_source_lock() -> dict[str, str]:
    required_new = {"freeze_launch.py", "tests/test_launch_receipt.py"}
    if not required_new.issubset(REQUIRED_SOURCE_LOCK):
        raise RuntimeError(
            "backend.REQUIRED_SOURCE_LOCK must include freeze_launch.py and "
            "tests/test_launch_receipt.py before any freeze"
        )
    lock: dict[str, str] = {}
    for relative in sorted(REQUIRED_SOURCE_LOCK):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"invalid source-lock member: {relative}")
        source = PROJECT_ROOT / relative
        _assert_regular_single_link(source, f"source-lock member {relative}")
        lock[relative] = sha256_file(source)
    return lock


def _require_same_source_lock(
    expected: Mapping[str, str],
    observed: Mapping[str, str],
    stage: str,
) -> None:
    if dict(observed) != dict(expected):
        changed = sorted(
            key
            for key in set(expected) | set(observed)
            if expected.get(key) != observed.get(key)
        )
        raise RuntimeError(f"production source changed {stage}: {changed}")


def _inspect_shadow_metadata(
    *,
    body_path: Path,
    marker: Mapping[str, Any],
    sidecar_path: Path,
    sidecar_expected_sha256: str,
) -> dict[str, Any]:
    """Inspect the sealed body with lstat only; never open or hash its bytes."""

    metadata = _assert_regular_single_link(body_path, "sealed-shadow body stat target")
    if (
        marker.get("sealed") is not True
        or marker.get("training_read_allowed") is not False
        or marker.get("artifact") != body_path.name
        or marker.get("orbit_count") != 400
        or marker.get("record_count") != 1_200
        or marker.get("bytes") != metadata.st_size
        or marker.get("sha256") != SHADOW_WRITER_SHA256
    ):
        raise ValueError("sealed-shadow marker or lstat metadata drift")
    _assert_regular_single_link(sidecar_path, "sealed-shadow checksum metadata")
    payload, digest = verified_bytes(sidecar_path)
    expected = f"{SHADOW_WRITER_SHA256}  {body_path.name}\n".encode("ascii")
    if payload != expected or digest != sidecar_expected_sha256:
        raise ValueError("sealed-shadow checksum metadata drift")
    return {
        "body_path": str(body_path),
        "body_size_bytes": metadata.st_size,
        "body_opened": False,
        "writer_sha256": SHADOW_WRITER_SHA256,
        "marker_path": str(SHADOW_MARKER_PATH),
        "marker_sha256": SHADOW_MARKER_SHA256,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": digest,
    }


def validate_frozen_inputs() -> dict[str, Any]:
    """Validate v2 protocol/data using public metadata; never open train/dev bodies."""

    for name, path in {
        "protocol": CANONICAL_PROTOCOL,
        "prepared_manifest": CANONICAL_PREPARED_MANIFEST,
        "train": TRAIN_PATH,
        "dev": DEV_PATH,
        "audit": AUDIT_PATH,
        "shadow_marker": SHADOW_MARKER_PATH,
        "shadow_sidecar": SHADOW_SIDECAR_PATH,
    }.items():
        _assert_regular_single_link(path, name)

    protocol = _load_sidecar_bound_json(
        CANONICAL_PROTOCOL,
        expected_sha256=PROTOCOL_SHA256,
        description="frozen MuSiQue v2 protocol",
    )
    manifest, manifest_sha = verified_json_object(
        CANONICAL_PREPARED_MANIFEST,
        expected_sha256=MANIFEST_SHA256,
        description="prepared_data_v2 manifest",
    )
    audit, audit_sha = verified_json_object(
        AUDIT_PATH,
        expected_sha256=AUDIT_SHA256,
        description="prepared_data_v2 audit",
    )
    marker, marker_sha = verified_json_object(
        SHADOW_MARKER_PATH,
        expected_sha256=SHADOW_MARKER_SHA256,
        description="sealed-shadow marker",
    )
    if (manifest_sha, audit_sha, marker_sha) != (
        MANIFEST_SHA256,
        AUDIT_SHA256,
        SHADOW_MARKER_SHA256,
    ):
        raise AssertionError("v2 public metadata digest drift")

    train_sidecar = _canonical_sidecar(
        TRAIN_PATH,
        expected_artifact_sha256=TRAIN_SHA256,
        expected_sidecar_sha256=TRAIN_SIDECAR_SHA256,
    )
    dev_sidecar = _canonical_sidecar(
        DEV_PATH,
        expected_artifact_sha256=DEV_SHA256,
        expected_sidecar_sha256=DEV_SIDECAR_SHA256,
    )
    if (
        protocol.get("schema_version") != "support-orbit-musique.protocol/v2"
        or protocol.get("protocol_status") != "FROZEN_PRE_GPU"
        or protocol.get("launch_status") != "STOP_BEFORE_GPU"
        or protocol.get("prepared_data_lock", {}).get("expected_manifest_sha256")
        != MANIFEST_SHA256
    ):
        raise ValueError("v2 protocol status or manifest lock drift")
    artifacts = manifest.get("artifact_sha256", {})
    expected_artifacts = {
        "train.jsonl": TRAIN_SHA256,
        "dev.jsonl": DEV_SHA256,
        "audit.json": AUDIT_SHA256,
        "SHADOW_SEALED.json": SHADOW_MARKER_SHA256,
        "shadow.sealed.jsonl": SHADOW_WRITER_SHA256,
        "shadow.sealed.jsonl.sha256": SHADOW_SIDECAR_SHA256,
    }
    if not isinstance(artifacts, dict) or any(
        artifacts.get(name) != digest for name, digest in expected_artifacts.items()
    ):
        raise ValueError("v2 manifest artifact lock drift")

    release = validate_data_release(
        protocol=protocol,
        manifest_path=CANONICAL_PREPARED_MANIFEST,
        audit_path=AUDIT_PATH,
        shadow_marker_path=SHADOW_MARKER_PATH,
        verified={
            "manifest": manifest,
            "audit": audit,
            "shadow_marker": marker,
            "sha256": {
                "manifest.json": MANIFEST_SHA256,
                "audit.json": AUDIT_SHA256,
                "SHADOW_SEALED.json": SHADOW_MARKER_SHA256,
                "train.jsonl": TRAIN_SHA256,
                "dev.jsonl": DEV_SHA256,
            },
        },
    )
    if release.get("passed") is not True or release.get("shadow_body_opened") is not False:
        raise ValueError("prepared_data_v2 release validation failed closed")

    shadow = _inspect_shadow_metadata(
        body_path=SHADOW_BODY_PATH,
        marker=marker,
        sidecar_path=SHADOW_SIDECAR_PATH,
        sidecar_expected_sha256=SHADOW_SIDECAR_SHA256,
    )
    return {
        "protocol": {"path": str(CANONICAL_PROTOCOL), "sha256": PROTOCOL_SHA256},
        "prepared_manifest": {
            "path": str(CANONICAL_PREPARED_MANIFEST),
            "sha256": MANIFEST_SHA256,
        },
        "train": {
            "path": str(TRAIN_PATH),
            "sha256": TRAIN_SHA256,
            "orbits": 1_920,
            "records": 5_760,
        },
        "dev": {
            "path": str(DEV_PATH),
            "sha256": DEV_SHA256,
            "orbits": 400,
            "records": 1_200,
        },
        # This pre-GPU freezer binds the previously frozen seed-17 digest but
        # does not open train.jsonl to reconstruct it.  Each training arm must
        # independently hash+parse train and recompute this schedule at use.
        "schedule_sha256": SCHEDULE_SHA256,
        "release_validation": release,
        "train_sidecar": train_sidecar,
        "dev_sidecar": dev_sidecar,
        "sealed_shadow_metadata": shadow,
        "body_access": {
            "train_opened": False,
            "dev_opened": False,
            "shadow_opened": False,
        },
    }


def _token_id(tokenizer_config: Mapping[str, Any], token_field: str) -> int:
    token = tokenizer_config.get(token_field)
    decoder = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(token, str) or not isinstance(decoder, dict):
        raise ValueError(f"tokenizer lacks {token_field} semantic identity")
    matches = [
        int(token_id)
        for token_id, value in decoder.items()
        if isinstance(value, dict) and value.get("content") == token
    ]
    if len(matches) != 1:
        raise ValueError(f"tokenizer {token_field} does not map to exactly one added token")
    return matches[0]


def validate_static_model_files() -> dict[str, Any]:
    """Hash local model/tokenizer files without importing Transformers or Torch."""

    _assert_no_symlink_components(CANONICAL_MODEL_PATH, description="canonical model root")
    model = base_model_manifest(CANONICAL_MODEL_PATH)
    tokenizer = tokenizer_manifest(CANONICAL_MODEL_PATH)
    for manifest_name, manifest in (("base model", model), ("tokenizer", tokenizer)):
        for entry in manifest["files"]:
            _assert_regular_single_link(
                CANONICAL_MODEL_PATH / entry["path"],
                f"{manifest_name} member {entry['path']}",
            )

    model_files = {entry["path"]: entry["sha256"] for entry in model["files"]}
    tokenizer_files = {entry["path"]: entry["sha256"] for entry in tokenizer["files"]}
    if model_files.get("config.json") != MODEL_CONFIG_SHA256:
        raise ValueError("base-model config differs from the tokenizer-audited v2 identity")
    if (
        tokenizer_files.get("tokenizer.json") != TOKENIZER_JSON_SHA256
        or tokenizer_files.get("tokenizer_config.json") != TOKENIZER_CONFIG_SHA256
    ):
        raise ValueError("tokenizer files differ from the prepared-data v2 audit identity")

    tokenizer_config_path = CANONICAL_MODEL_PATH / "tokenizer_config.json"
    tokenizer_config, tokenizer_config_sha = verified_json_object(
        tokenizer_config_path,
        expected_sha256=TOKENIZER_CONFIG_SHA256,
        description="canonical tokenizer configuration",
    )
    if tokenizer_config_sha != tokenizer_files["tokenizer_config.json"]:
        raise RuntimeError("tokenizer configuration changed during static identity validation")
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("canonical tokenizer has no chat template")
    eos_token_id = _token_id(tokenizer_config, "eos_token")
    pad_token_id = _token_id(tokenizer_config, "pad_token")
    chat_template_sha = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if chat_template_sha != CHAT_TEMPLATE_SHA256:
        raise ValueError("tokenizer chat template differs from the prepared-data v2 contract")
    exact_tokenizer = {
        "path": str(CANONICAL_MODEL_PATH),
        "sha256": tokenizer["sha256"],
        "chat_template_sha256": chat_template_sha,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    return {
        "base_model_binding": {
            "path": str(CANONICAL_MODEL_PATH),
            "sha256": model["sha256"],
        },
        "tokenizer_binding": exact_tokenizer,
        "base_model_manifest": model,
        "tokenizer_manifest": tokenizer,
        "model_loaded": False,
        "cuda_initialized": False,
    }


def _run_command(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    timeout: int = 1_800,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv),
        cwd=PROJECT_ROOT,
        env=dict(environment) if environment is not None else None,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _successful_command(argv: Sequence[str], *, environment: Mapping[str, str]) -> dict[str, Any]:
    completed = _run_command(argv, environment=environment)
    if completed.returncode != 0:
        stdout_tail = completed.stdout.decode("utf-8", errors="replace")[-2_000:]
        stderr_tail = completed.stderr.decode("utf-8", errors="replace")[-2_000:]
        raise RuntimeError(
            f"offline preflight command failed ({completed.returncode}): {list(argv)!r}\n"
            f"stdout tail:\n{stdout_tail}\nstderr tail:\n{stderr_tail}"
        )
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout_bytes": len(completed.stdout),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_bytes": len(completed.stderr),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }


def validate_runtime_environment() -> dict[str, Any]:
    if Path(sys.executable) != CANONICAL_PYTHON:
        raise RuntimeError(
            "launch freeze must run with the canonical project interpreter: "
            f"{CANONICAL_PYTHON}; observed {sys.executable}"
        )
    environment = package_versions()
    required = {
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "numpy",
        "tokenizers",
        "safetensors",
    }
    packages = environment.get("packages")
    if (
        environment.get("python", {}).get("executable") != str(CANONICAL_PYTHON)
        or not isinstance(packages, dict)
        or set(packages) != required
        or any(not isinstance(value, str) or not value for value in packages.values())
        or packages.get("transformers") != "4.57.3"
        or packages.get("numpy") != "2.4.6"
    ):
        raise RuntimeError("canonical environment lacks an exact required package identity")
    return environment


def run_offline_checks() -> dict[str, Any]:
    """Run the frozen full test, lint, and syntax checks with CUDA hidden."""

    if not RUFF_EXECUTABLE.is_file() or RUFF_EXECUTABLE.is_symlink():
        raise FileNotFoundError(f"canonical ruff executable is absent: {RUFF_EXECUTABLE}")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    python_sources = [
        str(PROJECT_ROOT / relative)
        for relative in sorted(REQUIRED_SOURCE_LOCK)
        if relative.endswith(".py")
    ]
    import_smoke = (
        "import torch, transformers, peft, accelerate, numpy, tokenizers, safetensors; "
        "assert not torch.cuda.is_initialized()"
    )
    commands = [
        [sys.executable, "-c", import_smoke],
        [sys.executable, "-m", "pytest", "-q"],
        [str(RUFF_EXECUTABLE), "check", "."],
        [sys.executable, "-m", "compileall", "-q", "-f", *python_sources],
    ]
    evidence = [_successful_command(command, environment=environment) for command in commands]
    return {
        "passed": True,
        "returncode": 0,
        "cuda_visible_devices": "",
        "offline": True,
        "commands": evidence,
    }


def _parse_gpu_inventory(payload: bytes) -> dict[str, dict[str, Any]]:
    text = payload.decode("utf-8", errors="strict")
    inventory: dict[str, dict[str, Any]] = {}
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(row) != 7:
            raise ValueError(f"unexpected nvidia-smi inventory row: {row!r}")
        index, uuid, name, driver, capability, memory_used, utilization = (
            item.strip() for item in row
        )
        if index in inventory or not index.isdigit():
            raise ValueError("duplicate or invalid GPU index")
        match = re.fullmatch(r"(\d+)\.(\d+)", capability)
        if match is None:
            raise ValueError(f"invalid GPU compute capability: {capability}")
        inventory[index] = {
            "index": index,
            "uuid": uuid,
            "name": name,
            "driver_version": driver,
            "compute_capability": [int(match.group(1)), int(match.group(2))],
            "memory_used_mib": int(memory_used),
            "utilization_percent": int(utilization),
        }
    return inventory


def _parse_compute_processes(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8", errors="strict")
    rows: list[dict[str, str]] = []
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(row) != 4:
            raise ValueError(f"unexpected nvidia-smi process row: {row!r}")
        uuid, pid, process_name, memory_used = (item.strip() for item in row)
        rows.append(
            {
                "uuid": uuid,
                "pid": pid,
                "process_name": process_name,
                "memory_used_mib": memory_used,
            }
        )
    return rows


def _driver_cuda_version(payload: bytes) -> str:
    text = payload.decode("utf-8", errors="strict")
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match is None:
        raise ValueError("nvidia-smi did not report a CUDA version")
    return match.group(1)


def _torch_cuda_build_version() -> str:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    code = "import torch; assert not torch.cuda.is_initialized(); print(torch.version.cuda or '')"
    result = _run_command([sys.executable, "-c", code], environment=environment, timeout=30)
    if result.returncode != 0:
        raise RuntimeError("canonical Torch CUDA build identity query failed")
    value = result.stdout.decode("ascii", errors="strict").strip()
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) is None:
        raise ValueError("canonical Torch package did not report a CUDA build version")
    return value


def validate_hardware() -> dict[str, Any]:
    """Bind idle same-model GPUs 0/1/2; observe but never assign GPU 3."""

    inventory_command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    processes_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    inventory_result = _run_command(inventory_command, timeout=30)
    processes_result = _run_command(processes_command, timeout=30)
    overview_result = _run_command(["nvidia-smi"], timeout=30)
    for name, result in {
        "inventory": inventory_result,
        "compute-process": processes_result,
        "overview": overview_result,
    }.items():
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi {name} query failed closed: {result.returncode}")
    inventory = _parse_gpu_inventory(inventory_result.stdout)
    processes = _parse_compute_processes(processes_result.stdout)
    driver_cuda_version = _driver_cuda_version(overview_result.stdout)
    torch_cuda_version = _torch_cuda_build_version()
    required = {"0", "1", "2", EXCLUDED_GPU_INDEX}
    if not required.issubset(inventory):
        raise ValueError(f"GPU inventory lacks required indices: {sorted(required - set(inventory))}")
    selected = [inventory[index] for index in ("0", "1", "2")]
    if len({entry["name"] for entry in selected}) != 1:
        raise ValueError("GPU0/1/2 are not the same model")
    if len({entry["driver_version"] for entry in selected}) != 1:
        raise ValueError("GPU0/1/2 driver identities differ")
    if len({tuple(entry["compute_capability"]) for entry in selected}) != 1:
        raise ValueError("GPU0/1/2 compute capabilities differ")
    selected_uuids = {entry["uuid"] for entry in selected}
    if len(selected_uuids) != 3 or any(
        not uuid.startswith(("GPU-", "MIG-")) for uuid in selected_uuids
    ):
        raise ValueError("GPU0/1/2 UUID identities are invalid or duplicated")
    busy_processes = [row for row in processes if row["uuid"] in selected_uuids]
    if busy_processes:
        raise ValueError("GPU0/1/2 have active compute processes")
    for entry in selected:
        if (
            entry["memory_used_mib"] > IDLE_MEMORY_CEILING_MIB
            or entry["utilization_percent"] > IDLE_UTILIZATION_CEILING_PERCENT
        ):
            raise ValueError(f"GPU{entry['index']} is not idle: {entry}")

    hardware_lock: dict[str, dict[str, Any]] = {}
    observations: dict[str, dict[str, Any]] = {}
    for arm, index in GPU_BY_ARM.items():
        entry = inventory[index]
        hardware_lock[arm] = {
            # Runtime selection is pinned by stable UUID, never by a mutable
            # CUDA/NVML ordinal.  The physical index remains preflight evidence.
            "cuda_visible_devices": entry["uuid"],
            "uuid": entry["uuid"],
            "name": entry["name"],
            "driver_version": entry["driver_version"],
            "driver_cuda_version": driver_cuda_version,
            "torch_cuda_version": torch_cuda_version,
            "compute_capability": entry["compute_capability"],
        }
        observations[arm] = {
            "index": index,
            "memory_used_mib": entry["memory_used_mib"],
            "utilization_percent": entry["utilization_percent"],
            "compute_processes": 0,
            "idle": True,
        }
    return {
        "hardware_lock": hardware_lock,
        "observations": observations,
        "same_model": True,
        "idle_memory_ceiling_mib": IDLE_MEMORY_CEILING_MIB,
        "idle_utilization_ceiling_percent": IDLE_UTILIZATION_CEILING_PERCENT,
        "excluded_gpu": {
            "index": EXCLUDED_GPU_INDEX,
            "uuid": inventory[EXCLUDED_GPU_INDEX]["uuid"],
            "present": True,
            "assigned": False,
        },
        "model_loaded": False,
        "cuda_initialized_by_freezer": False,
    }


def validate_canonical_output_roots() -> dict[str, str]:
    if set(OUTPUT_ROOTS) != EXPECTED_OUTPUT_ROOT_KEYS:
        raise ValueError("canonical output-root key set drift")
    paths: dict[str, Path] = {}
    for name, raw in OUTPUT_ROOTS.items():
        path = _assert_no_symlink_components(
            raw,
            description=f"canonical output root {name}",
            allow_missing_tail=True,
        )
        if path.exists():
            raise FileExistsError(f"canonical output root is not fresh: {path}")
        paths[name] = path
    _assert_disjoint(paths, "canonical output-root")
    protected = {
        "receipt": CANONICAL_RECEIPT,
        "protocol": CANONICAL_PROTOCOL,
        "prepared_manifest": CANONICAL_PREPARED_MANIFEST,
        "train": TRAIN_PATH,
        "dev": DEV_PATH,
        "model": CANONICAL_MODEL_PATH,
    }
    for output_name, output in paths.items():
        for protected_name, protected_path in protected.items():
            if (
                output == protected_path
                or output in protected_path.parents
                or protected_path in output.parents
            ):
                raise ValueError(
                    f"output root {output_name} overlaps protected {protected_name}: {output}"
                )
    probe = {"output_roots": dict(OUTPUT_ROOTS)}
    validate_output_roots(probe)
    return dict(OUTPUT_ROOTS)


def _redteam_expected(source_lock_sha256: str) -> dict[str, str]:
    return {
        "reviewed_protocol_sha256": PROTOCOL_SHA256,
        "reviewed_manifest_sha256": MANIFEST_SHA256,
        "reviewed_train_sha256": TRAIN_SHA256,
        "reviewed_dev_sha256": DEV_SHA256,
        "reviewed_schedule_sha256": SCHEDULE_SHA256,
        "reviewed_source_lock_sha256": source_lock_sha256,
    }


def validate_redteam_signoff(
    *,
    source_lock_sha256: str,
    required: bool,
) -> dict[str, Any] | None:
    path = CANONICAL_REDTEAM_SIGNOFF
    if not path.exists():
        if path.is_symlink():
            raise ValueError(f"independent red-team sign-off may not be a symlink: {path}")
        if required:
            raise FileNotFoundError(
                "--write-receipt requires the canonical independent red-team sign-off: "
                f"{path}"
            )
        return None
    signoff = _load_sidecar_bound_json(
        path,
        expected_sha256=_canonical_sidecar_digest(path),
        description="independent red-team sign-off",
    )
    if set(signoff) != REQUIRED_REDTEAM_KEYS:
        raise ValueError("independent red-team sign-off fields drift")
    if (
        signoff.get("schema_version") != REDTEAM_SCHEMA
        or signoff.get("status") != REDTEAM_STATUS
        or signoff.get("reviewer_kind") != "independent_redteam"
        or not isinstance(signoff.get("reviewer_id"), str)
        or not signoff["reviewer_id"].strip()
        or not isinstance(signoff.get("created_at"), str)
        or signoff.get("findings") != {"P0": 0, "P1": 0}
        or signoff.get("access_boundary")
        != {
            "withdrawn_v1_data_body_opened": False,
            "shadow_body_opened": False,
            "official_dev_opened": False,
            "official_test_opened": False,
            "official_archive_opened": False,
            "gpu_model_loaded": False,
        }
    ):
        raise ValueError("independent red-team approval contract failed")
    for key, expected in _redteam_expected(source_lock_sha256).items():
        if signoff.get(key) != expected:
            raise ValueError(f"independent red-team binding mismatch: {key}")
    _, digest = verified_json_object(path, description="independent red-team sign-off")
    return {"path": str(path), "sha256": digest, "reviewer_id": signoff["reviewer_id"]}


def _canonical_sidecar_digest(path: Path) -> str:
    """Read only a checksum sidecar to obtain a JSON artifact's expected hash."""

    sidecar = Path(f"{path}.sha256")
    _assert_regular_single_link(sidecar, f"{path.name} checksum sidecar")
    payload, _ = verified_bytes(sidecar)
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"non-ASCII checksum sidecar: {sidecar}") from exc
    suffix = f"  {path.name}\n"
    if len(text) != 64 + len(suffix) or not text.endswith(suffix):
        raise ValueError(f"noncanonical checksum sidecar: {sidecar}")
    return _assert_sha(text[:64], f"{path.name} checksum")


def _compose_receipt(
    *,
    data: Mapping[str, Any],
    model: Mapping[str, Any],
    source_lock: Mapping[str, str],
    test_evidence: Mapping[str, Any],
    hardware: Mapping[str, Any],
    output_roots: Mapping[str, str],
    redteam: Mapping[str, Any] | None,
    environment_lock: Mapping[str, Any],
) -> dict[str, Any]:
    source_lock_sha = canonical_json_sha256(dict(source_lock))
    ready = redteam is not None
    runtime = dict(EXPECTED_RUNTIME)
    runtime["generation"] = {
        "batch_size": 1,
        "max_new_tokens": 128,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "length_preflight_status": "PASS",
    }
    runtime["gpu_by_arm"] = {
        arm: hardware["hardware_lock"][arm]["uuid"] for arm in ("CONTROL", "HopPAIR", "BASE")
    }
    gates = [
        {"name": "protocol_v2_exact", "status": "PASS"},
        {"name": "prepared_data_v2_release", "status": "PASS"},
        {"name": "train_dev_and_seed17_schedule_exact", "status": "PASS"},
        {"name": "model_tokenizer_content_identity", "status": "PASS"},
        {"name": "complete_production_source_lock", "status": "PASS"},
        {"name": "offline_cpu_tests_ruff_compileall", "status": "PASS"},
        {"name": "canonical_disjoint_fresh_output_roots", "status": "PASS"},
        {"name": "same_model_idle_gpu0_gpu1_gpu2_gpu3_excluded", "status": "PASS"},
        {"name": "forbidden_access_boundary", "status": "PASS"},
        {
            "name": "independent_redteam_P0_P1_zero",
            "status": "PASS" if ready else "PENDING",
        },
    ]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS if ready else "STOP_PENDING_INDEPENDENT_REDTEAM",
        "canonical_path": str(CANONICAL_RECEIPT),
        "created_at": utc_now(),
        "exact_bindings": {
            "protocol": dict(data["protocol"]),
            "prepared_manifest": dict(data["prepared_manifest"]),
            "train": dict(data["train"]),
            "dev": dict(data["dev"]),
            "schedule_sha256": data["schedule_sha256"],
            "base_model": dict(model["base_model_binding"]),
            "tokenizer": dict(model["tokenizer_binding"]),
            "source_lock_sha256": source_lock_sha,
        },
        "source_lock": dict(source_lock),
        "runtime": runtime,
        "lora": LoraSpec().to_dict(),
        "objectives": {
            "CONTROL": {"kl_weight": 0.0, "flip_weight": 0.0},
            "HopPAIR": {"kl_weight": 0.1, "flip_weight": 0.2, "flip_margin": 2.0},
        },
        "hardware_lock": dict(hardware["hardware_lock"]),
        "hardware_preflight": {
            key: value for key, value in hardware.items() if key != "hardware_lock"
        },
        "environment_lock": dict(environment_lock),
        "output_roots": dict(output_roots),
        "forbidden_access": {
            "withdrawn_v1": True,
            "shadow": True,
            "official_dev": True,
            "official_test": True,
        },
        "access_evidence": {
            "withdrawn_v1_data_body_opened": False,
            "train_body_opened_pre_receipt": False,
            "dev_body_opened_pre_training": False,
            "shadow_body_opened": False,
            "official_dev_opened": False,
            "official_test_opened": False,
            "official_archive_opened": False,
            "network_evaluation_data_used": False,
            "gpu_model_loaded": False,
            "cuda_initialized_by_freezer": False,
        },
        "sealed_shadow_metadata": dict(data["sealed_shadow_metadata"]),
        "static_model_evidence": {
            "base_model_manifest": model["base_model_manifest"],
            "tokenizer_manifest": model["tokenizer_manifest"],
            "model_loaded": False,
            "cuda_initialized": False,
        },
        "data_release_evidence": {
            "validation": data["release_validation"],
            "train_sidecar": data["train_sidecar"],
            "dev_sidecar": data["dev_sidecar"],
        },
        "test_evidence": dict(test_evidence),
        "independent_redteam_signoff": dict(redteam) if redteam is not None else None,
        "pre_gpu_gates": gates,
    }
    receipt["evidence_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key not in {"created_at", "evidence_sha256"}
        }
    )
    return receipt


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_create(path: Path, payload: bytes) -> tuple[int, int]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("one-shot receipt creation requires O_NOFOLLOW")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        metadata_before = os.fstat(handle.fileno())
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        metadata_after = os.fstat(handle.fileno())
    if (
        metadata_before.st_dev != metadata_after.st_dev
        or metadata_before.st_ino != metadata_after.st_ino
        or metadata_after.st_size != len(payload)
    ):
        raise RuntimeError(f"exclusive artifact changed during creation: {path}")
    return metadata_after.st_dev, metadata_after.st_ino


def _unlink_if_same(path: Path, identity: tuple[int, int]) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if (metadata.st_dev, metadata.st_ino) != identity:
        raise RuntimeError(f"refusing to remove a changed failed artifact: {path}")
    os.unlink(path)


def _write_exclusive_receipt_pair(path: Path, receipt: Mapping[str, Any]) -> str:
    """Create receipt then sidecar without overwriting either destination."""

    _assert_no_symlink_components(path.parent, description="receipt directory")
    sidecar = Path(f"{path}.sha256")
    if path.exists() or path.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError("canonical receipt or sidecar already exists; overwrite is forbidden")
    receipt_bytes = _json_bytes(receipt)
    digest = hashlib.sha256(receipt_bytes).hexdigest()
    sidecar_bytes = f"{digest}  {path.name}\n".encode("ascii")
    receipt_identity: tuple[int, int] | None = None
    sidecar_identity: tuple[int, int] | None = None
    try:
        receipt_identity = _exclusive_create(path, receipt_bytes)
        sidecar_identity = _exclusive_create(sidecar, sidecar_bytes)
    except BaseException:
        if sidecar_identity is not None:
            _unlink_if_same(sidecar, sidecar_identity)
        if receipt_identity is not None:
            _unlink_if_same(path, receipt_identity)
        raise
    return digest


def _ensure_receipt_absent() -> None:
    sidecar = Path(f"{CANONICAL_RECEIPT}.sha256")
    if (
        CANONICAL_RECEIPT.exists()
        or CANONICAL_RECEIPT.is_symlink()
        or sidecar.exists()
        or sidecar.is_symlink()
    ):
        raise FileExistsError(
            "canonical launch receipt state already exists; this one-shot freezer never overwrites"
        )


def freeze(*, write_receipt: bool) -> dict[str, Any]:
    """Execute preflight; optionally perform the independently approved write."""

    _ensure_receipt_absent()
    environment_lock = validate_runtime_environment()
    source_before_tests = _production_source_lock()
    tests = run_offline_checks()
    source_after_tests = _production_source_lock()
    _require_same_source_lock(source_before_tests, source_after_tests, "during offline checks")

    data = validate_frozen_inputs()
    model = validate_static_model_files()
    outputs = validate_canonical_output_roots()
    hardware = validate_hardware()
    source_lock_sha = canonical_json_sha256(source_after_tests)
    redteam = validate_redteam_signoff(
        source_lock_sha256=source_lock_sha,
        required=write_receipt,
    )
    tests = {
        **tests,
        "source_lock_before_tests_sha256": canonical_json_sha256(source_before_tests),
        "source_lock_after_tests_sha256": canonical_json_sha256(source_after_tests),
    }
    receipt = _compose_receipt(
        data=data,
        model=model,
        source_lock=source_after_tests,
        test_evidence=tests,
        hardware=hardware,
        output_roots=outputs,
        redteam=redteam,
        environment_lock=environment_lock,
    )
    if not write_receipt:
        return {
            "mode": "DRY_RUN",
            "written": False,
            "status": receipt["status"],
            "canonical_receipt": str(CANONICAL_RECEIPT),
            "source_lock_sha256": source_lock_sha,
            "protocol_sha256": PROTOCOL_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "train_sha256": TRAIN_SHA256,
            "dev_sha256": DEV_SHA256,
            "schedule_sha256": SCHEDULE_SHA256,
            "redteam_signoff": redteam,
            "pre_gpu_gates": receipt["pre_gpu_gates"],
        }

    if receipt["status"] != RECEIPT_STATUS or redteam is None:
        raise RuntimeError("receipt write remains forbidden without a validated independent sign-off")
    source_before_write = _production_source_lock()
    _require_same_source_lock(source_after_tests, source_before_write, "before receipt write")
    _ensure_receipt_absent()
    receipt["test_evidence"]["source_lock_before_write_sha256"] = canonical_json_sha256(
        source_before_write
    )
    receipt["evidence_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key not in {"created_at", "evidence_sha256"}
        }
    )
    receipt_digest = _write_exclusive_receipt_pair(CANONICAL_RECEIPT, receipt)
    sidecar_path = Path(f"{CANONICAL_RECEIPT}.sha256")
    receipt_identity = (os.lstat(CANONICAL_RECEIPT).st_dev, os.lstat(CANONICAL_RECEIPT).st_ino)
    sidecar_identity = (os.lstat(sidecar_path).st_dev, os.lstat(sidecar_path).st_ino)
    try:
        validated = validate_launch_receipt(CANONICAL_RECEIPT, purpose="metadata")
        if validated["_validated"]["sha256"] != receipt_digest:
            raise RuntimeError("post-write receipt validation digest drift")
    except BaseException:
        _unlink_if_same(sidecar_path, sidecar_identity)
        _unlink_if_same(CANONICAL_RECEIPT, receipt_identity)
        raise
    return {
        "mode": "WRITE_RECEIPT",
        "written": True,
        "status": RECEIPT_STATUS,
        "canonical_receipt": str(CANONICAL_RECEIPT),
        "receipt_sha256": receipt_digest,
        "sidecar": str(sidecar_path),
        "source_lock_sha256": source_lock_sha,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = freeze(write_receipt=args.write_receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
