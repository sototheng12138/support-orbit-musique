"""Local Qwen/PEFT backend, cryptographic identities, and launch binding.

The module intentionally keeps Transformers and PEFT imports lazy.  Receipt,
data, protocol, source, and static model identities are checked before a CUDA
model is loaded.  The one canonical launch receipt is the only authority that
can start training or development generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MODEL_PATH = Path("/home/hesong/AI-Agent-Projects/models/Qwen3-4B-Instruct-2507")
CANONICAL_PYTHON = Path("/home/hesong/AI-Agent-Projects/.venv-schematune/bin/python")
CANONICAL_RECEIPT = PROJECT_ROOT / "protocols" / "pilot_v2_launch_receipt.json"
CANONICAL_PROTOCOL = PROJECT_ROOT / "protocols" / "support_orbit_pilot_v2.json"
CANONICAL_PREPARED_MANIFEST = PROJECT_ROOT / "prepared_data_v2" / "manifest.json"
CANONICAL_REDTEAM_SIGNOFF = PROJECT_ROOT / "protocols" / "pilot_v2_redteam_signoff.json"
RECEIPT_SCHEMA = "support-orbit-musique.launch-receipt.v2"
RECEIPT_STATUS = "READY_FOR_GPU"
REDTEAM_SIGNOFF_SCHEMA = "support-orbit-musique.independent-redteam-signoff.v2"
REDTEAM_SIGNOFF_STATUS = "APPROVED_FOR_RECEIPT_CREATION"
SCHEDULE_ALGORITHM = "sha256-sort-v1"
EXPECTED_OUTPUT_ROOT_KEYS = frozenset(
    {
        "control_train",
        "hoppair_train",
        "base_dev_generation",
        "control_dev_generation",
        "hoppair_dev_generation",
        "base_dev_evaluation",
        "control_dev_evaluation",
        "hoppair_dev_evaluation",
        "dev_comparison",
    }
)
CANONICAL_OUTPUT_ROOTS = {
    "control_train": str(PROJECT_ROOT / "runs" / "pilot_v2_seed17" / "control_train"),
    "hoppair_train": str(PROJECT_ROOT / "runs" / "pilot_v2_seed17" / "hoppair_train"),
    "base_dev_generation": str(
        PROJECT_ROOT / "runs" / "pilot_v2_seed17" / "base_dev_generation"
    ),
    "control_dev_generation": str(
        PROJECT_ROOT / "runs" / "pilot_v2_seed17" / "control_dev_generation"
    ),
    "hoppair_dev_generation": str(
        PROJECT_ROOT / "runs" / "pilot_v2_seed17" / "hoppair_dev_generation"
    ),
    "base_dev_evaluation": str(
        PROJECT_ROOT / "runs" / "pilot_v2_seed17" / "base_dev_evaluation"
    ),
    "control_dev_evaluation": str(
        PROJECT_ROOT / "runs" / "pilot_v2_seed17" / "control_dev_evaluation"
    ),
    "hoppair_dev_evaluation": str(
        PROJECT_ROOT / "runs" / "pilot_v2_seed17" / "hoppair_dev_evaluation"
    ),
    "dev_comparison": str(PROJECT_ROOT / "runs" / "pilot_v2_seed17" / "dev_comparison"),
}

ALL_LINEAR_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# A receipt may not omit any executable, objective, formatter, evaluator, or
# synthetic test that was used to authorize the run.  Keeping this set here
# makes omissions fail closed rather than silently accepting a partial lock.
REQUIRED_SOURCE_LOCK = frozenset(
    {
        "pyproject.toml",
        "build_data.py",
        "compare_dev.py",
        "evaluate_dev.py",
        "freeze_launch.py",
        "generate_dev.py",
        "train_arm.py",
        "support_orbit_musique/__init__.py",
        "support_orbit_musique/audit.py",
        "support_orbit_musique/backend.py",
        "support_orbit_musique/cli.py",
        "support_orbit_musique/data.py",
        "support_orbit_musique/evaluation.py",
        "support_orbit_musique/formatting.py",
        "support_orbit_musique/generate.py",
        "support_orbit_musique/losses.py",
        "support_orbit_musique/metrics.py",
        "support_orbit_musique/official_adapter.py",
        "support_orbit_musique/parsing.py",
        "support_orbit_musique/protocol.py",
        "support_orbit_musique/sft.py",
        "support_orbit_musique/train.py",
        "support_orbit_musique/trainer_core.py",
        "tests/test_audit.py",
        "tests/test_data.py",
        "tests/test_evaluation_core.py",
        "tests/test_launch_receipt.py",
        "tests/test_pipeline.py",
        "tests/test_protocol.py",
        "tests/test_training_core.py",
    }
)

EXPECTED_RUNTIME = {
    "seed": 17,
    "arms": ["CONTROL", "HopPAIR"],
    "epochs": 1,
    "optimizer_steps": 240,
    "group_microbatch_orbits": 1,
    "gradient_accumulation_orbits": 8,
    "effective_batch_orbits": 8,
    "states_per_orbit": 3,
    "max_length": 6_144,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "warmup_steps": 8,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "precision": "bfloat16",
    "attention_implementation": "sdpa",
    "gradient_checkpointing": True,
    "optimizer": "adamw_torch_fused",
    "dataloader_num_workers": 0,
    "schedule_algorithm": SCHEDULE_ALGORITHM,
    "single_gpu_per_process": True,
    "no_resume": True,
}


@dataclass(frozen=True)
class LoraSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    coverage: str = "all_transformer_linear"
    target_modules: tuple[str, ...] = ALL_LINEAR_TARGET_MODULES

    def __post_init__(self) -> None:
        if self.r <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.bias != "none" or self.task_type != "CAUSAL_LM":
            raise ValueError("frozen LoRA requires bias=none and task_type=CAUSAL_LM")
        if self.coverage != "all_transformer_linear":
            raise ValueError("frozen LoRA coverage must be all_transformer_linear")
        if self.target_modules != ALL_LINEAR_TARGET_MODULES:
            raise ValueError("frozen LoRA target modules drifted from all transformer linears")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_modules"] = list(self.target_modules)
        return value


def _open_readonly_nofollow(path: Path) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("this frozen pipeline requires O_NOFOLLOW support")
    return os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)


def _absolute_unresolved(path: str | Path) -> Path:
    """Return an absolute spelling without resolving the final symlink."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def exact_absolute_path(path: str | Path, *, description: str = "path") -> Path:
    """Require an absolute path spelling and preserve it for no-follow opens."""

    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{description} must be an absolute path")
    return _absolute_unresolved(expanded)


def assert_no_symlink_components(
    path: str | Path,
    *,
    description: str = "path",
    allow_missing_tail: bool = False,
) -> Path:
    """Reject symlinks in every existing path component without resolving them."""

    candidate = exact_absolute_path(path, description=description)
    if ".." in candidate.parts or os.path.normpath(str(candidate)) != str(candidate):
        raise ValueError(f"{description} contains a noncanonical path spelling")
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
            raise RuntimeError(f"{description} changed during component inspection")
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{description} refuses symlink component: {current}")
    return candidate


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _assert_stable_regular_file(
    path: Path,
    before: os.stat_result,
    after: os.stat_result,
) -> None:
    if not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(after.st_mode):
        raise ValueError(f"verified input is not a regular file: {path}")
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"verified input changed while open: {path}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _loads_strict_json(payload: bytes, description: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 in {description}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description}") from exc


def verified_bytes(path: str | Path, *, expected_sha256: str | None = None) -> tuple[bytes, str]:
    """Read and hash exact bytes through one no-follow file descriptor."""

    source = _absolute_unresolved(path)
    assert_no_symlink_components(source, description="verified byte input")
    descriptor = _open_readonly_nofollow(source)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"verified input is not a regular file: {source}")
        payload = handle.read()
        after = os.fstat(handle.fileno())
    _assert_stable_regular_file(source, before, after)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"verified input hash mismatch: {source}")
    return payload, digest


def verified_json_object(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    description: str = "JSON artifact",
) -> tuple[dict[str, Any], str]:
    payload, digest = verified_bytes(path, expected_sha256=expected_sha256)
    value = _loads_strict_json(payload, f"{description}: {Path(path)}")
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object: {Path(path)}")
    return value, digest


def verified_jsonl_objects(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Hash and parse JSONL in one streaming no-follow open."""

    source = _absolute_unresolved(path)
    assert_no_symlink_components(source, description="verified JSONL input")
    descriptor = _open_readonly_nofollow(source)
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"verified JSONL is not a regular file: {source}")
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            value = _loads_strict_json(raw, f"JSONL at {source}:{line_number}")
            if not isinstance(value, dict):
                raise TypeError(f"{source}:{line_number}: record must be an object")
            records.append(value)
        after = os.fstat(handle.fileno())
    _assert_stable_regular_file(source, before, after)
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ValueError(f"verified JSONL hash mismatch: {source}")
    return records


def sha256_file(path: str | Path) -> str:
    """Hash a stable regular file through one no-follow descriptor."""

    source = _absolute_unresolved(path)
    assert_no_symlink_components(source, description="hash input")
    digest = hashlib.sha256()
    descriptor = _open_readonly_nofollow(source)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"hash input is not a regular file: {source}")
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    _assert_stable_regular_file(source, before, after)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json_object(path: str | Path, description: str) -> dict[str, Any]:
    value, _ = verified_json_object(path, description=description)
    return value


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _manifest(root: Path, files: list[Path]) -> dict[str, Any]:
    if not files:
        raise ValueError(f"empty artifact identity: {root}")
    entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(files, key=lambda item: str(item.relative_to(root))):
        if path.is_symlink():
            raise ValueError(f"artifact identity refuses symlink: {path}")
        relative = str(path.relative_to(root))
        size = path.stat().st_size
        digest = sha256_file(path)
        entries.append({"path": relative, "size_bytes": size, "sha256": digest})
        aggregate.update(f"{relative}\0{size}\0{digest}\n".encode())
    return {"path": str(root), "files": entries, "sha256": aggregate.hexdigest()}


def base_model_manifest(model_path: str | Path) -> dict[str, Any]:
    requested = Path(model_path).expanduser()
    assert_no_symlink_components(requested, description="base-model identity root")
    root = requested.resolve()
    if root != CANONICAL_MODEL_PATH or not root.is_dir():
        raise ValueError(f"base model must be the canonical local Qwen path: {CANONICAL_MODEL_PATH}")
    names = {"config.json", "configuration.json", "generation_config.json", "model.safetensors.index.json"}
    files = [
        path
        for path in root.iterdir()
        if path.is_file()
        and (path.name in names or path.name.startswith("model-") and path.suffix == ".safetensors")
    ]
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError(f"model weights are missing from {root}")
    return _manifest(root, files)


def tokenizer_manifest(model_path: str | Path) -> dict[str, Any]:
    requested = exact_absolute_path(model_path, description="tokenizer model path")
    assert_no_symlink_components(requested, description="tokenizer identity root")
    root = requested
    if root != CANONICAL_MODEL_PATH or not root.is_dir():
        raise ValueError(f"tokenizer must be under the canonical model path: {CANONICAL_MODEL_PATH}")
    allowed = {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    }
    files = [path for path in root.iterdir() if path.is_file() and path.name in allowed]
    if not any(path.name == "tokenizer_config.json" for path in files):
        raise ValueError(f"tokenizer configuration is missing from {root}")
    return _manifest(root, files)


def artifact_manifest(root: str | Path, *, require_adapter: bool = False) -> dict[str, Any]:
    base = exact_absolute_path(root, description="artifact root")
    assert_no_symlink_components(base, description="artifact root")
    if not base.is_dir():
        raise FileNotFoundError(base)
    files = [path for path in base.rglob("*") if path.is_file()]
    if require_adapter:
        present = {path.name for path in files}
        missing = {"adapter_config.json", "adapter_model.safetensors"} - present
        if missing:
            raise ValueError(f"incomplete adapter; missing {sorted(missing)}")
    return _manifest(base, files)


def package_versions() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": {
            "executable": str(_absolute_unresolved(sys.executable)),
            "version": sys.version.split()[0],
        },
        "packages": {},
    }
    for package in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "numpy",
        "tokenizers",
        "safetensors",
    ):
        try:
            result["packages"][package] = version(package)
        except PackageNotFoundError:
            result["packages"][package] = None
    return result


def _validate_environment_lock(receipt: dict[str, Any]) -> None:
    environment = receipt.get("environment_lock")
    observed = package_versions()
    required_packages = {
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "numpy",
        "tokenizers",
        "safetensors",
    }
    if environment != observed:
        raise ValueError("runtime Python/package versions differ from the launch receipt")
    if (
        not isinstance(environment, dict)
        or environment.get("python", {}).get("executable") != str(CANONICAL_PYTHON)
        or set(environment.get("packages", {})) != required_packages
        or any(
            not isinstance(version_value, str) or not version_value
            for version_value in environment["packages"].values()
        )
        or environment["packages"].get("transformers") != "4.57.3"
        or environment["packages"].get("numpy") != "2.4.6"
    ):
        raise ValueError("runtime environment is not the complete canonical v2 environment")


def _validate_sha(value: object, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _exact_artifact(
    entry: Any,
    name: str,
    expected_path: Path | None = None,
    *,
    verify_content: bool = True,
) -> Path:
    if not isinstance(entry, dict) or not {"path", "sha256"}.issubset(entry):
        raise ValueError(f"receipt exact_bindings.{name} is incomplete")
    path = exact_absolute_path(str(entry["path"]), description=f"receipt {name} path")
    assert_no_symlink_components(path, description=f"receipt {name} path")
    if expected_path is not None and path != exact_absolute_path(expected_path):
        raise ValueError(f"receipt {name} path is not canonical: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = _validate_sha(entry["sha256"], f"exact_bindings.{name}.sha256")
    if verify_content and sha256_file(path) != expected_sha:
        raise ValueError(f"receipt-bound {name} hash mismatch")
    return path


def _sidecar_expected_sha256(path: Path, description: str) -> str:
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"{description} SHA sidecar is absent: {sidecar}")
    payload, _ = verified_bytes(sidecar)
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{description} SHA sidecar is not ASCII") from exc
    suffix = f"  {path.name}\n"
    if not text.endswith(suffix) or len(text) != 64 + len(suffix):
        raise ValueError(f"{description} SHA sidecar has noncanonical content")
    return _validate_sha(text[:64], f"{description} sidecar SHA256")


def _sidecar_bound_json(path: Path, description: str) -> tuple[dict[str, Any], str]:
    expected = _sidecar_expected_sha256(path, description)
    return verified_json_object(path, expected_sha256=expected, description=description)


def _validate_source_lock(receipt: dict[str, Any], bindings: dict[str, Any]) -> None:
    source_lock = receipt.get("source_lock")
    if not isinstance(source_lock, dict) or set(source_lock) != REQUIRED_SOURCE_LOCK:
        actual = set(source_lock) if isinstance(source_lock, dict) else set()
        raise ValueError(
            "launch source_lock is not the exact executable set; "
            f"missing={sorted(REQUIRED_SOURCE_LOCK - actual)}, "
            f"extra={sorted(actual - REQUIRED_SOURCE_LOCK)}"
        )
    for relative, expected_sha in source_lock.items():
        _validate_sha(expected_sha, f"source_lock.{relative}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"source lock escapes the project: {relative}")
        source = PROJECT_ROOT / relative_path
        assert_no_symlink_components(source, description=f"source lock {relative}")
        if not source.is_file() or sha256_file(source) != expected_sha:
            raise ValueError(f"source lock mismatch: {relative}")
    expected_lock_sha = _validate_sha(bindings.get("source_lock_sha256"), "source_lock_sha256")
    if canonical_json_sha256(source_lock) != expected_lock_sha:
        raise ValueError("source_lock canonical hash differs from exact_bindings")


def _validate_independent_redteam_signoff(
    receipt: dict[str, Any],
    bindings: dict[str, Any],
) -> None:
    """Re-read the sole canonical sign-off and bind it transitively to the receipt."""

    entry = receipt.get("independent_redteam_signoff")
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "reviewer_id"}:
        raise ValueError("launch receipt lacks the exact independent red-team sign-off binding")
    path = exact_absolute_path(str(entry["path"]), description="independent red-team sign-off")
    if path != CANONICAL_REDTEAM_SIGNOFF:
        raise ValueError("independent red-team sign-off path is noncanonical or a symlink")
    assert_no_symlink_components(path, description="independent red-team sign-off")
    signoff, actual_sha = _sidecar_bound_json(path, "independent red-team sign-off")
    if actual_sha != _validate_sha(entry["sha256"], "independent red-team sign-off SHA256"):
        raise ValueError("independent red-team sign-off hash differs from the launch receipt")
    required_keys = {
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
    if set(signoff) != required_keys:
        raise ValueError("independent red-team sign-off fields drift")
    reviewer_id = signoff.get("reviewer_id")
    if (
        signoff.get("schema_version") != REDTEAM_SIGNOFF_SCHEMA
        or signoff.get("status") != REDTEAM_SIGNOFF_STATUS
        or signoff.get("reviewer_kind") != "independent_redteam"
        or not isinstance(reviewer_id, str)
        or not reviewer_id.strip()
        or reviewer_id != entry["reviewer_id"]
        or not isinstance(signoff.get("created_at"), str)
        or not signoff["created_at"].strip()
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
        raise ValueError("independent red-team sign-off approval contract failed")
    expected = {
        "reviewed_protocol_sha256": bindings["protocol"]["sha256"],
        "reviewed_manifest_sha256": bindings["prepared_manifest"]["sha256"],
        "reviewed_train_sha256": bindings["train"]["sha256"],
        "reviewed_dev_sha256": bindings["dev"]["sha256"],
        "reviewed_schedule_sha256": bindings["schedule_sha256"],
        "reviewed_source_lock_sha256": bindings["source_lock_sha256"],
    }
    for key, expected_sha in expected.items():
        _validate_sha(expected_sha, key)
        if signoff.get(key) != expected_sha:
            raise ValueError(f"independent red-team sign-off binding mismatch: {key}")


def _validate_runtime(receipt: dict[str, Any]) -> None:
    runtime = receipt.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("launch receipt lacks runtime")
    common = {key: runtime.get(key) for key in EXPECTED_RUNTIME}
    if common != EXPECTED_RUNTIME:
        raise ValueError(f"runtime drift from frozen contract: {common!r}")
    if set(runtime) != {*EXPECTED_RUNTIME, "generation", "gpu_by_arm"}:
        raise ValueError("runtime contains missing or unexpected fields")
    generation = runtime.get("generation")
    if not isinstance(generation, dict) or set(generation) != {
        "batch_size",
        "max_new_tokens",
        "do_sample",
        "num_beams",
        "num_return_sequences",
        "length_preflight_status",
    }:
        raise ValueError("runtime.generation contract is incomplete")
    if (
        isinstance(generation["batch_size"], bool)
        or not isinstance(generation["batch_size"], int)
        or generation["batch_size"] <= 0
        or generation["max_new_tokens"] != 128
        or generation["do_sample"] is not False
        or generation["num_beams"] != 1
        or generation["num_return_sequences"] != 1
        or generation["length_preflight_status"] != "PASS"
    ):
        raise ValueError("runtime.generation v1 must bind the passed greedy 128-token budget")
    gpu_by_arm = runtime.get("gpu_by_arm")
    if not isinstance(gpu_by_arm, dict) or set(gpu_by_arm) != {"CONTROL", "HopPAIR", "BASE"}:
        raise ValueError("runtime.gpu_by_arm must bind CONTROL/HopPAIR/BASE")
    if (
        any(
            not isinstance(value, str) or not value.startswith(("GPU-", "MIG-"))
            for value in gpu_by_arm.values()
        )
        or len(set(gpu_by_arm.values())) != 3
    ):
        raise ValueError("runtime.gpu_by_arm must bind three distinct stable GPU UUIDs")


def _validate_objectives_and_lora(receipt: dict[str, Any]) -> None:
    if receipt.get("lora") != LoraSpec().to_dict():
        raise ValueError("launch receipt LoRA configuration drift")
    expected = {
        "CONTROL": {"kl_weight": 0.0, "flip_weight": 0.0},
        "HopPAIR": {"kl_weight": 0.1, "flip_weight": 0.2, "flip_margin": 2.0},
    }
    if receipt.get("objectives") != expected:
        raise ValueError("launch receipt objective weights drift")


def _validate_hardware_lock(receipt: dict[str, Any]) -> None:
    lock = receipt.get("hardware_lock")
    if not isinstance(lock, dict) or set(lock) != {"CONTROL", "HopPAIR", "BASE"}:
        raise ValueError("launch receipt hardware_lock must bind CONTROL/HopPAIR/BASE")
    expected_fields = {
        "cuda_visible_devices",
        "uuid",
        "name",
        "driver_version",
        "driver_cuda_version",
        "torch_cuda_version",
        "compute_capability",
    }
    for arm, entry in lock.items():
        if not isinstance(entry, dict) or set(entry) != expected_fields:
            raise ValueError(f"hardware_lock.{arm} fields drift")
        visible = entry["cuda_visible_devices"]
        uuid = entry["uuid"]
        capability = entry["compute_capability"]
        if (
            not isinstance(visible, str)
            or not visible
            or "," in visible
            or visible != receipt.get("runtime", {}).get("gpu_by_arm", {}).get(arm)
            or visible != uuid
            or not isinstance(uuid, str)
            or not uuid.startswith(("GPU-", "MIG-"))
            or not isinstance(entry["name"], str)
            or not entry["name"]
            or not isinstance(entry["driver_version"], str)
            or not entry["driver_version"]
            or not isinstance(entry["driver_cuda_version"], str)
            or not entry["driver_cuda_version"]
            or not isinstance(entry["torch_cuda_version"], str)
            or not entry["torch_cuda_version"]
            or not isinstance(capability, list)
            or len(capability) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
        ):
            raise ValueError(f"hardware_lock.{arm} contains an invalid identity")
    if (
        len({entry["name"] for entry in lock.values()}) != 1
        or len({entry["driver_version"] for entry in lock.values()}) != 1
        or len({entry["driver_cuda_version"] for entry in lock.values()}) != 1
        or len({entry["torch_cuda_version"] for entry in lock.values()}) != 1
        or len({tuple(entry["compute_capability"]) for entry in lock.values()}) != 1
    ):
        raise ValueError("hardware_lock must bind three same-model compatible GPUs")

    preflight = receipt.get("hardware_preflight")
    expected_preflight_keys = {
        "observations",
        "same_model",
        "idle_memory_ceiling_mib",
        "idle_utilization_ceiling_percent",
        "excluded_gpu",
        "model_loaded",
        "cuda_initialized_by_freezer",
    }
    if not isinstance(preflight, dict) or set(preflight) != expected_preflight_keys:
        raise ValueError("hardware_preflight fields drift")
    excluded = preflight["excluded_gpu"]
    selected_uuids = {entry["uuid"] for entry in lock.values()}
    if (
        preflight["same_model"] is not True
        or preflight["idle_memory_ceiling_mib"] != 1_024
        or preflight["idle_utilization_ceiling_percent"] != 0
        or preflight["model_loaded"] is not False
        or preflight["cuda_initialized_by_freezer"] is not False
        or not isinstance(excluded, dict)
        or excluded
        != {
            "index": "3",
            "uuid": excluded.get("uuid"),
            "present": True,
            "assigned": False,
        }
        or not isinstance(excluded.get("uuid"), str)
        or not excluded["uuid"].startswith(("GPU-", "MIG-"))
        or excluded["uuid"] in selected_uuids
    ):
        raise ValueError("hardware_preflight GPU3 exclusion or static boundary drift")
    observations = preflight["observations"]
    expected_indices = {"CONTROL": "0", "HopPAIR": "1", "BASE": "2"}
    if not isinstance(observations, dict) or set(observations) != set(expected_indices):
        raise ValueError("hardware_preflight observations must bind all three arms")
    for arm, index in expected_indices.items():
        observation = observations[arm]
        if (
            not isinstance(observation, dict)
            or observation
            != {
                "index": index,
                "memory_used_mib": observation.get("memory_used_mib"),
                "utilization_percent": 0,
                "compute_processes": 0,
                "idle": True,
            }
            or isinstance(observation.get("memory_used_mib"), bool)
            or not isinstance(observation.get("memory_used_mib"), int)
            or not 0 <= observation["memory_used_mib"] <= 1_024
        ):
            raise ValueError(f"hardware_preflight.{arm} is not an idle physical-index binding")


def validate_output_roots(receipt: dict[str, Any]) -> dict[str, Path]:
    """Validate that every production root is absolute, unique, and disjoint."""

    output_roots = receipt.get("output_roots")
    if not isinstance(output_roots, dict) or output_roots != CANONICAL_OUTPUT_ROOTS:
        raise ValueError("receipt output_roots differ from the sole canonical seed-17 mapping")
    resolved_outputs: dict[str, Path] = {}
    for key, raw_path in output_roots.items():
        output = exact_absolute_path(str(raw_path), description=f"output root {key}")
        assert_no_symlink_components(
            output,
            description=f"output root {key}",
            allow_missing_tail=True,
        )
        try:
            output.relative_to((PROJECT_ROOT / "runs").resolve())
        except ValueError as exc:
            raise ValueError(f"output root must stay under project runs/: {key}") from exc
        resolved_outputs[key] = output
    if len(set(resolved_outputs.values())) != len(resolved_outputs):
        raise ValueError("receipt output roots must be unique")
    ordered = sorted(resolved_outputs.items())
    for index, (first_key, first) in enumerate(ordered):
        for second_key, second in ordered[index + 1 :]:
            if first in second.parents or second in first.parents:
                raise ValueError(
                    f"receipt output roots may not overlap: {first_key}/{second_key}"
                )
    return resolved_outputs


def validate_launch_receipt(
    path: str | Path,
    *,
    purpose: str = "metadata",
) -> dict[str, Any]:
    """Validate the sole launch authority under a purpose-specific read boundary.

    This function never opens train/dev bodies.  Their paths and transitive
    hashes are frozen here; training and generation later hash *and parse* the
    authorized body through one descriptor at the exact point of use.
    """

    if purpose not in {"metadata", "train", "generation"}:
        raise ValueError("receipt validation purpose must be metadata, train, or generation")

    receipt_path = exact_absolute_path(path, description="launch receipt path")
    if receipt_path != CANONICAL_RECEIPT:
        raise ValueError(f"launch receipt must use the canonical path: {CANONICAL_RECEIPT}")
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"GPU launch is forbidden until the future receipt exists: {CANONICAL_RECEIPT}"
        )
    assert_no_symlink_components(receipt_path, description="launch receipt path")
    receipt, receipt_sha = _sidecar_bound_json(receipt_path, "launch receipt")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != RECEIPT_STATUS
        or receipt.get("canonical_path") != str(CANONICAL_RECEIPT)
    ):
        raise ValueError("launch receipt schema/status/canonical path mismatch")
    expected_evidence = canonical_json_sha256(
        {key: value for key, value in receipt.items() if key not in {"created_at", "evidence_sha256"}}
    )
    if receipt.get("evidence_sha256") != expected_evidence:
        raise ValueError("launch receipt self-evidence hash mismatch")
    bindings = receipt.get("exact_bindings")
    expected_binding_keys = {
        "protocol",
        "prepared_manifest",
        "train",
        "dev",
        "schedule_sha256",
        "base_model",
        "tokenizer",
        "source_lock_sha256",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_binding_keys:
        raise ValueError("launch receipt exact_bindings are incomplete or contain extras")

    protocol_entry = bindings["protocol"]
    if not isinstance(protocol_entry, dict) or set(protocol_entry) != {"path", "sha256"}:
        raise ValueError("exact_bindings.protocol must contain path/sha256 only")
    protocol_path = exact_absolute_path(
        str(protocol_entry["path"]), description="frozen protocol path"
    )
    if protocol_path != CANONICAL_PROTOCOL:
        raise ValueError(f"receipt protocol path is not canonical: {protocol_path}")
    assert_no_symlink_components(protocol_path, description="frozen protocol path")
    protocol, protocol_sha = _sidecar_bound_json(protocol_path, "frozen protocol")
    if protocol_sha != _validate_sha(protocol_entry["sha256"], "protocol.sha256"):
        raise ValueError("receipt protocol hash differs from its canonical sidecar")

    prepared_entry = bindings["prepared_manifest"]
    if not isinstance(prepared_entry, dict) or set(prepared_entry) != {"path", "sha256"}:
        raise ValueError("exact_bindings.prepared_manifest must contain path/sha256 only")
    prepared_path = exact_absolute_path(
        str(prepared_entry["path"]), description="prepared manifest path"
    )
    if prepared_path != CANONICAL_PREPARED_MANIFEST:
        raise ValueError(f"prepared manifest path is not canonical: {prepared_path}")
    assert_no_symlink_components(prepared_path, description="prepared manifest path")
    prepared_sha = _validate_sha(prepared_entry["sha256"], "prepared_manifest.sha256")
    manifest, actual_prepared_sha = verified_json_object(
        prepared_path,
        expected_sha256=prepared_sha,
        description="prepared-data manifest",
    )
    if actual_prepared_sha != prepared_sha:
        raise AssertionError("verified prepared manifest digest drift")
    train_entry = bindings["train"]
    dev_entry = bindings["dev"]
    train_path = _exact_artifact(
        train_entry,
        "train",
        PROJECT_ROOT / "prepared_data_v2" / "train.jsonl",
        verify_content=False,
    )
    dev_path = _exact_artifact(
        dev_entry,
        "dev",
        PROJECT_ROOT / "prepared_data_v2" / "dev.jsonl",
        verify_content=False,
    )
    if train_entry.get("orbits") != 1_920 or train_entry.get("records") != 5_760:
        raise ValueError("receipt train counts must be 1920 orbits/5760 records")
    if dev_entry.get("orbits") != 400 or dev_entry.get("records") != 1_200:
        raise ValueError("receipt dev counts must be 400 orbits/1200 records")
    _validate_sha(bindings["schedule_sha256"], "exact_bindings.schedule_sha256")

    counts = manifest.get("counts", {})
    artifact_hashes = manifest.get("artifact_sha256", {})
    if (
        manifest.get("schema_version") != "support-orbit-musique/v2"
        or manifest.get("release_gate", {}).get("status") != "READY"
        or manifest.get("tokenizer_audit", {}).get("zero_truncation_pass") is not True
        or counts.get("train_orbits") != 1_920
        or counts.get("dev_orbits") != 400
        or counts.get("train_records") != 5_760
        or counts.get("dev_records") != 1_200
        or artifact_hashes.get("train.jsonl") != train_entry["sha256"]
        or artifact_hashes.get("dev.jsonl") != dev_entry["sha256"]
    ):
        raise ValueError("prepared-data manifest did not pass the exact release contract")

    from .protocol import PROTOCOL_TO_DATA_SCHEMA, validate_data_release

    protocol_lock = protocol.get("prepared_data_lock")
    if (
        protocol.get("schema_version") != "support-orbit-musique.protocol/v2"
        or PROTOCOL_TO_DATA_SCHEMA.get(protocol.get("schema_version"))
        != "support-orbit-musique/v2"
        or protocol.get("protocol_status") != "FROZEN_PRE_GPU"
        or protocol.get("launch_status") != "STOP_BEFORE_GPU"
        or not isinstance(protocol_lock, dict)
        or protocol_lock.get("path") != "prepared_data_v2/manifest.json"
        or protocol_lock.get("expected_schema_version") != "support-orbit-musique/v2"
        or protocol_lock.get("expected_manifest_sha256") != prepared_sha
        or protocol_lock.get("release_status") != "READY"
        or protocol_lock.get("shadow_body_opened_during_protocol_validation") is not False
    ):
        raise ValueError("frozen protocol schema/status/prepared-data lock mismatch")
    audit_path = prepared_path.parent / "audit.json"
    marker_path = prepared_path.parent / "SHADOW_SEALED.json"
    audit_sha = _validate_sha(artifact_hashes.get("audit.json"), "manifest audit hash")
    marker_sha = _validate_sha(
        artifact_hashes.get("SHADOW_SEALED.json"), "manifest shadow marker hash"
    )
    audit, _ = verified_json_object(
        audit_path,
        expected_sha256=audit_sha,
        description="prepared-data audit",
    )
    marker, _ = verified_json_object(
        marker_path,
        expected_sha256=marker_sha,
        description="sealed-shadow marker",
    )
    release_report = validate_data_release(
        protocol=protocol,
        manifest_path=prepared_path,
        audit_path=audit_path,
        shadow_marker_path=marker_path,
        verified={
            "manifest": manifest,
            "audit": audit,
            "shadow_marker": marker,
            "sha256": {
                "manifest.json": prepared_sha,
                "audit.json": audit_sha,
                "SHADOW_SEALED.json": marker_sha,
                # Bodies remain unopened here.  The builder manifest and frozen
                # protocol independently bind these writer-time digests; each
                # body is verified again through its use-time descriptor.
                "train.jsonl": train_entry["sha256"],
                "dev.jsonl": dev_entry["sha256"],
            },
        },
    )
    if release_report.get("passed") is not True or release_report.get("shadow_body_opened") is not False:
        raise ValueError("protocol data-release validation did not pass without shadow access")

    model_entry = bindings["base_model"]
    tokenizer_entry = bindings["tokenizer"]
    if not isinstance(model_entry, dict) or set(model_entry) != {"path", "sha256"}:
        raise ValueError("exact_bindings.base_model must contain path/sha256 only")
    if exact_absolute_path(
        str(model_entry["path"]), description="base-model binding path"
    ) != CANONICAL_MODEL_PATH:
        raise ValueError("receipt base-model path drift")
    assert_no_symlink_components(CANONICAL_MODEL_PATH, description="base-model binding path")
    _validate_sha(model_entry["sha256"], "base_model.sha256")
    tokenizer_keys = {"path", "sha256", "chat_template_sha256", "eos_token_id", "pad_token_id"}
    if not isinstance(tokenizer_entry, dict) or set(tokenizer_entry) != tokenizer_keys:
        raise ValueError("exact_bindings.tokenizer fields drift")
    if exact_absolute_path(
        str(tokenizer_entry["path"]), description="tokenizer binding path"
    ) != CANONICAL_MODEL_PATH:
        raise ValueError("receipt tokenizer path drift")
    assert_no_symlink_components(CANONICAL_MODEL_PATH, description="tokenizer binding path")
    _validate_sha(tokenizer_entry["sha256"], "tokenizer.sha256")
    _validate_sha(tokenizer_entry["chat_template_sha256"], "tokenizer.chat_template_sha256")
    if not all(isinstance(tokenizer_entry[key], int) for key in ("eos_token_id", "pad_token_id")):
        raise ValueError("receipt tokenizer EOS/PAD IDs must be integers")

    _validate_source_lock(receipt, bindings)
    _validate_independent_redteam_signoff(receipt, bindings)
    _validate_runtime(receipt)
    _validate_objectives_and_lora(receipt)
    _validate_hardware_lock(receipt)
    if receipt.get("forbidden_access") != {
        "withdrawn_v1": True,
        "shadow": True,
        "official_dev": True,
        "official_test": True,
    }:
        raise ValueError("receipt forbidden-access policy drift")
    validate_output_roots(receipt)
    _validate_environment_lock(receipt)
    gates = receipt.get("pre_gpu_gates")
    if (
        not isinstance(gates, list)
        or not gates
        or any(not isinstance(row, dict) or row.get("status") != "PASS" for row in gates)
    ):
        raise ValueError("launch receipt pre-GPU gates are not all PASS")
    tests = receipt.get("test_evidence")
    if (
        not isinstance(tests, dict)
        or tests.get("passed") is not True
        or tests.get("returncode") != 0
        or tests.get("cuda_visible_devices") != ""
        or tests.get("offline") is not True
    ):
        raise ValueError("launch receipt CPU/offline test evidence is incomplete")
    receipt["_validated"] = {
        "path": str(receipt_path),
        "sha256": receipt_sha,
        "protocol_path": str(protocol_path),
        "prepared_manifest_path": str(prepared_path),
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "data_release": release_report,
    }
    return receipt


def validate_static_model_identity(receipt: dict[str, Any], model_path: str | Path) -> dict[str, Any]:
    """Hash static model/tokenizer files before any Transformers model load."""

    root = Path(model_path).expanduser().resolve()
    bindings = receipt["exact_bindings"]
    model = base_model_manifest(root)
    tokenizer = tokenizer_manifest(root)
    if model["sha256"] != bindings["base_model"]["sha256"]:
        raise ValueError("runtime base-model artifact differs from the launch receipt")
    if tokenizer["sha256"] != bindings["tokenizer"]["sha256"]:
        raise ValueError("runtime tokenizer artifact differs from the launch receipt")
    return {"base_model": model, "tokenizer_files": tokenizer}


def load_tokenizer(model_path: str | Path, receipt: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    root = Path(model_path).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "chat_template", None) or tokenizer.eos_token_id is None:
        raise ValueError("base tokenizer lacks its native chat template or EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    expected = receipt["exact_bindings"]["tokenizer"]
    actual_chat = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    if (
        actual_chat != expected["chat_template_sha256"]
        or tokenizer.eos_token_id != expected["eos_token_id"]
        or tokenizer.pad_token_id != expected["pad_token_id"]
    ):
        raise ValueError("runtime tokenizer semantic identity differs from the receipt")
    tokenizer.padding_side = "right"
    return tokenizer


def _nvidia_driver_cuda_version(expected_uuid: str) -> str:
    overview = subprocess.run(
        ["nvidia-smi", "-i", expected_uuid],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = re.search(
        r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)",
        overview.stdout,
    )
    if match is None:
        raise RuntimeError("nvidia-smi did not report the driver CUDA compatibility version")
    return match.group(1)


def assert_gpu_uuid_idle(expected_uuid: str) -> dict[str, str]:
    """Recheck a frozen physical GPU identity and idleness before CUDA use."""

    if not isinstance(expected_uuid, str) or not expected_uuid.startswith(("GPU-", "MIG-")):
        raise ValueError("runtime GPU selector must be a stable UUID")
    identity = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            expected_uuid,
            "--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    lines = [line.strip() for line in identity.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("nvidia-smi did not return exactly one UUID-selected GPU")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 5 or parts[0] != expected_uuid:
        raise RuntimeError("nvidia-smi UUID-selected identity row is malformed or drifted")
    try:
        memory_used = int(parts[3])
        utilization = int(parts[4])
    except ValueError as exc:
        raise RuntimeError("nvidia-smi UUID-selected idle metrics are not integers") from exc
    processes = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            expected_uuid,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    active_processes = [line for line in processes.stdout.splitlines() if line.strip()]
    if memory_used > 1_024 or utilization != 0 or active_processes:
        raise RuntimeError(
            "UUID-selected GPU is no longer idle immediately before CUDA/model use"
        )
    return {
        "uuid": parts[0],
        "name": parts[1],
        "driver_version": parts[2],
        "driver_cuda_version": _nvidia_driver_cuda_version(expected_uuid),
    }


def validate_runtime_cuda_versions(
    expected: dict[str, Any],
    *,
    observed_driver_cuda_version: str,
    observed_torch_cuda_version: str,
) -> None:
    """Compare driver and Torch CUDA identities independently, never to each other."""

    if (
        expected.get("driver_cuda_version") != observed_driver_cuda_version
        or expected.get("torch_cuda_version") != observed_torch_cuda_version
    ):
        raise RuntimeError("runtime driver/Torch CUDA identity differs from the launch receipt")


def ensure_bf16_single_cuda(receipt: dict[str, Any], arm: str) -> dict[str, Any]:

    expected_gpu = receipt["runtime"]["gpu_by_arm"][arm]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu:
        raise ValueError(f"{arm} requires CUDA_VISIBLE_DEVICES={expected_gpu}")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1" or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("frozen launch requires TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("frozen launch requires WORLD_SIZE=1")
    pre_cuda_identity = assert_gpu_uuid_idle(expected_gpu)

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("frozen launch requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the visible CUDA device does not support bfloat16")
    expected = receipt["hardware_lock"][arm]
    properties = torch.cuda.get_device_properties(0)
    observed_torch_cuda = str(torch.version.cuda)
    validate_runtime_cuda_versions(
        expected,
        observed_driver_cuda_version=pre_cuda_identity["driver_cuda_version"],
        observed_torch_cuda_version=observed_torch_cuda,
    )
    observed = {
        "cuda_visible_devices": expected_gpu,
        "uuid": pre_cuda_identity["uuid"],
        "name": properties.name,
        "driver_version": pre_cuda_identity["driver_version"],
        "driver_cuda_version": pre_cuda_identity["driver_cuda_version"],
        "torch_cuda_version": observed_torch_cuda,
        "compute_capability": [int(properties.major), int(properties.minor)],
    }
    if pre_cuda_identity["name"] != properties.name or observed != expected:
        raise RuntimeError(f"runtime GPU identity differs from receipt for {arm}: {observed}")
    return observed


def set_seed(seed: int) -> None:
    import torch
    from transformers import set_seed as transformers_set_seed

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)


def build_lora_model(model_path: str | Path, spec: LoraSpec) -> Any:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        Path(model_path).expanduser().resolve(),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=list(spec.target_modules),
        bias=spec.bias,
    )
    result = get_peft_model(model, lora)
    unexpected = [
        name
        for name, parameter in result.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected:
        raise ValueError(f"non-LoRA parameters unexpectedly trainable: {unexpected[:8]}")
    return result


def trainable_parameter_summary(model: Any) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if total <= 0 or trainable <= 0:
        raise ValueError("model has no trainable adapter parameters")
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_fraction": float(trainable / total),
    }


def tensor_collection_sha256(named_tensors: list[tuple[str, Any]]) -> str:
    """Hash exact tensor names, shapes, dtypes, and raw bytes deterministically."""

    import torch

    digest = hashlib.sha256()
    if not named_tensors:
        raise ValueError("cannot hash an empty tensor collection")
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} is not a tensor")
        value = tensor.detach().contiguous().cpu()
        # Optimizer ``step`` is commonly a zero-dimensional tensor; flattening
        # first makes its raw-byte view legal without changing the recorded
        # original shape.
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        header = json.dumps(
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype), "bytes": len(raw)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(header)
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\n")
    return digest.hexdigest()


def trainable_parameters_sha256(model: Any) -> str:
    values = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return tensor_collection_sha256(values)


def optimizer_state_sha256(optimizer: Any, model: Any) -> str:
    """Hash optimizer group scalars and tensor state using stable parameter names."""

    import torch

    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    digest = hashlib.sha256()
    for group_index, group in enumerate(optimizer.param_groups):
        names = [name_by_id[id(parameter)] for parameter in group["params"]]
        scalars = {
            key: value
            for key, value in group.items()
            if key != "params" and isinstance(value, (str, int, float, bool, type(None), tuple, list))
        }
        digest.update(
            json.dumps(
                {"group": group_index, "parameters": names, "scalars": scalars},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        )
        digest.update(b"\n")
    tensors: list[tuple[str, torch.Tensor]] = []
    scalar_state: list[dict[str, Any]] = []
    for parameter, state in optimizer.state.items():
        name = name_by_id[id(parameter)]
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                tensors.append((f"{name}.{key}", value))
            else:
                scalar_state.append({"parameter": name, "key": key, "value": value})
    digest.update(
        json.dumps(scalar_state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    digest.update(b"\n")
    digest.update(tensor_collection_sha256(tensors).encode())
    return digest.hexdigest()


__all__ = [
    "ALL_LINEAR_TARGET_MODULES",
    "CANONICAL_MODEL_PATH",
    "CANONICAL_OUTPUT_ROOTS",
    "CANONICAL_PYTHON",
    "CANONICAL_PREPARED_MANIFEST",
    "CANONICAL_PROTOCOL",
    "CANONICAL_REDTEAM_SIGNOFF",
    "CANONICAL_RECEIPT",
    "EXPECTED_RUNTIME",
    "EXPECTED_OUTPUT_ROOT_KEYS",
    "LoraSpec",
    "PROJECT_ROOT",
    "RECEIPT_SCHEMA",
    "RECEIPT_STATUS",
    "REDTEAM_SIGNOFF_SCHEMA",
    "REDTEAM_SIGNOFF_STATUS",
    "REQUIRED_SOURCE_LOCK",
    "SCHEDULE_ALGORITHM",
    "artifact_manifest",
    "assert_no_symlink_components",
    "assert_gpu_uuid_idle",
    "atomic_json",
    "base_model_manifest",
    "build_lora_model",
    "canonical_json_sha256",
    "ensure_bf16_single_cuda",
    "exact_absolute_path",
    "load_tokenizer",
    "optimizer_state_sha256",
    "package_versions",
    "read_json_object",
    "set_seed",
    "sha256_file",
    "tensor_collection_sha256",
    "tokenizer_manifest",
    "trainable_parameter_summary",
    "trainable_parameters_sha256",
    "validate_launch_receipt",
    "validate_output_roots",
    "validate_runtime_cuda_versions",
    "validate_static_model_identity",
    "verified_bytes",
    "verified_json_object",
    "verified_jsonl_objects",
]
