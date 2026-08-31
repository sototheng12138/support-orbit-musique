"""Fail-closed production I/O for receipt-bound development evaluation.

No caller chooses a gold file, prediction file, binding, report destination,
or comparison arm.  Those paths are derived from the already validated launch
receipt and completed generation manifest.  File contents are hashed and
parsed from the same ``O_NOFOLLOW`` descriptor after stable ``fstat`` checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import assert_no_symlink_components, atomic_json
from .metrics import SCHEMA_VERSION, compare_evaluations, evaluate_records, verify_run_binding


GENERATION_SCHEMA = "support-orbit-musique.generation.v1"
GENERATION_ARMS = ("BASE", "CONTROL", "HopPAIR")
TRAINED_ARMS = ("CONTROL", "HopPAIR")
DEV_ORBITS = 400
DEV_RECORDS = 1_200
MAX_NEW_TOKENS = 128
GENERATION_ROOT_KEYS = {
    "BASE": "base_dev_generation",
    "CONTROL": "control_dev_generation",
    "HopPAIR": "hoppair_dev_generation",
}
EVALUATION_ROOT_KEYS = {
    "BASE": "base_dev_evaluation",
    "CONTROL": "control_dev_evaluation",
    "HopPAIR": "hoppair_dev_evaluation",
}
EXPECTED_OUTPUT_ROOT_KEYS = {
    "control_train",
    "hoppair_train",
    *GENERATION_ROOT_KEYS.values(),
    *EVALUATION_ROOT_KEYS.values(),
    "dev_comparison",
}
FAIRNESS_CHECKS = {
    "initial_trainable_parameters_exact",
    "schedule_exact",
    "train_artifact_exact",
    "static_model_exact",
    "tokenizer_exact",
    "runtime_exact",
    "lora_exact",
}
PROTECTED_SUBSTRINGS = (
    "prepared_data_v1",
    "shadow",
    "protected",
    "official_dev",
    "official-dev",
    "official_test",
    "official-test",
    "musique_full_v1.0_dev",
    "musique_full_v1.0_test",
)
PROTECTED_COMPONENTS = {"test", "tests", "shadow", "official"}


@dataclass(frozen=True, slots=True)
class BoundPayload:
    path: Path
    sha256: str
    size_bytes: int
    value: Any


@dataclass(frozen=True, slots=True)
class GenerationContext:
    arm: str
    paths: Mapping[str, Path]
    manifest: BoundPayload
    binding: BoundPayload
    expected_binding: Mapping[str, Any]


def _sha256(value: object, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def reject_protected_path(path: str | Path) -> Path:
    """Reject protected names lexically, before any file body can be opened."""

    absolute = _absolute(path)
    lowered = str(absolute).casefold().replace("\\", "/")
    components = {part.casefold() for part in absolute.parts}
    if any(token in lowered for token in PROTECTED_SUBSTRINGS) or components & PROTECTED_COMPONENTS:
        raise ValueError(f"evaluation refuses a protected path before open: {absolute}")
    return absolute


def _canonical_path(supplied: str | Path, expected: str | Path, name: str) -> Path:
    supplied_path = reject_protected_path(supplied)
    expected_path = reject_protected_path(expected)
    if supplied_path != expected_path:
        raise ValueError(f"{name} path is noncanonical: {supplied_path}; expected {expected_path}")
    return expected_path


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _loads_json(raw: bytes, description: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{description} is not strict UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description} JSON") from exc


def _read_stable_bytes(
    path: str | Path,
    *,
    description: str,
    expected_sha256: str | None,
    maximum_bytes: int,
) -> tuple[Path, bytes, str]:
    canonical = reject_protected_path(path)
    assert_no_symlink_components(canonical, description=description)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("production evaluation requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(canonical, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{description} is not a regular file: {canonical}")
        if before.st_nlink != 1:
            raise ValueError(f"{description} must have exactly one hard link: {canonical}")
        if before.st_size > maximum_bytes:
            raise ValueError(f"{description} exceeds the frozen size ceiling")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"{description} exceeded the frozen size ceiling while reading")
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or total != before.st_size:
        raise RuntimeError(f"{description} changed while it was being verified")
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None and actual_sha256 != _sha256(expected_sha256, description):
        raise ValueError(f"{description} hash mismatch")
    return canonical, b"".join(chunks), actual_sha256


def read_bound_json_object(
    path: str | Path,
    *,
    description: str,
    expected_sha256: str | None = None,
) -> BoundPayload:
    canonical, raw, digest = _read_stable_bytes(
        path,
        description=description,
        expected_sha256=expected_sha256,
        maximum_bytes=16 * 1024 * 1024,
    )
    value = _loads_json(raw, description)
    if not isinstance(value, dict):
        raise TypeError(f"{description} must contain one JSON object")
    return BoundPayload(canonical, digest, len(raw), value)


def read_bound_jsonl(
    path: str | Path,
    *,
    description: str,
    expected_sha256: str,
) -> BoundPayload:
    canonical, raw, digest = _read_stable_bytes(
        path,
        description=description,
        expected_sha256=expected_sha256,
        maximum_bytes=512 * 1024 * 1024,
    )
    if not raw.endswith(b"\n"):
        raise ValueError(f"{description} must end with exactly delimited JSONL rows")
    lines = raw.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"{description} contains an empty JSONL row")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        value = _loads_json(line, f"{description} row {line_number}")
        if not isinstance(value, dict):
            raise TypeError(f"{description} row {line_number} must be an object")
        records.append(value)
    return BoundPayload(canonical, digest, len(raw), records)


def _require_keys(value: object, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{name} fields drifted; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _require_positive_integer(value: object, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _receipt_contract(receipt: Mapping[str, Any]) -> None:
    if (
        receipt.get("schema_version") != "support-orbit-musique.launch-receipt.v2"
        or receipt.get("status") != "READY_FOR_GPU"
    ):
        raise ValueError("evaluation requires a validated READY_FOR_GPU launch receipt")
    validated = receipt.get("_validated")
    if not isinstance(validated, Mapping):
        raise ValueError("launch receipt lacks the backend validation marker")
    for field in ("path", "sha256", "protocol_path", "prepared_manifest_path", "train_path", "dev_path"):
        if field not in validated:
            raise ValueError(f"launch receipt validation marker lacks {field}")
    _sha256(validated["sha256"], "launch receipt")
    roots = receipt.get("output_roots")
    if not isinstance(roots, Mapping) or set(roots) != EXPECTED_OUTPUT_ROOT_KEYS:
        raise ValueError("launch receipt output roots are not the exact production set")
    paths = [_absolute(str(roots[key])) for key in sorted(roots)]
    for path in paths:
        reject_protected_path(path)
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("launch receipt output roots must be unique and non-nested")


def _expected_paths(receipt: Mapping[str, Any], arm: str) -> dict[str, Path]:
    if arm not in GENERATION_ARMS:
        raise ValueError(f"arm must be one of {GENERATION_ARMS}, preserving exact case")
    _receipt_contract(receipt)
    roots = receipt["output_roots"]
    generation_root = reject_protected_path(roots[GENERATION_ROOT_KEYS[arm]])
    evaluation_root = reject_protected_path(roots[EVALUATION_ROOT_KEYS[arm]])
    return {
        "generation_manifest": generation_root / "manifest.json",
        "predictions": generation_root / "predictions.jsonl",
        "binding": generation_root / "binding.json",
        "gold": reject_protected_path(receipt["exact_bindings"]["dev"]["path"]),
        "evaluation_root": evaluation_root,
        "evaluation": evaluation_root / "evaluation.json",
    }


def _artifact_identity(
    value: object,
    *,
    name: str,
    expected_path: str | Path,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    artifact = _require_keys(value, {"path", "files", "sha256"}, name)
    _canonical_path(artifact["path"], expected_path, f"{name}.path")
    digest = _sha256(artifact["sha256"], f"{name}.sha256")
    if expected_sha256 is not None and digest != _sha256(expected_sha256, name):
        raise ValueError(f"{name} aggregate hash differs from its bound identity")
    files = artifact["files"]
    if not isinstance(files, list) or not files:
        raise ValueError(f"{name}.files must be a nonempty list")
    seen: set[str] = set()
    for index, row in enumerate(files):
        entry = _require_keys(row, {"path", "size_bytes", "sha256"}, f"{name}.files[{index}]")
        relative = entry["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError(f"{name}.files[{index}] has an invalid relative path")
        seen.add(relative)
        _require_positive_integer(entry["size_bytes"], f"{name}.files[{index}].size_bytes")
        _sha256(entry["sha256"], f"{name}.files[{index}].sha256")
    if [str(row["path"]) for row in files] != sorted(seen):
        raise ValueError(f"{name}.files must be uniquely sorted")
    return artifact


def _validate_fairness(value: object, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    fairness = _require_keys(
        value,
        {
            "passed",
            "checks",
            "diagnostics",
            "initial_trainable_parameters_sha256",
            "schedule_sha256",
            "control_run_sha256",
            "hoppair_run_sha256",
        },
        "generation.paired_training_fairness",
    )
    checks = fairness["checks"]
    diagnostics = fairness["diagnostics"]
    if (
        fairness["passed"] is not True
        or not isinstance(checks, Mapping)
        or set(checks) != FAIRNESS_CHECKS
        or any(value is not True for value in checks.values())
        or not isinstance(diagnostics, Mapping)
        or set(diagnostics) != {"checkpoints_distinct"}
        or not isinstance(diagnostics["checkpoints_distinct"], bool)
    ):
        raise ValueError("generation paired-training fairness is not exactly PASS")
    for field in (
        "initial_trainable_parameters_sha256",
        "schedule_sha256",
        "control_run_sha256",
        "hoppair_run_sha256",
    ):
        _sha256(fairness[field], f"generation.paired_training_fairness.{field}")
    if fairness["schedule_sha256"] != receipt["exact_bindings"]["schedule_sha256"]:
        raise ValueError("generation fairness schedule differs from the launch receipt")
    return fairness


def _validate_generation_manifest(
    manifest: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    arm: str,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "status",
        "started_at",
        "completed_at",
        "arm",
        "receipt",
        "input",
        "model_identity",
        "hardware_identity",
        "training_run",
        "paired_training_fairness",
        "decoding",
        "predictions",
        "binding",
    }
    _require_keys(manifest, expected_top, "generation manifest")
    if (
        manifest["schema_version"] != GENERATION_SCHEMA
        or manifest["status"] != "completed"
        or manifest["arm"] != arm
        or not isinstance(manifest["started_at"], str)
        or not manifest["started_at"]
        or not isinstance(manifest["completed_at"], str)
        or not manifest["completed_at"]
    ):
        raise ValueError("generation manifest is not the exact completed requested arm")
    if manifest["receipt"] != receipt["_validated"]:
        raise ValueError("generation manifest launch-receipt binding drifted")
    if manifest["hardware_identity"] != receipt["hardware_lock"][arm]:
        raise ValueError("generation hardware identity differs from the requested receipt arm")

    dev_entry = receipt["exact_bindings"]["dev"]
    input_entry = _require_keys(
        manifest["input"],
        {"path", "sha256", "orbits", "records", "prompt_tokens_max", "prompt_tokens_total"},
        "generation.input",
    )
    _canonical_path(input_entry["path"], paths["gold"], "generation.input")
    if (
        input_entry["sha256"] != dev_entry["sha256"]
        or input_entry["orbits"] != DEV_ORBITS
        or input_entry["records"] != DEV_RECORDS
    ):
        raise ValueError("generation input differs from the frozen 400-orbit dev artifact")
    prompt_max = _require_positive_integer(input_entry["prompt_tokens_max"], "prompt_tokens_max")
    prompt_total = _require_positive_integer(
        input_entry["prompt_tokens_total"], "prompt_tokens_total"
    )
    if prompt_max > receipt["runtime"]["max_length"] or prompt_total < prompt_max:
        raise ValueError("generation prompt-token accounting is invalid")

    model_identity = _require_keys(
        manifest["model_identity"], {"base_model", "adapter"}, "generation.model_identity"
    )
    base_entry = receipt["exact_bindings"]["base_model"]
    _artifact_identity(
        model_identity["base_model"],
        name="generation.model_identity.base_model",
        expected_path=base_entry["path"],
        expected_sha256=base_entry["sha256"],
    )
    fairness = _validate_fairness(manifest["paired_training_fairness"], receipt)

    initialization_sha256: str | None = None
    if arm == "BASE":
        if model_identity["adapter"] is not None or manifest["training_run"] is not None:
            raise ValueError("BASE generation must have no adapter or training run")
        checkpoint_sha256 = base_entry["sha256"]
    else:
        root_key = "control_train" if arm == "CONTROL" else "hoppair_train"
        training_run = _require_keys(
            manifest["training_run"], {"path", "sha256", "adapter"}, "generation.training_run"
        )
        expected_run_path = _absolute(receipt["output_roots"][root_key]) / "run_manifest.json"
        _canonical_path(training_run["path"], expected_run_path, "generation.training_run")
        run_sha = _sha256(training_run["sha256"], "generation.training_run.sha256")
        fairness_key = "control_run_sha256" if arm == "CONTROL" else "hoppair_run_sha256"
        if run_sha != fairness[fairness_key]:
            raise ValueError("generation training-run hash differs from paired fairness")
        expected_adapter_path = _absolute(receipt["output_roots"][root_key]) / "final_adapter"
        adapter = _artifact_identity(
            model_identity["adapter"],
            name="generation.model_identity.adapter",
            expected_path=expected_adapter_path,
        )
        if training_run["adapter"] != adapter:
            raise ValueError("generation training-run and model adapter identities differ")
        checkpoint_sha256 = adapter["sha256"]
        initialization_sha256 = fairness["initial_trainable_parameters_sha256"]

    generation_contract = receipt["runtime"]["generation"]
    decoding = _require_keys(
        manifest["decoding"],
        {
            "do_sample",
            "batch_size",
            "max_new_tokens",
            "num_beams",
            "num_return_sequences",
            "length_preflight_status",
            "budget_exhausted_rows",
        },
        "generation.decoding",
    )
    if (
        decoding["do_sample"] is not False
        or decoding["batch_size"] != generation_contract["batch_size"]
        or decoding["max_new_tokens"] != MAX_NEW_TOKENS
        or decoding["max_new_tokens"] != generation_contract["max_new_tokens"]
        or decoding["num_beams"] != 1
        or decoding["num_return_sequences"] != 1
        or decoding["length_preflight_status"] != "PASS"
        or decoding["length_preflight_status"]
        != generation_contract["length_preflight_status"]
    ):
        raise ValueError("generation decoding is not exact greedy max_new_tokens=128")
    exhausted = _require_positive_integer(
        decoding["budget_exhausted_rows"], "budget_exhausted_rows", allow_zero=True
    )
    if exhausted > DEV_RECORDS:
        raise ValueError("budget_exhausted_rows exceeds the development set")

    predictions = _require_keys(
        manifest["predictions"],
        {
            "path",
            "sha256",
            "rows",
            "unique_ids",
            "generated_tokens_total",
            "generated_tokens_max",
        },
        "generation.predictions",
    )
    _canonical_path(predictions["path"], paths["predictions"], "generation.predictions")
    _sha256(predictions["sha256"], "generation.predictions.sha256")
    if predictions["rows"] != DEV_RECORDS or predictions["unique_ids"] != DEV_RECORDS:
        raise ValueError("generation must contain exactly 1200 unique predictions")
    generated_total = _require_positive_integer(
        predictions["generated_tokens_total"], "generated_tokens_total", allow_zero=True
    )
    generated_max = _require_positive_integer(
        predictions["generated_tokens_max"], "generated_tokens_max", allow_zero=True
    )
    if generated_max > MAX_NEW_TOKENS or generated_total < generated_max:
        raise ValueError("generation token accounting is invalid")

    binding_entry = _require_keys(
        manifest["binding"], {"path", "sha256"}, "generation.binding"
    )
    _canonical_path(binding_entry["path"], paths["binding"], "generation.binding")
    _sha256(binding_entry["sha256"], "generation.binding.sha256")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "initialization_sha256": initialization_sha256,
        "fairness": fairness,
        "predictions": predictions,
        "binding": binding_entry,
        "decoding": decoding,
    }


def load_generation_context(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    generation_manifest_path: str | Path,
) -> GenerationContext:
    """Validate generation metadata before opening either dev data body."""

    paths = _expected_paths(receipt, arm)
    manifest_path = _canonical_path(
        generation_manifest_path,
        paths["generation_manifest"],
        "generation manifest",
    )
    manifest_payload = read_bound_json_object(
        manifest_path,
        description=f"{arm} generation manifest",
    )
    validated = _validate_generation_manifest(
        manifest_payload.value,
        receipt=receipt,
        arm=arm,
        paths=paths,
    )

    binding_payload = read_bound_json_object(
        paths["binding"],
        description=f"{arm} generation binding",
        expected_sha256=validated["binding"]["sha256"],
    )
    required_binding_keys = {
        "arm_id",
        "dataset_manifest_sha256",
        "split_artifact_sha256",
        "predictions_sha256",
        "protocol_sha256",
        "checkpoint_sha256",
        "launch_receipt_sha256",
        "schedule_sha256",
    }
    if arm in TRAINED_ARMS:
        required_binding_keys.add("initialization_sha256")
    observed_binding = _require_keys(
        binding_payload.value,
        required_binding_keys,
        f"{arm} generation binding",
    )
    expected_binding: dict[str, Any] = {
        "arm_id": arm,
        "dataset_manifest_sha256": receipt["exact_bindings"]["prepared_manifest"]["sha256"],
        "split_artifact_sha256": receipt["exact_bindings"]["dev"]["sha256"],
        "predictions_sha256": validated["predictions"]["sha256"],
        "protocol_sha256": receipt["exact_bindings"]["protocol"]["sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "launch_receipt_sha256": receipt["_validated"]["sha256"],
        "schedule_sha256": receipt["exact_bindings"]["schedule_sha256"],
    }
    if arm in TRAINED_ARMS:
        expected_binding["initialization_sha256"] = validated["initialization_sha256"]
    if observed_binding != expected_binding:
        raise ValueError(f"{arm} generation binding differs from the exact provenance contract")
    return GenerationContext(
        arm=arm,
        paths=paths,
        manifest=manifest_payload,
        binding=binding_payload,
        expected_binding=expected_binding,
    )


def _validate_dev_bodies(
    gold_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    dataset_manifest_sha256: str,
) -> None:
    if len(gold_records) != DEV_RECORDS or len(prediction_records) != DEV_RECORDS:
        raise ValueError("production evaluation requires exactly 1200 gold and prediction rows")
    gold_ids: list[str] = []
    orbit_ids: set[str] = set()
    state_counts: Counter[str] = Counter()
    for row_number, record in enumerate(gold_records, 1):
        record_id = record.get("id")
        orbit_id = record.get("orbit_id")
        state = record.get("state")
        if (
            not isinstance(record_id, str)
            or not isinstance(orbit_id, str)
            or state not in {"C", "D", "M"}
            or record_id != f"{orbit_id}::{state}"
            or record.get("split") != "dev"
            or record.get("training_read_allowed") is not True
            or record.get("sealed") is not False
        ):
            raise ValueError(f"gold dev row {row_number} violates the canonical dev contract")
        gold_ids.append(record_id)
        orbit_ids.add(orbit_id)
        state_counts[str(state)] += 1
    if (
        len(set(gold_ids)) != DEV_RECORDS
        or len(orbit_ids) != DEV_ORBITS
        or state_counts != {"C": DEV_ORBITS, "D": DEV_ORBITS, "M": DEV_ORBITS}
    ):
        raise ValueError("gold dev body is not exactly 400 complete C/D/M orbits")
    for start in range(0, DEV_RECORDS, 3):
        group = gold_records[start : start + 3]
        if (
            tuple(record["state"] for record in group) != ("C", "D", "M")
            or len({record["orbit_id"] for record in group}) != 1
        ):
            raise ValueError("gold dev rows are not in canonical orbit-major C/D/M order")

    prediction_ids: list[str] = []
    expected_keys = {"id", "prediction", "arm_id", "dataset_manifest_sha256"}
    for row_number, record in enumerate(prediction_records, 1):
        if set(record) != expected_keys:
            raise ValueError(f"prediction row {row_number} fields differ from the exact schema")
        record_id = record["id"]
        if (
            not isinstance(record_id, str)
            or not isinstance(record["prediction"], str)
            or record["arm_id"] != arm
            or record["dataset_manifest_sha256"] != dataset_manifest_sha256
        ):
            raise ValueError(f"prediction row {row_number} arm/provenance binding drifted")
        prediction_ids.append(record_id)
    if prediction_ids != gold_ids or len(set(prediction_ids)) != DEV_RECORDS:
        raise ValueError("predictions are not one-to-one in exact canonical dev order")


def evaluate_production_arm(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    generation_manifest_path: str | Path,
    write: bool,
) -> tuple[dict[str, Any], Path]:
    """Evaluate one receipt-derived arm after all metadata gates pass."""

    context = load_generation_context(
        receipt,
        arm=arm,
        generation_manifest_path=generation_manifest_path,
    )
    gold_payload = read_bound_jsonl(
        context.paths["gold"],
        description="canonical development gold",
        expected_sha256=receipt["exact_bindings"]["dev"]["sha256"],
    )
    prediction_payload = read_bound_jsonl(
        context.paths["predictions"],
        description=f"{arm} canonical development predictions",
        expected_sha256=context.expected_binding["predictions_sha256"],
    )
    _validate_dev_bodies(
        gold_payload.value,
        prediction_payload.value,
        arm=arm,
        dataset_manifest_sha256=context.expected_binding["dataset_manifest_sha256"],
    )
    binding_report = verify_run_binding(
        context.binding.value,
        context.expected_binding,
        actual_split_sha256=gold_payload.sha256,
        actual_predictions_sha256=prediction_payload.sha256,
    )
    if not binding_report["passed"]:
        raise ValueError(f"{arm} production binding unexpectedly failed after validation")
    report = evaluate_records(
        gold_payload.value,
        prediction_payload.value,
        binding=binding_report,
    )
    if (
        report["schema_version"] != SCHEMA_VERSION
        or report["run_integrity"] is not True
        or report["counts"] != {"rows": DEV_RECORDS, "orbits": DEV_ORBITS}
    ):
        raise RuntimeError("production metric report violated its postconditions")
    report["production"] = {
        "arm": arm,
        "launch_receipt": {
            "path": receipt["_validated"]["path"],
            "sha256": receipt["_validated"]["sha256"],
        },
        "source_lock_sha256": receipt["exact_bindings"]["source_lock_sha256"],
        "schedule_sha256": receipt["exact_bindings"]["schedule_sha256"],
        "generation_manifest": {
            "path": str(context.manifest.path),
            "sha256": context.manifest.sha256,
        },
        "generation_binding": {
            "path": str(context.binding.path),
            "sha256": context.binding.sha256,
        },
        "gold": {"path": str(gold_payload.path), "sha256": gold_payload.sha256},
        "predictions": {
            "path": str(prediction_payload.path),
            "sha256": prediction_payload.sha256,
        },
        "evaluation": {"path": str(context.paths["evaluation"])},
    }
    destination = context.paths["evaluation"]
    if write:
        assert_no_symlink_components(
            context.paths["evaluation_root"],
            description=f"{arm} evaluation output root before creation",
            allow_missing_tail=True,
        )
        try:
            context.paths["evaluation_root"].mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"evaluation refuses an existing arm output root: {context.paths['evaluation_root']}"
            ) from exc
        assert_no_symlink_components(
            context.paths["evaluation_root"],
            description=f"{arm} evaluation output root after creation",
        )
        atomic_json(destination, report)
    return report, destination


def _load_saved_exact_report(path: Path, expected: Mapping[str, Any], arm: str) -> BoundPayload:
    payload = read_bound_json_object(path, description=f"{arm} saved evaluation report")
    if payload.value != expected:
        raise ValueError(f"{arm} saved evaluation report differs from a fresh bound recomputation")
    return payload


def compare_production(receipt: Mapping[str, Any], *, write: bool) -> tuple[dict[str, Any], Path]:
    """Recompute and compare only canonical CONTROL versus canonical HopPAIR."""

    _receipt_contract(receipt)
    reports: dict[str, dict[str, Any]] = {}
    report_payloads: dict[str, BoundPayload] = {}
    for arm in TRAINED_ARMS:
        paths = _expected_paths(receipt, arm)
        report, expected_report_path = evaluate_production_arm(
            receipt,
            arm=arm,
            generation_manifest_path=paths["generation_manifest"],
            write=False,
        )
        payload = _load_saved_exact_report(expected_report_path, report, arm)
        reports[arm] = report
        report_payloads[arm] = payload
    comparison = compare_evaluations(reports["CONTROL"], reports["HopPAIR"])
    comparison["production"] = {
        "roles": {"control": "CONTROL", "treatment": "HopPAIR"},
        "launch_receipt": {
            "path": receipt["_validated"]["path"],
            "sha256": receipt["_validated"]["sha256"],
        },
        "control_evaluation": {
            "path": str(report_payloads["CONTROL"].path),
            "sha256": report_payloads["CONTROL"].sha256,
        },
        "hoppair_evaluation": {
            "path": str(report_payloads["HopPAIR"].path),
            "sha256": report_payloads["HopPAIR"].sha256,
        },
        "control_generation_manifest_sha256": reports["CONTROL"]["production"][
            "generation_manifest"
        ]["sha256"],
        "hoppair_generation_manifest_sha256": reports["HopPAIR"]["production"][
            "generation_manifest"
        ]["sha256"],
    }
    comparison_root = reject_protected_path(receipt["output_roots"]["dev_comparison"])
    destination = comparison_root / "dev_comparison.json"
    if write:
        assert_no_symlink_components(
            comparison_root,
            description="comparison output root before creation",
            allow_missing_tail=True,
        )
        try:
            comparison_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"comparison refuses an existing output root: {comparison_root}"
            ) from exc
        assert_no_symlink_components(
            comparison_root,
            description="comparison output root after creation",
        )
        atomic_json(destination, comparison)
    return comparison, destination


__all__ = [
    "DEV_ORBITS",
    "DEV_RECORDS",
    "EVALUATION_ROOT_KEYS",
    "GENERATION_ARMS",
    "GENERATION_ROOT_KEYS",
    "GenerationContext",
    "compare_production",
    "evaluate_production_arm",
    "load_generation_context",
    "read_bound_json_object",
    "read_bound_jsonl",
    "reject_protected_path",
]
