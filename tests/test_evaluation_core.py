from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import compare_dev
import evaluate_dev

from support_orbit_musique.metrics import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    SCHEMA_VERSION,
    compare_evaluations,
    evaluate_records,
    paired_cluster_bootstrap,
    verify_run_binding,
)
from support_orbit_musique.evaluation import (
    compare_production,
    evaluate_production_arm,
    load_generation_context,
    read_bound_json_object,
    reject_protected_path,
)
from support_orbit_musique.official_adapter import (
    answer_scores,
    normalize_answer,
    support_scores,
)
from support_orbit_musique.parsing import parse_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("text", "status", "indices", "answer"),
    [
        ("S | evidence=[P06,P10] | answer=north", "S", (6, 10), "north"),
        (" \tS| evidence = [ P00, P19 ]|answer = New York \t", "S", (0, 19), "New York"),
        (
            "U | evidence=[] | answer=INSUFFICIENT_EVIDENCE",
            "U",
            (),
            "INSUFFICIENT_EVIDENCE",
        ),
    ],
)
def test_strict_parser_accepts_only_the_frozen_grammar(
    text: str, status: str, indices: tuple[int, ...], answer: str
) -> None:
    parsed = parse_prediction(text)
    assert parsed.parse_valid
    assert parsed.status == status
    assert parsed.support_indices == indices
    assert parsed.answer == answer


@pytest.mark.parametrize(
    "text",
    [
        "prefix S | evidence=[P06] | answer=north",
        "S | evidence=[P06] | answer=north trailing | prose",
        "S | evidence=[P06] | answer=north\n",
        "s | evidence=[P06] | answer=north",
        "S | evidence=[P6] | answer=north",
        "S | evidence=[P20] | answer=north",
        "S | evidence=[P06,P06] | answer=north",
        "S | evidence=[P10,P06] | answer=north",
        "S | evidence=[] | answer=north",
        "S | evidence=[] | answer=INSUFFICIENT_EVIDENCE",
        "U | evidence=[P06] | answer=INSUFFICIENT_EVIDENCE",
        "U | evidence=[] | answer=unknown",
        "S | evidence=[P06] | answer=",
        "S | evidence=[P06] | answer=north | explanation=guess",
    ],
)
def test_strict_parser_rejects_malformed_or_inconsistent_output(text: str) -> None:
    assert not parse_prediction(text).parse_valid


def test_official_answer_alias_and_support_semantics() -> None:
    assert normalize_answer(" The North, ") == "north"
    assert answer_scores("north", ("northern direction", "the north")) == (1.0, 1.0)
    support_em, support_f1, precision, recall = support_scores((6, 10), (6, 12))
    assert support_em == 0.0
    assert support_f1 == pytest.approx(0.5)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert support_scores((), ())[:2] == (1.0, 1.0)


def _gold(orbit: str, state: str) -> dict[str, object]:
    return {
        "id": f"{orbit}::{state}",
        "orbit_id": orbit,
        "state": state,
        "answer": "the north",
        "answer_aliases": ["north"],
        "gold_support_idxs": [6, 10] if state != "M" else [6],
        "answerable": state != "M",
    }


def test_evaluate_records_uses_full_orbit_sufficiency_semantics() -> None:
    gold = [_gold(orbit, state) for orbit in ("o1", "o2") for state in ("C", "D", "M")]
    predictions = [
        {"id": "o1::C", "prediction": "S | evidence=[P06,P10] | answer=north"},
        {"id": "o1::D", "prediction": "S | evidence=[P06,P10] | answer=north"},
        {
            "id": "o1::M",
            "prediction": "U | evidence=[] | answer=INSUFFICIENT_EVIDENCE",
        },
        {"id": "o2::C", "prediction": "S | evidence=[P06,P10] | answer=north"},
        {
            "id": "o2::D",
            "prediction": "U | evidence=[] | answer=INSUFFICIENT_EVIDENCE",
        },
        {"id": "o2::M", "prediction": "S | evidence=[P06] | answer=north"},
    ]
    report = evaluate_records(gold, predictions)

    assert report["counts"] == {"rows": 6, "orbits": 2}
    assert report["state_metrics"]["C"]["answer_f1"] == 1.0
    assert report["state_metrics"]["D"]["answer_f1"] == 0.5
    assert report["state_metrics"]["M"]["answer_f1"] is None
    assert report["state_metrics"]["M"]["support_f1"] is None
    assert report["orbit_metrics"]["cd_min_f1"] == 0.5
    assert report["orbit_metrics"]["orbit_answer_suff_f1"] == 0.5
    assert report["orbit_metrics"]["orbit_support_suff_f1"] == 0.5
    assert report["orbit_metrics"]["false_refusal_rate"] == 0.25
    assert report["orbit_metrics"]["m_refusal_rate"] == 0.5


def test_invalid_parse_scores_zero_without_repair() -> None:
    gold = [_gold("o1", state) for state in ("C", "D", "M")]
    predictions = [
        {"id": "o1::C", "prediction": "answer is north"},
        {"id": "o1::D", "prediction": "S | evidence=[P06,P10] | answer=north"},
        {
            "id": "o1::M",
            "prediction": "U | evidence=[] | answer=INSUFFICIENT_EVIDENCE",
        },
    ]
    report = evaluate_records(gold, predictions)
    assert report["state_metrics"]["C"]["parse_rate"] == 0.0
    assert report["state_metrics"]["C"]["answer_f1"] == 0.0
    assert report["orbit_metrics"]["parse_rate"] == pytest.approx(2 / 3)
    assert report["orbit_metrics"]["orbit_answer_suff_f1"] == 0.0


def _binding(arm: str, checkpoint_character: str) -> dict[str, object]:
    observed = {
        "arm_id": arm,
        "dataset_manifest_sha256": "a" * 64,
        "split_artifact_sha256": "b" * 64,
        "predictions_sha256": "c" * 64,
        "protocol_sha256": "d" * 64,
        "checkpoint_sha256": checkpoint_character * 64,
    }
    return {"passed": True, "checks": {"synthetic": True}, "observed": observed, "expected": {}}


def test_exact_binding_detects_file_or_expectation_mismatch() -> None:
    observed = _binding("control", "e")["observed"]
    expected = {
        "arm_id": "control",
        "dataset_manifest_sha256": "a" * 64,
        "split_artifact_sha256": "b" * 64,
        "protocol_sha256": "d" * 64,
        "checkpoint_sha256": "e" * 64,
    }
    valid = verify_run_binding(
        observed,
        expected,
        actual_split_sha256="b" * 64,
        actual_predictions_sha256="c" * 64,
    )
    assert valid["passed"]
    invalid = verify_run_binding(
        observed,
        {**expected, "arm_id": "treatment"},
        actual_split_sha256="b" * 64,
        actual_predictions_sha256="c" * 64,
    )
    assert not invalid["passed"]


def _fake_evaluation(arm: str, treatment: bool) -> dict[str, object]:
    orbit_scores: dict[str, dict[str, float]] = {}
    for index in range(20):
        base = 0.50 + (index % 2) * 0.05
        gain = 0.10 if treatment else 0.0
        orbit_scores[f"o{index:02d}"] = {
            "cd_min_f1": base + gain,
            "orbit_answer_suff_f1": base + gain,
            "d_answer_f1": base + gain,
            "c_answer_f1": 0.70,
            "false_refusal_rate": 0.05,
            "m_refusal_rate": 0.90 if treatment else 0.80,
            "orbit_support_suff_f1": 0.65,
            "parse_rate": 1.0,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "binding": _binding(arm, "f" if treatment else "e"),
        "run_integrity": True,
        "orbit_scores": orbit_scores,
    }


def test_paired_bootstrap_is_frozen_and_gate_comparison_can_pass() -> None:
    first = paired_cluster_bootstrap([0.1, 0.2], [0.2, 0.3])
    second = paired_cluster_bootstrap([0.1, 0.2], [0.2, 0.3])
    assert first == second
    assert first.delta == pytest.approx(0.1)

    comparison = compare_evaluations(
        _fake_evaluation("CONTROL", treatment=False),
        _fake_evaluation("HopPAIR", treatment=True),
    )
    assert comparison["bootstrap"] == {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "unit": "orbit",
    }
    assert comparison["metrics"]["cd_min_f1"]["delta_pp"] == pytest.approx(10.0)
    assert comparison["decision"] == "GO"
    assert all(gate["passed"] for gate in comparison["gates"].values())


def test_any_failed_gate_forces_stop() -> None:
    control = _fake_evaluation("CONTROL", treatment=False)
    treatment = _fake_evaluation("HopPAIR", treatment=True)
    treatment = deepcopy(treatment)
    for scores in treatment["orbit_scores"].values():
        scores["parse_rate"] = 0.98
    comparison = compare_evaluations(control, treatment)
    assert not comparison["gates"]["parse_rate"]["passed"]
    assert comparison["decision"] == "STOP"


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(path: Path, digest_character: str) -> dict[str, object]:
    return {
        "path": str(path),
        "files": [
            {
                "path": "artifact.bin",
                "size_bytes": 1,
                "sha256": digest_character * 64,
            }
        ],
        "sha256": digest_character * 64,
    }


def _production_receipt(tmp_path: Path) -> dict[str, object]:
    roots = {
        key: str(tmp_path / "outputs" / key)
        for key in (
            "control_train",
            "hoppair_train",
            "base_dev_generation",
            "control_dev_generation",
            "hoppair_dev_generation",
            "base_dev_evaluation",
            "control_dev_evaluation",
            "hoppair_dev_evaluation",
            "dev_comparison",
        )
    }
    gold_path = tmp_path / "prepared" / "dev.jsonl"
    gold_rows: list[dict[str, object]] = []
    for orbit_index in range(400):
        orbit_id = f"orbit-{orbit_index:04d}"
        for state in ("C", "D", "M"):
            row = _gold(orbit_id, state)
            row.update(
                {
                    "split": "dev",
                    "training_read_allowed": True,
                    "sealed": False,
                }
            )
            gold_rows.append(row)
    gold_path.parent.mkdir(parents=True)
    _write_jsonl(gold_path, gold_rows)
    validated = {
        "path": str(tmp_path / "pilot_v2_launch_receipt.json"),
        "sha256": "1" * 64,
        "protocol_path": str(tmp_path / "protocol.json"),
        "prepared_manifest_path": str(tmp_path / "prepared" / "manifest.json"),
        "train_path": str(tmp_path / "prepared" / "train.jsonl"),
        "dev_path": str(gold_path),
    }
    hardware = {
        "cuda_visible_devices": "0",
        "uuid": "GPU-synthetic",
        "name": "Synthetic GPU",
        "driver_version": "0",
        "driver_cuda_version": "0",
        "torch_cuda_version": "0",
        "compute_capability": "8.0",
    }
    return {
        "schema_version": "support-orbit-musique.launch-receipt.v2",
        "status": "READY_FOR_GPU",
        "_validated": validated,
        "output_roots": roots,
        "exact_bindings": {
            "dev": {
                "path": str(gold_path),
                "sha256": _file_sha256(gold_path),
                "orbits": 400,
                "records": 1_200,
            },
            "prepared_manifest": {
                "path": validated["prepared_manifest_path"],
                "sha256": "2" * 64,
            },
            "protocol": {"path": validated["protocol_path"], "sha256": "3" * 64},
            "schedule_sha256": "4" * 64,
            "source_lock_sha256": "5" * 64,
            "base_model": {"path": str(tmp_path / "model"), "sha256": "6" * 64},
        },
        "runtime": {
            "max_length": 6_144,
            "generation": {
                "batch_size": 4,
                "max_new_tokens": 128,
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "length_preflight_status": "PASS",
            },
        },
        "hardware_lock": {arm: dict(hardware) for arm in ("BASE", "CONTROL", "HopPAIR")},
    }


def _write_generation(
    receipt: dict[str, object],
    arm: str,
    *,
    correct_answer: bool = True,
) -> Path:
    root_key = {
        "BASE": "base_dev_generation",
        "CONTROL": "control_dev_generation",
        "HopPAIR": "hoppair_dev_generation",
    }[arm]
    root = Path(receipt["output_roots"][root_key])
    root.mkdir(parents=True)
    gold_path = Path(receipt["exact_bindings"]["dev"]["path"])
    gold_rows = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines()]
    predictions: list[dict[str, object]] = []
    for row in gold_rows:
        if row["state"] == "M":
            prediction = "U | evidence=[] | answer=INSUFFICIENT_EVIDENCE"
        else:
            answer = "north" if correct_answer else "south"
            prediction = f"S | evidence=[P06,P10] | answer={answer}"
        predictions.append(
            {
                "id": row["id"],
                "prediction": prediction,
                "arm_id": arm,
                "dataset_manifest_sha256": receipt["exact_bindings"]["prepared_manifest"][
                    "sha256"
                ],
            }
        )
    prediction_path = root / "predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    fairness = {
        "passed": True,
        "checks": {
            name: True
            for name in (
                "initial_trainable_parameters_exact",
                "schedule_exact",
                "train_artifact_exact",
                "static_model_exact",
                "tokenizer_exact",
                "runtime_exact",
                "lora_exact",
            )
        },
        "diagnostics": {"checkpoints_distinct": True},
        "initial_trainable_parameters_sha256": "7" * 64,
        "schedule_sha256": receipt["exact_bindings"]["schedule_sha256"],
        "control_run_sha256": "8" * 64,
        "hoppair_run_sha256": "9" * 64,
    }
    checkpoint = receipt["exact_bindings"]["base_model"]["sha256"]
    adapter: dict[str, object] | None = None
    training_run: dict[str, object] | None = None
    if arm != "BASE":
        train_key = "control_train" if arm == "CONTROL" else "hoppair_train"
        checkpoint_character = "a" if arm == "CONTROL" else "b"
        adapter = _artifact(
            Path(receipt["output_roots"][train_key]) / "final_adapter",
            checkpoint_character,
        )
        checkpoint = adapter["sha256"]
        training_run = {
            "path": str(Path(receipt["output_roots"][train_key]) / "run_manifest.json"),
            "sha256": fairness[
                "control_run_sha256" if arm == "CONTROL" else "hoppair_run_sha256"
            ],
            "adapter": adapter,
        }
    binding = {
        "arm_id": arm,
        "dataset_manifest_sha256": receipt["exact_bindings"]["prepared_manifest"]["sha256"],
        "split_artifact_sha256": receipt["exact_bindings"]["dev"]["sha256"],
        "predictions_sha256": _file_sha256(prediction_path),
        "protocol_sha256": receipt["exact_bindings"]["protocol"]["sha256"],
        "checkpoint_sha256": checkpoint,
        "launch_receipt_sha256": receipt["_validated"]["sha256"],
        "schedule_sha256": receipt["exact_bindings"]["schedule_sha256"],
    }
    if arm != "BASE":
        binding["initialization_sha256"] = fairness["initial_trainable_parameters_sha256"]
    binding_path = root / "binding.json"
    _write_json(binding_path, binding)
    manifest = {
        "schema_version": "support-orbit-musique.generation.v1",
        "status": "completed",
        "started_at": "2026-08-14T00:00:00+00:00",
        "completed_at": "2026-08-14T00:01:00+00:00",
        "arm": arm,
        "receipt": receipt["_validated"],
        "input": {
            **receipt["exact_bindings"]["dev"],
            "prompt_tokens_max": 100,
            "prompt_tokens_total": 40_000,
        },
        "model_identity": {
            "base_model": _artifact(Path(receipt["exact_bindings"]["base_model"]["path"]), "6"),
            "adapter": adapter,
        },
        "hardware_identity": receipt["hardware_lock"][arm],
        "training_run": training_run,
        "paired_training_fairness": fairness,
        "decoding": {
            "do_sample": False,
            "batch_size": 4,
            "max_new_tokens": 128,
            "num_beams": 1,
            "num_return_sequences": 1,
            "length_preflight_status": "PASS",
            "budget_exhausted_rows": 0,
        },
        "predictions": {
            "path": str(prediction_path),
            "sha256": _file_sha256(prediction_path),
            "rows": 1_200,
            "unique_ids": 1_200,
            "generated_tokens_total": 12_000,
            "generated_tokens_max": 10,
        },
        "binding": {"path": str(binding_path), "sha256": _file_sha256(binding_path)},
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_production_evaluation_accepts_only_exact_400_orbit_fixture(tmp_path: Path) -> None:
    receipt = _production_receipt(tmp_path)
    manifest = _write_generation(receipt, "CONTROL")
    report, destination = evaluate_production_arm(
        receipt,
        arm="CONTROL",
        generation_manifest_path=manifest,
        write=False,
    )
    assert destination == Path(receipt["output_roots"]["control_dev_evaluation"]) / "evaluation.json"
    assert report["counts"] == {"rows": 1_200, "orbits": 400}
    assert report["run_integrity"]
    assert report["production"]["arm"] == "CONTROL"


def test_base_role_uses_only_the_receipt_bound_base_checkpoint(tmp_path: Path) -> None:
    receipt = _production_receipt(tmp_path)
    manifest = _write_generation(receipt, "BASE")
    report, _ = evaluate_production_arm(
        receipt,
        arm="BASE",
        generation_manifest_path=manifest,
        write=False,
    )
    assert report["binding"]["observed"]["arm_id"] == "BASE"
    assert (
        report["binding"]["observed"]["checkpoint_sha256"]
        == receipt["exact_bindings"]["base_model"]["sha256"]
    )


def test_production_cli_surface_has_no_arbitrary_data_or_comparison_paths() -> None:
    parsed = evaluate_dev.parser().parse_args(
        [
            "--launch-receipt",
            "/receipt",
            "--generation-manifest",
            "/manifest",
            "--arm",
            "CONTROL",
        ]
    )
    assert vars(parsed) == {
        "launch_receipt": Path("/receipt"),
        "generation_manifest": Path("/manifest"),
        "arm": "CONTROL",
    }
    assert vars(compare_dev.parser().parse_args(["--launch-receipt", "/receipt"])) == {
        "launch_receipt": Path("/receipt")
    }
    with pytest.raises(SystemExit):
        evaluate_dev.parser().parse_args(
            [
                "--launch-receipt",
                "/receipt",
                "--generation-manifest",
                "/manifest",
                "--arm",
                "HOPPAIR",
            ]
        )
    with pytest.raises(SystemExit):
        compare_dev.parser().parse_args(
            ["--launch-receipt", "/receipt", "--control", "/arbitrary"]
        )


def test_cli_validates_receipt_before_evaluator_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached_evaluator = False

    def fail_receipt(*args: object, **kwargs: object) -> dict[str, object]:
        raise ValueError("source lock mismatch")

    def forbidden_evaluation(*args: object, **kwargs: object) -> object:
        nonlocal reached_evaluator
        reached_evaluator = True
        raise AssertionError("generation input reached before receipt validation")

    monkeypatch.setattr(evaluate_dev, "validate_launch_receipt", fail_receipt)
    monkeypatch.setattr(evaluate_dev, "evaluate_production_arm", forbidden_evaluation)
    with pytest.raises(ValueError, match="source lock mismatch"):
        evaluate_dev.main(
            [
                "--launch-receipt",
                "/receipt",
                "--generation-manifest",
                "/manifest",
                "--arm",
                "CONTROL",
            ]
        )
    assert not reached_evaluator


def test_protected_path_is_rejected_before_open(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def forbidden_open(*args: object, **kwargs: object) -> int:
        nonlocal calls
        calls += 1
        raise AssertionError("protected path reached os.open")

    monkeypatch.setattr("support_orbit_musique.evaluation.os.open", forbidden_open)
    with pytest.raises(ValueError, match="protected path before open"):
        read_bound_json_object("/does/not/exist/shadow.sealed.json", description="protected")
    assert calls == 0
    with pytest.raises(ValueError, match="protected"):
        reject_protected_path("/does/not/exist/official_test.jsonl")


def test_same_descriptor_reader_refuses_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    link = tmp_path / "link.json"
    _write_json(target, {"safe": True})
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink component"):
        read_bound_json_object(link, description="symlink fixture")


def test_same_descriptor_reader_refuses_parent_symlink_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    target = real_parent / "payload.json"
    _write_json(target, {"safe": True})
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    calls = 0
    real_open = __import__("os").open

    def counted_open(*args: object, **kwargs: object) -> int:
        nonlocal calls
        calls += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr("support_orbit_musique.evaluation.os.open", counted_open)
    with pytest.raises(ValueError, match="symlink component"):
        read_bound_json_object(alias_parent / "payload.json", description="parent alias fixture")
    assert calls == 0


def test_swapped_arm_manifest_is_rejected_before_manifest_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _production_receipt(tmp_path)
    hoppair_manifest = _write_generation(receipt, "HopPAIR")
    calls = 0
    real_open = __import__("os").open

    def counted_open(*args: object, **kwargs: object) -> int:
        nonlocal calls
        calls += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr("support_orbit_musique.evaluation.os.open", counted_open)
    with pytest.raises(ValueError, match="noncanonical"):
        load_generation_context(
            receipt,
            arm="CONTROL",
            generation_manifest_path=hoppair_manifest,
        )
    assert calls == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["decoding"].pop("num_beams"), "fields drifted"),
        (lambda value: value["predictions"].update({"rows": 1_199}), "1200 unique"),
        (lambda value: value["decoding"].update({"do_sample": True}), "exact greedy"),
    ],
)
def test_missing_field_wrong_count_and_decoding_fail_closed(
    tmp_path: Path, mutation: object, message: str
) -> None:
    receipt = _production_receipt(tmp_path)
    manifest_path = _write_generation(receipt, "CONTROL")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match=message):
        load_generation_context(receipt, arm="CONTROL", generation_manifest_path=manifest_path)


def test_wrong_prediction_hash_fails_before_prediction_json_parse(tmp_path: Path) -> None:
    receipt = _production_receipt(tmp_path)
    manifest_path = _write_generation(receipt, "CONTROL")
    prediction_path = manifest_path.parent / "predictions.jsonl"
    prediction_path.write_bytes(prediction_path.read_bytes() + b"not-json\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_production_arm(
            receipt,
            arm="CONTROL",
            generation_manifest_path=manifest_path,
            write=False,
        )


def _rebind_changed_predictions(manifest_path: Path) -> None:
    prediction_path = manifest_path.parent / "predictions.jsonl"
    binding_path = manifest_path.parent / "binding.json"
    prediction_sha = _file_sha256(prediction_path)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["predictions_sha256"] = prediction_sha
    _write_json(binding_path, binding)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["predictions"]["sha256"] = prediction_sha
    manifest["binding"]["sha256"] = _file_sha256(binding_path)
    _write_json(manifest_path, manifest)


def test_inline_prediction_arm_and_manifest_fields_are_mandatory(tmp_path: Path) -> None:
    receipt = _production_receipt(tmp_path)
    manifest_path = _write_generation(receipt, "CONTROL")
    prediction_path = manifest_path.parent / "predictions.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    rows[0].pop("arm_id")
    _write_jsonl(prediction_path, rows)
    _rebind_changed_predictions(manifest_path)
    with pytest.raises(ValueError, match="fields differ from the exact schema"):
        evaluate_production_arm(
            receipt,
            arm="CONTROL",
            generation_manifest_path=manifest_path,
            write=False,
        )


def test_body_count_is_rechecked_after_manifest_count_passes(tmp_path: Path) -> None:
    receipt = _production_receipt(tmp_path)
    manifest_path = _write_generation(receipt, "CONTROL")
    prediction_path = manifest_path.parent / "predictions.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    _write_jsonl(prediction_path, rows[:-3])
    _rebind_changed_predictions(manifest_path)
    with pytest.raises(ValueError, match="exactly 1200"):
        evaluate_production_arm(
            receipt,
            arm="CONTROL",
            generation_manifest_path=manifest_path,
            write=False,
        )


def test_compare_recomputes_only_exact_control_and_hoppair_reports(tmp_path: Path) -> None:
    receipt = _production_receipt(tmp_path)
    control_manifest = _write_generation(receipt, "CONTROL", correct_answer=False)
    hoppair_manifest = _write_generation(receipt, "HopPAIR", correct_answer=True)
    for arm, manifest in (("CONTROL", control_manifest), ("HopPAIR", hoppair_manifest)):
        report, destination = evaluate_production_arm(
            receipt,
            arm=arm,
            generation_manifest_path=manifest,
            write=True,
        )
        assert report["run_integrity"]
        assert destination.is_file()
    comparison, destination = compare_production(receipt, write=False)
    assert destination == Path(receipt["output_roots"]["dev_comparison"]) / "dev_comparison.json"
    assert comparison["production"]["roles"] == {
        "control": "CONTROL",
        "treatment": "HopPAIR",
    }
    assert comparison["counts"]["paired_orbits"] == 400
    assert comparison["decision"] == "GO"
