"""Frozen Support-Orbit metrics, paired bootstrap, and decision gates."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .official_adapter import OfficialGold, adapt_official_gold, answer_scores, support_scores
from .parsing import ParsedPrediction, parse_prediction


SCHEMA_VERSION = "support-orbit-metrics-v1"
STATES = ("C", "D", "M")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_814
REQUIRED_BINDING_FIELDS = (
    "arm_id",
    "dataset_manifest_sha256",
    "split_artifact_sha256",
    "predictions_sha256",
    "protocol_sha256",
    "checkpoint_sha256",
)
REQUIRED_EXPECTED_BINDING_FIELDS = (
    "arm_id",
    "dataset_manifest_sha256",
    "split_artifact_sha256",
    "protocol_sha256",
    "checkpoint_sha256",
)
COMPARISON_METRICS = (
    "cd_min_f1",
    "orbit_answer_suff_f1",
    "d_answer_f1",
    "c_answer_f1",
    "false_refusal_rate",
    "m_refusal_rate",
    "orbit_support_suff_f1",
    "parse_rate",
)


@dataclass(frozen=True, slots=True)
class StateScore:
    state: str
    parse_valid: float
    answer_em: float | None
    answer_f1: float | None
    support_em: float | None
    support_f1: float | None
    predicted_answerable: bool | None
    answerability_correct: float
    refusal: float


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    control: float
    treatment: float
    delta: float
    delta_pp: float
    ci_low: float
    ci_high: float
    ci_low_pp: float
    ci_high_pp: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_run_binding(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    actual_split_sha256: str,
    actual_predictions_sha256: str,
) -> dict[str, Any]:
    """Verify an observed run receipt against files and an exact expectation.

    ``expected`` may contain additional frozen keys; every one is compared
    exactly.  The observed receipt must always include the six cryptographic
    arm/provenance fields in ``REQUIRED_BINDING_FIELDS``.
    """

    checks: dict[str, bool] = {}
    for field in REQUIRED_BINDING_FIELDS:
        checks[f"required:{field}"] = field in observed and bool(observed[field])
    checks["type:arm_id"] = isinstance(observed.get("arm_id"), str)
    for field in REQUIRED_BINDING_FIELDS[1:]:
        checks[f"sha256:{field}"] = _is_sha256(observed.get(field))
    checks["file:split_artifact_sha256"] = (
        observed.get("split_artifact_sha256") == actual_split_sha256
    )
    checks["file:predictions_sha256"] = (
        observed.get("predictions_sha256") == actual_predictions_sha256
    )
    for field in REQUIRED_EXPECTED_BINDING_FIELDS:
        checks[f"expected_required:{field}"] = field in expected and bool(expected[field])
    for key, expected_value in sorted(expected.items()):
        checks[f"expected:{key}"] = key in observed and observed[key] == expected_value
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": dict(observed),
        "expected": dict(expected),
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _score_state(gold: OfficialGold, parsed: ParsedPrediction) -> StateScore:
    predicted_answerable = parsed.predicted_answerable
    action_correct = float(
        parsed.parse_valid and predicted_answerable is not None and predicted_answerable == gold.answerable
    )
    answer_em: float | None = None
    answer_f1: float | None = None
    support_em: float | None = None
    support_f1: float | None = None
    if gold.answerable:
        predicted_answer = parsed.answer if parsed.parse_valid else ""
        answer_em, answer_f1 = answer_scores(predicted_answer, gold.answers)
        predicted_support = parsed.support_indices if parsed.parse_valid else ()
        support_em, support_f1, _, _ = support_scores(predicted_support, gold.support_indices)
    return StateScore(
        state=gold.state,
        parse_valid=float(parsed.parse_valid),
        answer_em=answer_em,
        answer_f1=answer_f1,
        support_em=support_em,
        support_f1=support_f1,
        predicted_answerable=predicted_answerable,
        answerability_correct=action_correct,
        refusal=float(parsed.refused),
    )


def _validate_orbits(gold_by_id: Mapping[str, OfficialGold]) -> dict[str, dict[str, OfficialGold]]:
    grouped: dict[str, dict[str, OfficialGold]] = defaultdict(dict)
    for gold in gold_by_id.values():
        if gold.state in grouped[gold.orbit_id]:
            raise ValueError(f"duplicate state {gold.state} in orbit {gold.orbit_id}")
        grouped[gold.orbit_id][gold.state] = gold
    for orbit_id, states in grouped.items():
        if set(states) != set(STATES):
            raise ValueError(f"orbit {orbit_id} does not contain exactly C/D/M")
        if not states["C"].answerable or not states["D"].answerable or states["M"].answerable:
            raise ValueError(f"orbit {orbit_id} violates C=true,D=true,M=false")
    if not grouped:
        raise ValueError("evaluation set is empty")
    return dict(grouped)


def _state_summary(scores: Sequence[StateScore]) -> dict[str, float | int | None]:
    answer_em = [score.answer_em for score in scores if score.answer_em is not None]
    answer_f1 = [score.answer_f1 for score in scores if score.answer_f1 is not None]
    support_em = [score.support_em for score in scores if score.support_em is not None]
    support_f1 = [score.support_f1 for score in scores if score.support_f1 is not None]
    return {
        "count": len(scores),
        "parse_rate": _mean([score.parse_valid for score in scores]),
        "answer_em": _mean(answer_em) if answer_em else None,
        "answer_f1": _mean(answer_f1) if answer_f1 else None,
        "support_em": _mean(support_em) if support_em else None,
        "support_f1": _mean(support_f1) if support_f1 else None,
        "answerability_accuracy": _mean([score.answerability_correct for score in scores]),
        "refusal_rate": _mean([score.refusal for score in scores]),
        "false_refusal_rate": (
            _mean([score.refusal for score in scores])
            if scores and scores[0].state in {"C", "D"}
            else None
        ),
    }


def _orbit_score(states: Mapping[str, StateScore]) -> dict[str, float]:
    c_score, d_score, m_score = (states[state] for state in STATES)
    assert c_score.answer_f1 is not None and d_score.answer_f1 is not None
    assert c_score.support_f1 is not None and d_score.support_f1 is not None

    cm_sufficiency = c_score.predicted_answerable is True and m_score.predicted_answerable is False
    dm_sufficiency = d_score.predicted_answerable is True and m_score.predicted_answerable is False
    orbit_sufficiency = cm_sufficiency and dm_sufficiency
    cd_min_f1 = min(c_score.answer_f1, d_score.answer_f1)
    cd_min_support_f1 = min(c_score.support_f1, d_score.support_f1)
    return {
        "c_answer_f1": c_score.answer_f1,
        "d_answer_f1": d_score.answer_f1,
        "c_support_f1": c_score.support_f1,
        "d_support_f1": d_score.support_f1,
        "cd_min_f1": cd_min_f1,
        "cd_min_support_f1": cd_min_support_f1,
        "cm_answer_suff_f1": c_score.answer_f1 if cm_sufficiency else 0.0,
        "dm_answer_suff_f1": d_score.answer_f1 if dm_sufficiency else 0.0,
        "cm_support_suff_f1": c_score.support_f1 if cm_sufficiency else 0.0,
        "dm_support_suff_f1": d_score.support_f1 if dm_sufficiency else 0.0,
        "orbit_answer_suff_f1": cd_min_f1 if orbit_sufficiency else 0.0,
        "orbit_support_suff_f1": cd_min_support_f1 if orbit_sufficiency else 0.0,
        "false_refusal_rate": (c_score.refusal + d_score.refusal) / 2.0,
        "m_refusal_rate": m_score.refusal,
        "parse_rate": (c_score.parse_valid + d_score.parse_valid + m_score.parse_valid) / 3.0,
    }


def evaluate_records(
    gold_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an exactly aligned C/D/M dev set.

    Predictions have the strict minimal schema ``{"id": ..., "prediction": ...}``.
    Alignment is by exact id set, never by line position or substring recovery.
    """

    gold_by_id: dict[str, OfficialGold] = {}
    for record in gold_records:
        gold = adapt_official_gold(record)
        if gold.instance_id in gold_by_id:
            raise ValueError(f"duplicate gold id: {gold.instance_id}")
        gold_by_id[gold.instance_id] = gold
    orbits = _validate_orbits(gold_by_id)

    binding_report: dict[str, Any] = dict(
        binding or {"passed": False, "checks": {"binding_supplied": False}}
    )
    if binding:
        observed_binding = binding.get("observed", {})
        binding_checks = dict(binding.get("checks", {}))
        for field in ("arm_id", "dataset_manifest_sha256"):
            inline_values = [record[field] for record in prediction_records if field in record]
            binding_checks[f"prediction_rows:{field}"] = not inline_values or (
                len(inline_values) == len(prediction_records)
                and all(value == observed_binding.get(field) for value in inline_values)
            )
        binding_report["checks"] = binding_checks
        binding_report["passed"] = bool(binding.get("passed")) and all(binding_checks.values())

    predictions_by_id: dict[str, ParsedPrediction] = {}
    for record in prediction_records:
        if set(record) - {"id", "prediction", "arm_id", "dataset_manifest_sha256"}:
            unexpected = sorted(set(record) - {"id", "prediction", "arm_id", "dataset_manifest_sha256"})
            raise ValueError(f"unexpected prediction fields: {unexpected}")
        instance_id = str(record["id"])
        if instance_id in predictions_by_id:
            raise ValueError(f"duplicate prediction id: {instance_id}")
        predictions_by_id[instance_id] = parse_prediction(record["prediction"])
    if set(predictions_by_id) != set(gold_by_id):
        missing = sorted(set(gold_by_id) - set(predictions_by_id))
        extra = sorted(set(predictions_by_id) - set(gold_by_id))
        raise ValueError(f"prediction/gold id mismatch; missing={missing[:5]}, extra={extra[:5]}")

    state_scores: dict[str, list[StateScore]] = {state: [] for state in STATES}
    score_by_orbit_state: dict[str, dict[str, StateScore]] = defaultdict(dict)
    for instance_id, gold in gold_by_id.items():
        score = _score_state(gold, predictions_by_id[instance_id])
        state_scores[gold.state].append(score)
        score_by_orbit_state[gold.orbit_id][gold.state] = score

    orbit_scores = {
        orbit_id: _orbit_score(score_by_orbit_state[orbit_id]) for orbit_id in sorted(orbits)
    }
    orbit_metrics = {
        name: _mean([scores[name] for scores in orbit_scores.values()])
        for name in next(iter(orbit_scores.values()))
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "binding": binding_report,
        "run_integrity": binding_report.get("passed") is True,
        "counts": {"rows": len(gold_by_id), "orbits": len(orbits)},
        "state_metrics": {
            state: _state_summary(state_scores[state]) for state in STATES
        },
        "orbit_metrics": orbit_metrics,
        "orbit_scores": orbit_scores,
    }


def paired_cluster_bootstrap(
    control: Sequence[float],
    treatment: Sequence[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapInterval:
    """Orbit-clustered paired percentile bootstrap with frozen defaults."""

    if len(control) != len(treatment) or not control:
        raise ValueError("paired bootstrap requires equal, nonempty vectors")
    control_array = np.asarray(control, dtype=np.float64)
    treatment_array = np.asarray(treatment, dtype=np.float64)
    paired_deltas = treatment_array - control_array
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    chunk = 512
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        indices = generator.integers(0, len(paired_deltas), size=(stop - start, len(paired_deltas)))
        bootstrap_means[start:stop] = paired_deltas[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    delta = float(paired_deltas.mean())
    return BootstrapInterval(
        control=float(control_array.mean()),
        treatment=float(treatment_array.mean()),
        delta=delta,
        delta_pp=100.0 * delta,
        ci_low=float(low),
        ci_high=float(high),
        ci_low_pp=100.0 * float(low),
        ci_high_pp=100.0 * float(high),
    )


def _comparison_integrity(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, bool]:
    control_binding = control.get("binding", {}).get("observed", {})
    treatment_binding = treatment.get("binding", {}).get("observed", {})
    control_orbits = set(control.get("orbit_scores", {}))
    treatment_orbits = set(treatment.get("orbit_scores", {}))
    return {
        "control_run_integrity": control.get("run_integrity") is True,
        "treatment_run_integrity": treatment.get("run_integrity") is True,
        "schema_match": control.get("schema_version") == treatment.get("schema_version") == SCHEMA_VERSION,
        "orbit_ids_exact": control_orbits == treatment_orbits and bool(control_orbits),
        "arm_roles_exact": control_binding.get("arm_id") == "CONTROL"
        and treatment_binding.get("arm_id") == "HopPAIR",
        "dataset_manifest_exact": control_binding.get("dataset_manifest_sha256")
        == treatment_binding.get("dataset_manifest_sha256"),
        "split_artifact_exact": control_binding.get("split_artifact_sha256")
        == treatment_binding.get("split_artifact_sha256"),
        "protocol_exact": control_binding.get("protocol_sha256")
        == treatment_binding.get("protocol_sha256"),
    }


def compare_evaluations(
    control: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the frozen paired comparison and all predeclared gates."""

    control_orbits = control.get("orbit_scores", {})
    treatment_orbits = treatment.get("orbit_scores", {})
    if set(control_orbits) != set(treatment_orbits) or not control_orbits:
        raise ValueError("control/treatment orbit ids must match exactly and be nonempty")
    orbit_ids = sorted(control_orbits)
    intervals: dict[str, BootstrapInterval] = {}
    for metric in COMPARISON_METRICS:
        intervals[metric] = paired_cluster_bootstrap(
            [float(control_orbits[orbit_id][metric]) for orbit_id in orbit_ids],
            [float(treatment_orbits[orbit_id][metric]) for orbit_id in orbit_ids],
        )

    integrity_checks = _comparison_integrity(control, treatment)
    cd = intervals["cd_min_f1"]
    orbit_answer = intervals["orbit_answer_suff_f1"]
    d_answer = intervals["d_answer_f1"]
    c_answer = intervals["c_answer_f1"]
    false_refusal = intervals["false_refusal_rate"]
    m_refusal = intervals["m_refusal_rate"]
    orbit_support = intervals["orbit_support_suff_f1"]
    parse_rate = intervals["parse_rate"]
    gates = {
        "cd_min_f1_gain": {
            "passed": cd.delta >= 0.04 and cd.ci_low > 0.0,
            "rule": "delta>=0.04 and paired_bootstrap_ci_low>0",
        },
        "orbit_answer_suff_gain": {
            "passed": orbit_answer.delta >= 0.04 and orbit_answer.ci_low > 0.0,
            "rule": "delta>=0.04 and paired_bootstrap_ci_low>0",
        },
        "d_answer_f1_gain": {"passed": d_answer.delta >= 0.04, "rule": "delta>=0.04"},
        "c_answer_f1_noninferiority": {
            "passed": c_answer.delta >= -0.02,
            "rule": "delta>=-0.02",
        },
        "false_refusal_noninferiority": {
            "passed": false_refusal.delta <= 0.02,
            "rule": "delta<=0.02",
        },
        "m_refusal": {
            "passed": m_refusal.delta >= 0.0 and m_refusal.treatment >= 0.80,
            "rule": "delta>=0 and treatment>=0.80",
        },
        "orbit_support_suff_noninferiority": {
            "passed": orbit_support.delta >= -0.02,
            "rule": "delta>=-0.02",
        },
        "parse_rate": {"passed": parse_rate.treatment >= 0.99, "rule": "treatment>=0.99"},
        "run_integrity": {"passed": all(integrity_checks.values()), "rule": "all checks true"},
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    diagnostics = {
        "checkpoints_distinct": control.get("binding", {}).get("observed", {}).get(
            "checkpoint_sha256"
        )
        != treatment.get("binding", {}).get("observed", {}).get("checkpoint_sha256")
    }
    return {
        "schema_version": "support-orbit-comparison-v1",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "unit": "orbit"},
        "counts": {"paired_orbits": len(orbit_ids)},
        "integrity_checks": integrity_checks,
        "diagnostics": diagnostics,
        "metrics": {name: asdict(interval) for name, interval in intervals.items()},
        "gates": gates,
        "decision": "GO" if passed else "STOP",
    }


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "COMPARISON_METRICS",
    "SCHEMA_VERSION",
    "compare_evaluations",
    "evaluate_records",
    "paired_cluster_bootstrap",
    "sha256_file",
    "verify_run_binding",
]
