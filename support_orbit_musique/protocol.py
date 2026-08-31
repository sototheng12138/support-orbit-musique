"""Fail-closed validation for the Support-Orbit MuSiQue pilot protocol.

This module deliberately never opens the sealed-shadow JSONL.  It validates
the manifest, public CPU audits, the train/dev artifacts, and the shadow marker
that the builder emitted while streaming the sealed file once.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL_SCHEMA = "support-orbit-musique.protocol/v1"
EXPECTED_DATA_SCHEMA = "support-orbit-musique/v1"
PROTOCOL_TO_DATA_SCHEMA = {
    "support-orbit-musique.protocol/v1": "support-orbit-musique/v1",
    "support-orbit-musique.protocol/v2": "support-orbit-musique/v2",
}
EXPECTED_COUNTS = {
    "train_orbits": 1_920,
    "dev_orbits": 400,
    "shadow_orbits": 400,
    "train_records": 5_760,
    "dev_records": 1_200,
    "shadow_records": 1_200,
}
EXPECTED_SOURCE = {
    "sha256": "b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034",
    "rows": 39_876,
    "pairs": 19_938,
    "repository": "StonyBrookNLP/musique",
    "commit": "922ac98f19a201998dbdae6d7f2887a5258dbdeb",
    "license": "CC-BY-4.0",
}
SURFACE_AUC_CEILING = 0.60
SEMANTIC_CD_AUC_CEILING = 0.60
MAX_SEQUENCE_TOKENS = 6_144


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _check(checks: dict[str, bool], name: str, condition: object) -> None:
    checks[name] = condition is True


def validate_data_release(
    *,
    protocol: Mapping[str, Any],
    manifest_path: Path,
    audit_path: Path,
    shadow_marker_path: Path,
    verified: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every frozen data gate without opening the shadow body.

    A launch-receipt validator may supply objects and digests obtained through
    its same-open-file-descriptor verifier.  The legacy path remains available
    for standalone CPU audits and tests.
    """

    if verified is None:
        manifest = load_json(manifest_path)
        audit = load_json(audit_path)
        marker = load_json(shadow_marker_path)
        verified_hashes = {
            "manifest.json": sha256_file(manifest_path),
            "audit.json": sha256_file(audit_path),
            "SHADOW_SEALED.json": sha256_file(shadow_marker_path),
            "train.jsonl": sha256_file(manifest_path.parent / "train.jsonl"),
            "dev.jsonl": sha256_file(manifest_path.parent / "dev.jsonl"),
        }
    else:
        if set(verified) != {"manifest", "audit", "shadow_marker", "sha256"}:
            raise ValueError("verified release bundle has missing or unexpected fields")
        manifest = verified["manifest"]
        audit = verified["audit"]
        marker = verified["shadow_marker"]
        verified_hashes = verified["sha256"]
        if (
            not all(isinstance(value, Mapping) for value in (manifest, audit, marker))
            or not isinstance(verified_hashes, Mapping)
            or set(verified_hashes)
            != {
                "manifest.json",
                "audit.json",
                "SHADOW_SEALED.json",
                "train.jsonl",
                "dev.jsonl",
            }
        ):
            raise ValueError("verified release bundle is malformed")
    checks: dict[str, bool] = {}

    protocol_schema = protocol.get("schema_version")
    expected_data_schema = PROTOCOL_TO_DATA_SCHEMA.get(protocol_schema)
    _check(checks, "protocol_schema", protocol_schema in PROTOCOL_TO_DATA_SCHEMA)
    _check(
        checks,
        "data_schema",
        expected_data_schema is not None
        and manifest.get("schema_version") == expected_data_schema
        and audit.get("schema_version") == expected_data_schema
        and marker.get("schema_version") == expected_data_schema,
    )
    _check(
        checks,
        "source_exact",
        manifest.get("source")
        == EXPECTED_SOURCE
        | {
            "path": protocol["source_lock"]["allowed_train_path"],
            "dataset": "MuSiQue full v1.0 train",
        },
    )

    counts = manifest.get("counts", {})
    for name, expected in EXPECTED_COUNTS.items():
        _check(checks, f"count:{name}", counts.get(name) == expected)
    _check(checks, "selected_orbits", counts.get("selected_orbits") == 2_720)

    release = manifest.get("release_gate", {})
    _check(checks, "release_status", release.get("status") == "READY")
    for name in (
        "construction_invariants_pass",
        "surface_probe_pass",
        "semantic_C_D_probe_pass",
        "zero_truncation_pass",
    ):
        _check(checks, f"release:{name}", release.get(name) is True)
    _check(checks, "audit_release_matches", audit.get("release_gate") == release)

    construction = audit.get("construction", {})
    _check(checks, "construction_all", construction.get("all_invariants_pass") is True)
    invariant_counts = construction.get("invariant_counts", {})
    for name in (
        "orbits_with_exactly_two_nonsupport_replacements",
        "orbits_with_byte_identical_support_in_C_and_D",
        "orbits_with_exact_C_and_D_target",
        "orbits_with_same_split_donors",
    ):
        _check(checks, f"construction:{name}", invariant_counts.get(name) == 2_720)
    if expected_data_schema == "support-orbit-musique/v2":
        _check(
            checks,
            "construction:no_C_support_text_as_donor",
            invariant_counts.get("orbits_with_no_C_support_text_as_donor") == 2_720,
        )
        _check(
            checks,
            "construction:state_specific_official_aliases",
            invariant_counts.get("orbits_with_state_specific_official_aliases") == 2_720,
        )
        alias_policy = construction.get("answer_alias_policy", {})
        _check(
            checks,
            "construction:C_D_alias_policy",
            alias_policy.get("C_and_D") == "exact complete-row answer_aliases only",
        )
        _check(
            checks,
            "construction:M_alias_policy",
            alias_policy.get("M") == "exact missing-row answer_aliases only",
        )
    donor_ratio = construction.get("donor_length_ratio", {})
    _check(checks, "donor_ratio_ceiling", donor_ratio.get("ceiling") == 1.25)
    _check(checks, "donor_ratio_no_violations", donor_ratio.get("violations") == 0)
    _check(
        checks,
        "donor_ratio_max",
        isinstance(donor_ratio.get("max"), (int, float)) and float(donor_ratio["max"]) <= 1.25,
    )

    isolation = audit.get("component_isolation", {})
    _check(checks, "whole_components", isolation.get("whole_components_only") is True)
    overlaps = isolation.get("cross_split_key_overlap_counts", {})
    _check(
        checks,
        "cross_split_zero",
        set(overlaps) == {"train-dev", "train-shadow", "dev-shadow"}
        and all(value == 0 for value in overlaps.values()),
    )

    probes = audit.get("separability_probes", {})
    surface_pairs = probes.get("surface_only", {}).get("pairs", {})
    _check(
        checks,
        "surface_pair_set",
        set(surface_pairs) == {"C-D", "D-M", "C-M"},
    )
    for pair in ("C-D", "D-M", "C-M"):
        auc = surface_pairs.get(pair, {}).get("separability_auc")
        _check(
            checks,
            f"surface_auc:{pair}",
            isinstance(auc, (int, float)) and 0.5 <= float(auc) <= SURFACE_AUC_CEILING,
        )
    semantic_pairs = probes.get("semantic_question_overlap", {}).get("pairs", {})
    semantic_cd = semantic_pairs.get("C-D", {}).get("separability_auc")
    _check(
        checks,
        "semantic_auc:C-D",
        isinstance(semantic_cd, (int, float))
        and 0.5 <= float(semantic_cd) <= SEMANTIC_CD_AUC_CEILING,
    )

    token = manifest.get("tokenizer_audit", {})
    _check(checks, "token_zero_truncation", token.get("zero_truncation_pass") is True)
    _check(
        checks,
        "token_contract_exact",
        token.get("prepared_contract_byte_identical") is True
        and token.get("prepared_vs_canonical_mismatches") == 0,
    )
    _check(checks, "token_over_limit_zero", token.get("over_limit_records") == 0)
    _check(checks, "token_shadow_excluded", token.get("shadow_excluded") is True)
    _check(checks, "token_records", token.get("records") == 6_960)
    _check(
        checks,
        "token_max",
        isinstance(token.get("max_tokens"), int) and 0 < token["max_tokens"] <= MAX_SEQUENCE_TOKENS,
    )

    artifacts = manifest.get("artifact_sha256", {})
    _check(checks, "audit_hash", artifacts.get("audit.json") == verified_hashes["audit.json"])
    _check(
        checks,
        "shadow_marker_hash",
        artifacts.get("SHADOW_SEALED.json") == verified_hashes["SHADOW_SEALED.json"],
    )
    _check(checks, "shadow_marker_sealed", marker.get("sealed") is True)
    _check(checks, "shadow_marker_read_forbidden", marker.get("training_read_allowed") is False)
    _check(checks, "shadow_marker_orbits", marker.get("orbit_count") == 400)
    _check(checks, "shadow_marker_records", marker.get("record_count") == 1_200)
    _check(
        checks,
        "shadow_hash_transitive",
        marker.get("sha256") == artifacts.get("shadow.sealed.jsonl"),
    )

    for name in ("train.jsonl", "dev.jsonl"):
        path = manifest_path.parent / name
        _check(checks, f"artifact_exists:{name}", path.is_file())
        if path.is_file():
            _check(checks, f"artifact_hash:{name}", verified_hashes[name] == artifacts.get(name))

    protocol_lock = protocol.get("prepared_data_lock", {})
    expected_manifest_hash = protocol_lock.get("expected_manifest_sha256")
    _check(
        checks,
        "protocol_manifest_hash",
        isinstance(expected_manifest_hash, str)
        and expected_manifest_hash == verified_hashes["manifest.json"],
    )
    _check(
        checks,
        "protocol_status_frozen",
        protocol.get("protocol_status") == "FROZEN_PRE_GPU",
    )
    _check(
        checks,
        "protocol_data_status_ready",
        protocol_lock.get("release_status") == "READY",
    )
    locked_artifacts = protocol_lock.get("artifact_sha256", {})
    for name in ("train.jsonl", "dev.jsonl", "audit.json", "SHADOW_SEALED.json"):
        _check(
            checks,
            f"protocol_artifact_hash:{name}",
            locked_artifacts.get(name) == artifacts.get(name),
        )
    _check(
        checks,
        "protocol_artifact_hash:shadow_writer_digest",
        locked_artifacts.get("shadow.sealed.jsonl_writer_digest")
        == artifacts.get("shadow.sealed.jsonl"),
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_sha256": verified_hashes["manifest.json"],
        "shadow_body_opened": False,
    }


__all__ = [
    "EXPECTED_COUNTS",
    "EXPECTED_DATA_SCHEMA",
    "EXPECTED_PROTOCOL_SCHEMA",
    "MAX_SEQUENCE_TOKENS",
    "PROTOCOL_TO_DATA_SCHEMA",
    "SEMANTIC_CD_AUC_CEILING",
    "SURFACE_AUC_CEILING",
    "load_json",
    "sha256_file",
    "validate_data_release",
]
