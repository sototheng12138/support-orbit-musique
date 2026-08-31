from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from support_orbit_musique.protocol import (
    EXPECTED_COUNTS,
    EXPECTED_PROTOCOL_SCHEMA,
    sha256_file,
    validate_data_release,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "protocols" / "support_orbit_pilot_v1.json"
PROTOCOL_MD = PROJECT_ROOT / "protocols" / "support_orbit_pilot_v1.md"
PROTOCOL_SIDECAR = PROJECT_ROOT / "protocols" / "support_orbit_pilot_v1.json.sha256"
PROTOCOL_V2_PATH = PROJECT_ROOT / "protocols" / "support_orbit_pilot_v2.json"
PROTOCOL_V2_MD = PROJECT_ROOT / "protocols" / "support_orbit_pilot_v2.md"
PROTOCOL_V2_SIDECAR = PROJECT_ROOT / "protocols" / "support_orbit_pilot_v2.json.sha256"
PREPARED_V2 = PROJECT_ROOT / "prepared_data_v2"


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_protocol_identity_and_stop_default_are_explicit() -> None:
    protocol = _protocol()
    assert protocol["schema_version"] == EXPECTED_PROTOCOL_SCHEMA
    assert protocol["protocol_id"] == "support_orbit_musique_pilot_v1"
    assert protocol["protocol_status"] in {
        "DRAFT_AWAITING_DATA_RELEASE",
        "FROZEN_PRE_GPU",
    }
    assert protocol["launch_status"] == "STOP_BEFORE_GPU"
    assert protocol["status_semantics"]["default_on_missing_or_mismatch"] == "STOP_BEFORE_GPU"
    lock = protocol["prepared_data_lock"]
    if protocol["protocol_status"] == "DRAFT_AWAITING_DATA_RELEASE":
        assert lock["expected_manifest_sha256"] is None
        assert lock["release_status"] == "HOLD"
    else:
        assert len(lock["expected_manifest_sha256"]) == 64
        assert lock["release_status"] == "READY"


def test_frozen_protocol_sidecar_binds_exact_bytes() -> None:
    expected = f"{sha256_file(PROTOCOL_PATH)}  {PROTOCOL_PATH.name}\n"
    assert PROTOCOL_SIDECAR.read_text(encoding="utf-8") == expected


def test_v2_identity_supersession_and_sidecar_are_exact() -> None:
    protocol = json.loads(PROTOCOL_V2_PATH.read_text(encoding="utf-8"))
    assert protocol["schema_version"] == "support-orbit-musique.protocol/v2"
    assert protocol["protocol_id"] == "support_orbit_musique_pilot_v2"
    assert protocol["protocol_status"] == "FROZEN_PRE_GPU"
    assert protocol["launch_status"] == "STOP_BEFORE_GPU"
    assert protocol["supersession"]["withdrawn_protocol_sha256"] == sha256_file(PROTOCOL_PATH)
    assert protocol["supersession"]["status"] == (
        "WITHDRAWN_INVALID_AUDIT_RATIO_AND_DONOR_ALIAS_CONTAMINATION"
    )
    expected = f"{sha256_file(PROTOCOL_V2_PATH)}  {PROTOCOL_V2_PATH.name}\n"
    assert PROTOCOL_V2_SIDECAR.read_text(encoding="utf-8") == expected


def test_v2_release_validates_without_opening_shadow_body() -> None:
    if not (PREPARED_V2 / "train.jsonl").exists() or not (PREPARED_V2 / "dev.jsonl").exists():
        pytest.skip("public release omits MuSiQue-derived train/dev JSONL bodies")
    protocol = json.loads(PROTOCOL_V2_PATH.read_text(encoding="utf-8"))
    manifest_path = PREPARED_V2 / "manifest.json"
    audit_path = PREPARED_V2 / "audit.json"
    marker_path = PREPARED_V2 / "SHADOW_SEALED.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    report = validate_data_release(
        protocol=protocol,
        manifest_path=manifest_path,
        audit_path=audit_path,
        shadow_marker_path=marker_path,
        verified={
            "manifest": manifest,
            "audit": audit,
            "shadow_marker": marker,
            "sha256": {
                "manifest.json": sha256_file(manifest_path),
                "audit.json": sha256_file(audit_path),
                "SHADOW_SEALED.json": sha256_file(marker_path),
                # Bodies are not opened before the launch/training boundary;
                # their writer digests are transitively frozen in the manifest.
                "train.jsonl": manifest["artifact_sha256"]["train.jsonl"],
                "dev.jsonl": manifest["artifact_sha256"]["dev.jsonl"],
            },
        },
    )
    assert report["passed"]
    assert report["shadow_body_opened"] is False
    for key in (
        "construction:no_C_support_text_as_donor",
        "construction:state_specific_official_aliases",
        "construction:C_D_alias_policy",
        "construction:M_alias_policy",
        "protocol_artifact_hash:shadow_writer_digest",
    ):
        assert report["checks"][key]


def test_v2_carries_v1_method_and_gates_without_change() -> None:
    first = _protocol()
    second = json.loads(PROTOCOL_V2_PATH.read_text(encoding="utf-8"))
    assert second["method_lock_unchanged_from_v1"]["inherited_from_sha256"] == sha256_file(
        PROTOCOL_PATH
    )
    assert second["method_lock_unchanged_from_v1"]["arms"]["CONTROL"] == "L_sft"
    assert (
        second["method_lock_unchanged_from_v1"]["arms"]["HopPAIR"]
        == first["arms"]["HopPAIR"]["objective"]
    )
    v1_gates = {
        gate["id"]: gate["rule"]
        for gate in first["development_evaluation"]["gates_from_support_orbit_metrics_v1"]
    }
    normalized_v1 = {
        "CD_MIN_F1_GAIN": "delta>=0.04 and paired bootstrap CI low>0",
        "ORBIT_ANSWER_SUFF_GAIN": "delta>=0.04 and paired bootstrap CI low>0",
        "D_ANSWER_F1_GAIN": "delta>=0.04",
        "C_ANSWER_F1_NONINFERIORITY": "delta>=-0.02",
        "FALSE_REFUSAL_NONINFERIORITY": "delta<=0.02",
        "M_REFUSAL": "delta>=0 and HopPAIR absolute rate>=0.80",
        "ORBIT_SUPPORT_SUFF_NONINFERIORITY": "delta>=-0.02",
        "PARSE_RATE": "HopPAIR absolute rate>=0.99",
        "RUN_INTEGRITY": "all bindings and runtime checks true",
    }
    assert set(v1_gates) == set(normalized_v1)
    assert second["development_evaluation_and_gates_unchanged_from_v1"]["gates"] == normalized_v1


def test_source_and_access_allowlist_are_exact_and_fail_closed() -> None:
    protocol = _protocol()
    source = protocol["source_lock"]
    assert source["repository_commit"] == "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
    assert source["archive_sha256"] == (
        "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd"
    )
    assert source["allowed_train_sha256"] == (
        "b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034"
    )
    assert source["license_sha256"] == (
        "cce5d01fa4a83b794271bd2c28cffdf99afd43c803e6ddefddae39b591ea7448"
    )
    boundary = protocol["access_boundary"]
    assert boundary["builder_allowlist"] == [source["allowed_train_path"]]
    assert boundary["training_allowlist_after_launch_receipt"] == ["prepared_data_v1/train.jsonl"]
    assert boundary["development_evaluation_allowlist_after_training"] == [
        "prepared_data_v1/dev.jsonl"
    ]
    assert boundary["forbidden_until_three_seed_dev_go"] == ["prepared_data_v1/shadow.sealed.jsonl"]
    forbidden = "\n".join(boundary["forbidden_for_entire_pilot"])
    assert "official MuSiQue dev" in forbidden
    assert "official MuSiQue test" in forbidden
    assert boundary["fail_closed"] is True


def test_orbit_split_and_data_gates_are_frozen() -> None:
    protocol = _protocol()
    orbit = protocol["orbit_definition"]
    assert orbit["states"] == ["C", "D", "M"]
    assert orbit["paragraph_slots_per_state"] == 20
    assert "replace exactly two" in orbit["D"]["intervention"]
    assert any(
        "all C support paragraphs byte-for-byte" in value for value in orbit["D"]["preserved"]
    )
    assert orbit["M"]["target"] == "U | evidence=[] | answer=INSUFFICIENT_EVIDENCE"
    split = protocol["split_contract"]
    assert split["counts"] == EXPECTED_COUNTS
    assert split["selected_orbits"] == 2_720
    assert split["whole_component_assignment"] is True
    assert split["required_cross_split_leakage_key_intersections"] == 0
    gates = protocol["data_release_gates"]
    assert gates["all_required"] is True
    assert "<= 0.60" in gates["learned_surface_probe"]["gate"]
    assert "<= 0.60" in gates["semantic_diagnostic_probe"]["gate"]
    assert gates["tokenizer"]["max_sequence_tokens"] == 6_144
    assert gates["learned_surface_probe"]["shadow_excluded"] is True


def test_arms_are_exposure_matched_and_objective_is_exact() -> None:
    protocol = _protocol()
    arms = protocol["arms"]
    assert arms["CONTROL"]["presentations_per_orbit"] == ["C", "D", "M"]
    assert arms["HopPAIR"]["presentations_per_orbit"] == ["C", "D", "M"]
    assert arms["CONTROL"]["objective"] == "L_sft"
    assert arms["HopPAIR"]["kl_weight"] == 0.1
    assert arms["HopPAIR"]["flip_weight"] == 0.2
    assert arms["HopPAIR"]["flip_margin"] == 2.0
    assert "stopgrad" in arms["HopPAIR"]["objective"]
    assert "one model forward" in arms["single_forward"]
    training = protocol["matched_training_contract"]
    assert training["train_orbits"] == 1_920
    assert training["group_microbatch_orbits"] == 1
    assert training["gradient_accumulation_orbits"] == 8
    assert training["effective_batch_orbits"] == 8
    assert training["optimizer_steps_per_arm"] == 240
    assert training["train_orbits"] // training["effective_batch_orbits"] == 240
    assert training["development_seeds_before_shadow"] == [17, 29, 43]
    assert "base-model hash" in training["initialization_audit"]["pre_gpu"]
    assert (
        "initial_trainable_parameters_sha256"
        in training["initialization_audit"]["pre_dev_generation"]
    )
    assert training["lora"]["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_eval_core_gates_and_three_seed_shadow_lock_are_exact() -> None:
    protocol = _protocol()
    evaluation = protocol["development_evaluation"]
    assert evaluation["decoding"]["max_new_tokens"] == 128
    assert "no output-dependent" in evaluation["decoding"]["budget_exhausted_rows"]
    gates = {item["id"]: item["rule"] for item in evaluation["gates_from_support_orbit_metrics_v1"]}
    assert set(gates) == {
        "CD_MIN_F1_GAIN",
        "ORBIT_ANSWER_SUFF_GAIN",
        "D_ANSWER_F1_GAIN",
        "C_ANSWER_F1_NONINFERIORITY",
        "FALSE_REFUSAL_NONINFERIORITY",
        "M_REFUSAL",
        "ORBIT_SUPPORT_SUFF_NONINFERIORITY",
        "PARSE_RATE",
        "RUN_INTEGRITY",
    }
    assert gates["CD_MIN_F1_GAIN"] == "delta >= +0.04 and paired-bootstrap CI lower bound > 0"
    assert gates["ORBIT_ANSWER_SUFF_GAIN"] == (
        "delta >= +0.04 and paired-bootstrap CI lower bound > 0"
    )
    assert gates["C_ANSWER_F1_NONINFERIORITY"] == "delta >= -0.02"
    assert gates["FALSE_REFUSAL_NONINFERIORITY"] == "delta <= +0.02"
    assert evaluation["all_gates_required_per_seed"] is True
    shadow = protocol["shadow_and_official_boundary"]
    assert shadow["shadow"]["initial_status"] == "SEALED"
    assert "3/3" in shadow["shadow"]["open_condition"]
    assert shadow["official_dev_and_test"]["status"] == "UNEXTRACTED_UNREAD_OUTSIDE_PILOT"
    assert "not authorized" in shadow["official_dev_and_test"]["open_condition"]


def test_literature_and_claim_boundaries_do_not_overclaim() -> None:
    protocol = _protocol()
    literature = protocol["literature_and_novelty_boundary"]
    assert set(literature) >= {"RAFT", "Trust-Align", "CORD", "GRACE"}
    assert "COLM 2024" in literature["RAFT"]["reference"]
    assert "ICLR 2025" in literature["Trust-Align"]["reference"]
    assert "NAACL 2025" in literature["CORD"]["reference"]
    assert "preprint" in literature["GRACE"]["reference"]
    forbidden = "\n".join(protocol["claim_boundary"]["forbidden_without_later_protocol"])
    assert "official MuSiQue dev/test" in forbidden
    assert "state-of-the-art" in forbidden
    assert "RL or preference-optimization" in forbidden


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _synthetic_release(tmp_path: Path) -> tuple[dict[str, object], Path, Path, Path]:
    protocol = deepcopy(_protocol())
    protocol["protocol_status"] = "FROZEN_PRE_GPU"
    protocol["prepared_data_lock"]["release_status"] = "READY"
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text("synthetic train\n", encoding="utf-8")
    dev.write_text("synthetic dev\n", encoding="utf-8")
    marker_path = tmp_path / "SHADOW_SEALED.json"
    marker = {
        "schema_version": "support-orbit-musique/v1",
        "sealed": True,
        "training_read_allowed": False,
        "orbit_count": 400,
        "record_count": 1_200,
        "sha256": "f" * 64,
    }
    _write_json(marker_path, marker)
    audit_path = tmp_path / "audit.json"
    release = {
        "status": "READY",
        "construction_invariants_pass": True,
        "surface_probe_pass": True,
        "semantic_C_D_probe_pass": True,
        "zero_truncation_pass": True,
    }
    audit = {
        "schema_version": "support-orbit-musique/v1",
        "release_gate": release,
        "construction": {
            "all_invariants_pass": True,
            "invariant_counts": {
                "orbits_with_exactly_two_nonsupport_replacements": 2_720,
                "orbits_with_byte_identical_support_in_C_and_D": 2_720,
                "orbits_with_exact_C_and_D_target": 2_720,
                "orbits_with_same_split_donors": 2_720,
            },
            "donor_length_ratio": {"ceiling": 1.25, "max": 1.25, "violations": 0},
        },
        "component_isolation": {
            "whole_components_only": True,
            "cross_split_key_overlap_counts": {
                "train-dev": 0,
                "train-shadow": 0,
                "dev-shadow": 0,
            },
        },
        "separability_probes": {
            "surface_only": {
                "pairs": {
                    name: {"separability_auc": auc}
                    for name, auc in (("C-D", 0.55), ("D-M", 0.57), ("C-M", 0.59))
                }
            },
            "semantic_question_overlap": {"pairs": {"C-D": {"separability_auc": 0.58}}},
        },
    }
    _write_json(audit_path, audit)
    source = {
        "path": protocol["source_lock"]["allowed_train_path"],
        "sha256": protocol["source_lock"]["allowed_train_sha256"],
        "rows": 39_876,
        "pairs": 19_938,
        "dataset": "MuSiQue full v1.0 train",
        "repository": "StonyBrookNLP/musique",
        "commit": protocol["source_lock"]["repository_commit"],
        "license": "CC-BY-4.0",
    }
    manifest = {
        "schema_version": "support-orbit-musique/v1",
        "source": source,
        "counts": {
            **EXPECTED_COUNTS,
            "selected_orbits": 2_720,
        },
        "release_gate": release,
        "tokenizer_audit": {
            "zero_truncation_pass": True,
            "prepared_contract_byte_identical": True,
            "prepared_vs_canonical_mismatches": 0,
            "over_limit_records": 0,
            "shadow_excluded": True,
            "records": 6_960,
            "max_tokens": 6_000,
        },
        "artifact_sha256": {
            "audit.json": sha256_file(audit_path),
            "SHADOW_SEALED.json": sha256_file(marker_path),
            "shadow.sealed.jsonl": "f" * 64,
            "train.jsonl": sha256_file(train),
            "dev.jsonl": sha256_file(dev),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    protocol["prepared_data_lock"]["expected_manifest_sha256"] = sha256_file(manifest_path)
    protocol["prepared_data_lock"]["artifact_sha256"] = {
        "train.jsonl": sha256_file(train),
        "dev.jsonl": sha256_file(dev),
        "audit.json": sha256_file(audit_path),
        "SHADOW_SEALED.json": sha256_file(marker_path),
        "shadow.sealed.jsonl_writer_digest": "f" * 64,
    }
    return protocol, manifest_path, audit_path, marker_path


def test_release_validator_passes_without_shadow_body(tmp_path: Path) -> None:
    protocol, manifest, audit, marker = _synthetic_release(tmp_path)
    assert not (tmp_path / "shadow.sealed.jsonl").exists()
    report = validate_data_release(
        protocol=protocol,
        manifest_path=manifest,
        audit_path=audit,
        shadow_marker_path=marker,
    )
    assert report["passed"]
    assert report["shadow_body_opened"] is False
    assert all(report["checks"].values())


def test_release_validator_fails_a_surface_or_token_gate(tmp_path: Path) -> None:
    protocol, manifest_path, audit_path, marker = _synthetic_release(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["separability_probes"]["surface_only"]["pairs"]["C-D"]["separability_auc"] = 0.61
    _write_json(audit_path, audit)
    manifest["artifact_sha256"]["audit.json"] = sha256_file(audit_path)
    manifest["tokenizer_audit"]["over_limit_records"] = 1
    _write_json(manifest_path, manifest)
    protocol["prepared_data_lock"]["expected_manifest_sha256"] = sha256_file(manifest_path)
    report = validate_data_release(
        protocol=protocol,
        manifest_path=manifest_path,
        audit_path=audit_path,
        shadow_marker_path=marker,
    )
    assert not report["passed"]
    assert not report["checks"]["surface_auc:C-D"]
    assert not report["checks"]["token_over_limit_zero"]


def test_markdown_carries_the_machine_contract() -> None:
    text = PROTOCOL_MD.read_text(encoding="utf-8")
    for token in (
        "STOP_BEFORE_GPU",
        "1,920/400/400",
        "CONTROL",
        "HopPAIR",
        "0.1 KL",
        "240 optimizer steps",
        "seeds 29 and",
        "official MuSiQue dev/test",
        "RAFT (COLM 2024)",
        "Trust-Align (ICLR 2025)",
        "CORD (NAACL 2025 Short Papers)",
        "GRACE (arXiv:2601.04525)",
    ):
        assert token in text
