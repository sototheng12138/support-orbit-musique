from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from support_orbit_musique.data import (
    EXPECTED_SOURCE_PATH,
    BuildConfig,
    LoadedOrbit,
    Orbit,
    _build_donor_index,
    _components,
    _construct_distractor,
    _header_from_payload,
    _ordered_paragraphs,
    _paragraph_content,
    _raw_text_hash,
    _record,
    _tokens,
    validate_source_path,
)


def _paragraphs(prefix: str, support: tuple[int, ...]) -> list[dict[str, object]]:
    return [
        {
            "idx": index,
            "title": f"{prefix} title {index}",
            "paragraph_text": (
                f"{prefix} paragraph {index} has stable words about topic number {index} "
                "and enough material for deterministic length matching"
            ),
            "is_supporting": index in support,
        }
        for index in range(20)
    ]


def _payload(
    pair_id: str,
    *,
    answerable: bool,
    prefix: str,
    aliases: tuple[str, ...] = (),
) -> dict[str, object]:
    support = (0, 1) if answerable else (0,)
    paragraphs = _paragraphs(prefix, support)
    if answerable:
        paragraphs[0]["paragraph_text"] += " intermediate"
        paragraphs[1]["paragraph_text"] += " finalanswer"
    else:
        paragraphs[0]["paragraph_text"] += " intermediate"
    return {
        "id": pair_id,
        "paragraphs": paragraphs,
        "question": f"Question for {pair_id}?",
        "question_decomposition": [
            {
                "id": 1000 + int(pair_id[-1]),
                "question": "first hop",
                "answer": "intermediate",
                "paragraph_support_idx": 0,
            },
            {
                "id": 2000 + int(pair_id[-1]),
                "question": "second hop",
                "answer": "finalanswer",
                "paragraph_support_idx": 1 if answerable else None,
            },
        ],
        "answer": "finalanswer",
        "answer_aliases": list(aliases),
        "answerable": answerable,
    }


def _header(payload: dict[str, object], line: int):
    raw = (json.dumps(payload) + "\n").encode()
    return _header_from_payload(
        payload, line_number=line, byte_offset=line * 1000, raw_line=raw
    )


def _loaded(pair_id: str, prefix: str, aliases: tuple[str, ...] = ()) -> LoadedOrbit:
    complete = _payload(pair_id, answerable=True, prefix=f"{prefix} complete", aliases=aliases)
    missing = _payload(pair_id, answerable=False, prefix=f"{prefix} missing")
    orbit = Orbit(pair_id, _header(complete, 1), _header(missing, 2))
    return LoadedOrbit(orbit=orbit, complete=complete, missing=missing)


def test_path_guard_accepts_only_frozen_train_member(tmp_path: Path) -> None:
    assert validate_source_path(EXPECTED_SOURCE_PATH) == EXPECTED_SOURCE_PATH.resolve()
    impostor = tmp_path / EXPECTED_SOURCE_PATH.name
    impostor.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_source_path(impostor)


def test_pair_semantics_allow_alias_lists_to_differ() -> None:
    complete = _payload("pair1", answerable=True, prefix="c", aliases=("alias",))
    missing = _payload("pair1", answerable=False, prefix="m", aliases=())
    c_header, m_header = _header(complete, 1), _header(missing, 2)
    assert c_header.question == m_header.question
    assert c_header.answer == m_header.answer
    assert c_header.decomposition_core == m_header.decomposition_core
    assert c_header.aliases != m_header.aliases
    assert c_header.c_support_decomposition_valid
    assert not all(m_header.subanswer_presence)


def test_component_graph_uses_ids_support_and_normalized_answers() -> None:
    first = _loaded("pair1", "alpha", aliases=("shared alias",)).orbit
    second = _loaded("pair2", "beta", aliases=("Shared   Alias",)).orbit
    raw_third = _loaded("pair3", "gamma", aliases=("unrelated",)).orbit
    third_core = ((3003, "third first", "uniquehop"), (4003, "third second", "uniqueanswer"))
    third = Orbit(
        raw_third.pair_id,
        dataclasses.replace(
            raw_third.complete,
            answer="uniqueanswer",
            aliases=("unrelated",),
            decomposition_core=third_core,
        ),
        dataclasses.replace(
            raw_third.missing,
            answer="uniqueanswer",
            aliases=(),
            decomposition_core=third_core,
        ),
    )
    components, histogram = _components([first, second, third])
    assert sorted(component.size for component in components) == [1, 2]
    assert histogram == {"1": 1, "2": 1}


def test_d_replaces_exactly_two_nonsupport_slots_with_same_split_donors() -> None:
    target = _loaded("pair1", "target")
    donor_a = _loaded("pair2", "donora")
    donor_b = _loaded("pair3", "donorb")
    values = [target, donor_a, donor_b]
    selected_support_hashes = frozenset(
        text_hash
        for value in values
        for text_hash in value.orbit.complete.support_text_hashes
    )
    donor_index = _build_donor_index(
        "train", values, selected_support_hashes=selected_support_hashes
    )
    support_hash_split = {
        text_hash: frozenset({"train"}) for text_hash in selected_support_hashes
    }
    distractor, metadata = _construct_distractor(
        loaded=target,
        split="train",
        donor_index=donor_index,
        support_hash_split=support_hash_split,
        config=BuildConfig(),
    )
    complete = _ordered_paragraphs(target.complete)
    changed = [slot for slot in range(20) if complete[slot] != distractor[slot]]
    assert len(changed) == 2
    assert not set(changed) & set(target.orbit.complete.support_indices)
    assert all(
        complete[slot] == distractor[slot]
        for slot in target.orbit.complete.support_indices
    )
    assert [item["target_slot"] for item in metadata] == sorted(changed)
    assert all(item["source_split"] == "train" for item in metadata)
    loaded_by_orbit = {value.orbit.orbit_id: value for value in values}
    for item in metadata:
        target_paragraph = complete[item["target_slot"]]
        donor_loaded = loaded_by_orbit[item["source_orbit_id"]]
        donor_paragraph = _ordered_paragraphs(donor_loaded.complete)[item["source_slot"]]
        target_tokens = max(1, len(_tokens(_paragraph_content(target_paragraph))))
        donor_tokens = max(1, len(_tokens(_paragraph_content(donor_paragraph))))
        actual_ratio = max(target_tokens, donor_tokens) / min(target_tokens, donor_tokens)
        recorded_ratio = item["matching_diagnostics"]["length_ratio"]
        assert recorded_ratio == pytest.approx(actual_ratio, abs=0.0, rel=0.0)
        assert recorded_ratio <= 1.25
        assert item["source_text_sha256"] not in selected_support_hashes


def test_donor_index_excludes_text_used_as_support_in_another_orbit() -> None:
    target = _loaded("pair1", "target")
    donor = _loaded("pair2", "donor")
    support_owner = _loaded("pair3", "support-owner")
    shared_text = str(support_owner.complete["paragraphs"][0]["paragraph_text"])
    donor.complete["paragraphs"][2]["paragraph_text"] = shared_text
    selected_support_hashes = frozenset(
        text_hash
        for value in (target, donor, support_owner)
        for text_hash in value.orbit.complete.support_text_hashes
    )
    donor_index = _build_donor_index(
        "train",
        [target, donor, support_owner],
        selected_support_hashes=selected_support_hashes,
    )
    shared_hash = _raw_text_hash(shared_text)
    assert shared_hash in selected_support_hashes
    assert all(value.text_hash != shared_hash for value in donor_index.donors)


def test_eval_contract_and_targets_are_exact() -> None:
    loaded = _loaded("pair1", "target", aliases=("Final Answer",))
    paragraphs = _ordered_paragraphs(loaded.complete)
    record = _record(
        loaded=loaded,
        split="dev",
        state="C",
        paragraphs=paragraphs,
        answerable=True,
        target="finalanswer",
        support_indices=(0, 1),
        decomposition=loaded.complete["question_decomposition"],
    )
    for field in (
        "id",
        "orbit_id",
        "source_id",
        "state",
        "answer",
        "answer_aliases",
        "gold_support_idxs",
        "answerable",
    ):
        assert field in record
    assert record["id"] == f"{record['orbit_id']}::C"
    assert record["target"] == "S | evidence=[P00,P01] | answer=finalanswer"
    assert record["answer_aliases"] == ["Final Answer"]
    assert record["messages"][-1]["content"] == record["target"]

    distractor = _record(
        loaded=loaded,
        split="dev",
        state="D",
        paragraphs=paragraphs,
        answerable=True,
        target="finalanswer",
        support_indices=(0, 1),
        decomposition=loaded.complete["question_decomposition"],
        donor_metadata=(),
    )
    missing = _record(
        loaded=loaded,
        split="dev",
        state="M",
        paragraphs=_ordered_paragraphs(loaded.missing),
        answerable=False,
        target="U | evidence=[] | answer=INSUFFICIENT_EVIDENCE",
        support_indices=(0,),
        decomposition=loaded.missing["question_decomposition"],
    )
    assert distractor["answer_aliases"] == ["Final Answer"]
    assert missing["answer_aliases"] == []


def test_config_exceeds_frozen_minimum_gate() -> None:
    BuildConfig().validate()
    with pytest.raises(ValueError):
        BuildConfig(train_orbits=639).validate()
