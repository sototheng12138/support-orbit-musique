"""Deterministic construction of MuSiQue C/D/M support orbits.

The production entry point accepts one exact, checksum-pinned MuSiQue training
file.  It first validates all 39,876 rows as 19,938 answerable/unanswerable
pairs, constructs leakage components, assigns whole components to splits, and
then creates D by replacing exactly two non-support C slots with same-split
donors.  Official C support paragraphs and the C target are never rewritten.
"""

from __future__ import annotations

import bisect
import collections
import copy
import dataclasses
import hashlib
import itertools
import json
import math
import re
import statistics
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .audit import (
    SURFACE_FEATURE_NAMES,
    grouped_oof_auc,
    jaccard,
    normalize_text,
    relative_feature_distance,
    semantic_features,
    surface_features,
    token_set,
    tokenizer_length_audit,
)
from .formatting import (
    FIXED_UNANSWERABLE_TARGET,
    Paragraph,
    render_user_content,
    supported_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_PATH = Path(
    "/home/hesong/AI-Agent-Projects/data/musique_official/train_only/"
    "musique_full_v1.0_train.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "prepared_data_v2"
TOKENIZER_PATH = Path("/home/hesong/AI-Agent-Projects/models/Qwen3-4B-Instruct-2507")

EXPECTED_SOURCE_SHA256 = "b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034"
EXPECTED_SOURCE_ROWS = 39_876
EXPECTED_SOURCE_PAIRS = 19_938
SCHEMA_VERSION = "support-orbit-musique/v2"
BUILDER_VERSION = "2.0.0"
DEFAULT_SEED = "support-orbit-musique-v1"
SOURCE_REPOSITORY = "StonyBrookNLP/musique"
SOURCE_COMMIT = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
SOURCE_LICENSE = "CC-BY-4.0"
FIXED_REFUSAL = FIXED_UNANSWERABLE_TARGET


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _contains_token_sequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    target = tuple(needle)
    return any(tuple(haystack[start : start + width]) == target for start in range(len(haystack) - width + 1))


def _raw_text_hash(value: str) -> str:
    return _sha256_text(value)


def _paragraph_content(paragraph: Mapping[str, Any]) -> str:
    return f"{paragraph['title']} {paragraph['paragraph_text']}"


def validate_source_path(source: str | Path) -> Path:
    """Accept only the checksum-pinned, separately extracted official train member."""

    path = Path(source).expanduser().resolve()
    expected = EXPECTED_SOURCE_PATH.resolve()
    if path != expected:
        raise ValueError(f"source must be the frozen train-only path: {expected}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclasses.dataclass(frozen=True)
class BuildConfig:
    train_orbits: int = 1_920
    dev_orbits: int = 400
    shadow_orbits: int = 400
    seed: str = DEFAULT_SEED
    donor_length_ratio: float = 1.25
    max_sequence_tokens: int = 6_144
    surface_auc_ceiling: float = 0.60
    semantic_cd_auc_ceiling: float = 0.60
    minimum_train_orbits: int = 640
    minimum_dev_orbits: int = 160
    minimum_shadow_orbits: int = 160

    @property
    def total_orbits(self) -> int:
        return self.train_orbits + self.dev_orbits + self.shadow_orbits

    def validate(self) -> None:
        if min(self.train_orbits, self.dev_orbits, self.shadow_orbits) <= 0:
            raise ValueError("every split must contain at least one orbit")
        if self.train_orbits < self.minimum_train_orbits:
            raise ValueError("train_orbits is below the frozen minimum")
        if self.dev_orbits < self.minimum_dev_orbits:
            raise ValueError("dev_orbits is below the frozen minimum")
        if self.shadow_orbits < self.minimum_shadow_orbits:
            raise ValueError("shadow_orbits is below the frozen minimum")
        if self.donor_length_ratio < 1.0:
            raise ValueError("donor_length_ratio must be at least one")
        if self.max_sequence_tokens <= 0:
            raise ValueError("max_sequence_tokens must be positive")


@dataclasses.dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    manifest: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class RowHeader:
    pair_id: str
    line_number: int
    byte_offset: int
    byte_length: int
    row_sha256: str
    answerable: bool
    question: str
    answer: str
    aliases: tuple[str, ...]
    decomposition_core: tuple[tuple[int, str, str], ...]
    decomposition_support_indices: tuple[int | None, ...]
    paragraph_count: int
    paragraph_indices_valid: bool
    paragraph_text_hashes: frozenset[str]
    support_text_hashes: frozenset[str]
    support_indices: tuple[int, ...]
    c_support_decomposition_valid: bool
    subanswer_presence: tuple[bool, ...]

    @property
    def decomposition_ids(self) -> tuple[int, ...]:
        return tuple(item[0] for item in self.decomposition_core)

    @property
    def subanswers(self) -> tuple[str, ...]:
        return tuple(item[2] for item in self.decomposition_core)


@dataclasses.dataclass(frozen=True)
class Orbit:
    pair_id: str
    complete: RowHeader
    missing: RowHeader

    @property
    def orbit_id(self) -> str:
        return "musique_" + _sha256_text(self.pair_id)[:20]

    @property
    def aliases(self) -> tuple[str, ...]:
        values = dict.fromkeys((*self.complete.aliases, *self.missing.aliases))
        return tuple(values)

    @property
    def leakage_keys(self) -> frozenset[str]:
        keys: set[str] = set()
        keys.update(f"source_id:{value}" for value in self.complete.decomposition_ids)
        keys.update(f"support_text:{value}" for value in self.complete.support_text_hashes)
        answer_values = (
            self.complete.answer,
            *self.complete.aliases,
            *self.missing.aliases,
            *self.complete.subanswers,
        )
        keys.update(
            f"answer:{normalized}"
            for value in answer_values
            if (normalized := normalize_text(value))
        )
        return frozenset(keys)


@dataclasses.dataclass(frozen=True)
class Component:
    component_id: str
    orbits: tuple[Orbit, ...]
    leakage_keys: frozenset[str]

    @property
    def size(self) -> int:
        return len(self.orbits)


@dataclasses.dataclass(frozen=True)
class LoadedOrbit:
    orbit: Orbit
    complete: Mapping[str, Any]
    missing: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class Donor:
    split: str
    source_orbit_id: str
    source_pair_id_sha256: str
    source_slot: int
    paragraph: Mapping[str, Any]
    text_hash: str
    normalized_content: str
    normalized_title: str
    normalized_text: str
    content_tokens: frozenset[str]
    title_tokens: frozenset[str]
    text_tokens: frozenset[str]
    token_count: int
    stable_key: str


@dataclasses.dataclass(frozen=True)
class DonorProposal:
    target_slot: int
    donor: Donor
    local_cost: float
    diagnostics: Mapping[str, float]


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.sizes = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        left, right = self.find(first), self.find(second)
        if left == right:
            return
        if self.sizes[left] < self.sizes[right]:
            left, right = right, left
        self.parent[right] = left
        self.sizes[left] += self.sizes[right]


def _validate_payload_schema(payload: Mapping[str, Any], line_number: int) -> None:
    required = {
        "id",
        "paragraphs",
        "question",
        "question_decomposition",
        "answer",
        "answer_aliases",
        "answerable",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"line {line_number}: missing fields {sorted(missing)}")
    if not isinstance(payload["id"], str) or not payload["id"]:
        raise ValueError(f"line {line_number}: id must be a nonempty string")
    for field in ("question", "answer"):
        if not isinstance(payload[field], str):
            raise ValueError(f"line {line_number}: {field} must be a string")
    if not isinstance(payload["answerable"], bool):
        raise ValueError(f"line {line_number}: answerable must be bool")
    if not isinstance(payload["answer_aliases"], list) or not all(
        isinstance(value, str) for value in payload["answer_aliases"]
    ):
        raise ValueError(f"line {line_number}: answer_aliases must be list[str]")
    if not isinstance(payload["paragraphs"], list):
        raise ValueError(f"line {line_number}: paragraphs must be a list")
    for paragraph in payload["paragraphs"]:
        if not isinstance(paragraph, dict):
            raise ValueError(f"line {line_number}: paragraph must be an object")
        if not isinstance(paragraph.get("idx"), int):
            raise ValueError(f"line {line_number}: paragraph idx must be int")
        if not isinstance(paragraph.get("title"), str):
            raise ValueError(f"line {line_number}: paragraph title must be str")
        if not isinstance(paragraph.get("paragraph_text"), str):
            raise ValueError(f"line {line_number}: paragraph_text must be str")
        if not isinstance(paragraph.get("is_supporting"), bool):
            raise ValueError(f"line {line_number}: is_supporting must be bool")
    if not isinstance(payload["question_decomposition"], list) or not payload[
        "question_decomposition"
    ]:
        raise ValueError(f"line {line_number}: decomposition must be a nonempty list")
    for step in payload["question_decomposition"]:
        if not isinstance(step, dict):
            raise ValueError(f"line {line_number}: decomposition step must be an object")
        if not isinstance(step.get("id"), int):
            raise ValueError(f"line {line_number}: decomposition id must be int")
        if not isinstance(step.get("question"), str) or not isinstance(step.get("answer"), str):
            raise ValueError(f"line {line_number}: decomposition text fields must be str")
        support_index = step.get("paragraph_support_idx")
        if support_index is not None and not isinstance(support_index, int):
            raise ValueError(f"line {line_number}: support index must be int or null")


def _header_from_payload(
    payload: Mapping[str, Any],
    *,
    line_number: int,
    byte_offset: int,
    raw_line: bytes,
) -> RowHeader:
    _validate_payload_schema(payload, line_number)
    paragraphs = payload["paragraphs"]
    decomposition = payload["question_decomposition"]
    indices = [int(paragraph["idx"]) for paragraph in paragraphs]
    index_map = {int(paragraph["idx"]): paragraph for paragraph in paragraphs}
    indices_valid = len(indices) == len(set(indices)) and set(indices) == set(range(20))
    support_indices = tuple(
        sorted(int(paragraph["idx"]) for paragraph in paragraphs if paragraph["is_supporting"])
    )
    decomposition_support_indices = tuple(
        step["paragraph_support_idx"] for step in decomposition
    )
    c_support_valid = (
        len(paragraphs) == 20
        and indices_valid
        and all(
            isinstance(index, int)
            and index in index_map
            and index_map[index]["is_supporting"] is True
            for index in decomposition_support_indices
        )
        and set(decomposition_support_indices) == set(support_indices)
    )
    context_tokens: list[str] = []
    for paragraph in paragraphs:
        context_tokens.extend(_tokens(_paragraph_content(paragraph)))
    subanswer_presence = tuple(
        _contains_token_sequence(context_tokens, _tokens(str(step["answer"])))
        for step in decomposition
    )
    return RowHeader(
        pair_id=str(payload["id"]),
        line_number=line_number,
        byte_offset=byte_offset,
        byte_length=len(raw_line),
        row_sha256=_sha256_bytes(raw_line),
        answerable=bool(payload["answerable"]),
        question=str(payload["question"]),
        answer=str(payload["answer"]),
        aliases=tuple(str(value) for value in payload["answer_aliases"]),
        decomposition_core=tuple(
            (int(step["id"]), str(step["question"]), str(step["answer"]))
            for step in decomposition
        ),
        decomposition_support_indices=decomposition_support_indices,
        paragraph_count=len(paragraphs),
        paragraph_indices_valid=indices_valid,
        paragraph_text_hashes=frozenset(
            _raw_text_hash(str(paragraph["paragraph_text"])) for paragraph in paragraphs
        ),
        support_text_hashes=frozenset(
            _raw_text_hash(str(paragraph["paragraph_text"]))
            for paragraph in paragraphs
            if paragraph["is_supporting"]
        ),
        support_indices=support_indices,
        c_support_decomposition_valid=c_support_valid,
        subanswer_presence=subanswer_presence,
    )


def _scan_source(source: Path) -> tuple[list[Orbit], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[RowHeader]] = collections.defaultdict(list)
    source_digest = hashlib.sha256()
    byte_offset = 0
    with source.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            source_digest.update(raw_line)
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            header = _header_from_payload(
                payload,
                line_number=line_number,
                byte_offset=byte_offset,
                raw_line=raw_line,
            )
            groups[header.pair_id].append(header)
            byte_offset += len(raw_line)
    row_count = sum(len(values) for values in groups.values())
    digest = source_digest.hexdigest()
    if row_count != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(f"source rows {row_count}, expected {EXPECTED_SOURCE_ROWS}")
    if len(groups) != EXPECTED_SOURCE_PAIRS:
        raise RuntimeError(f"source pairs {len(groups)}, expected {EXPECTED_SOURCE_PAIRS}")
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source SHA256 {digest}, expected {EXPECTED_SOURCE_SHA256}")

    eligible: list[Orbit] = []
    ledger: list[dict[str, Any]] = []
    reason_counts: collections.Counter[str] = collections.Counter()
    aliases_differ = 0
    missing_support_histogram: collections.Counter[int] = collections.Counter()
    missing_subanswer_histogram: collections.Counter[int] = collections.Counter()
    for pair_id, rows in sorted(groups.items()):
        if len(rows) != 2:
            raise RuntimeError(f"pair {pair_id!r} has {len(rows)} rows, expected two")
        complete_rows = [row for row in rows if row.answerable]
        missing_rows = [row for row in rows if not row.answerable]
        if len(complete_rows) != 1 or len(missing_rows) != 1:
            raise RuntimeError(f"pair {pair_id!r} must have one answerable and one unanswerable row")
        complete, missing = complete_rows[0], missing_rows[0]
        if (
            complete.question != missing.question
            or complete.answer != missing.answer
            or complete.decomposition_core != missing.decomposition_core
        ):
            raise RuntimeError(f"pair {pair_id!r} does not share question/answer/decomposition")
        aliases_differ += complete.aliases != missing.aliases
        reason = "eligible"
        if complete.paragraph_count != 20 or missing.paragraph_count != 20:
            reason = "not_exactly_20_paragraphs"
        elif not complete.paragraph_indices_valid or not missing.paragraph_indices_valid:
            reason = "paragraph_slots_not_exact_0_to_19"
        elif not complete.c_support_decomposition_valid:
            reason = "complete_support_decomposition_invalid"
        else:
            missing_support = len(complete.support_text_hashes - missing.paragraph_text_hashes)
            missing_subanswers = sum(not value for value in missing.subanswer_presence)
            missing_support_histogram[missing_support] += 1
            missing_subanswer_histogram[missing_subanswers] += 1
            if missing_support < 1:
                reason = "missing_world_retains_every_complete_support_text"
            elif missing_subanswers < 1:
                reason = "missing_world_contains_every_subanswer_token_sequence"
            elif 20 - len(complete.support_indices) < 2:
                reason = "fewer_than_two_complete_nonsupport_slots"
        reason_counts[reason] += 1
        if reason == "eligible":
            eligible.append(Orbit(pair_id=pair_id, complete=complete, missing=missing))
        else:
            ledger.append(
                {
                    "pair_id_sha256": _sha256_text(pair_id),
                    "source_line_numbers": sorted(
                        [complete.line_number, missing.line_number]
                    ),
                    "reason": reason,
                }
            )
    audit = {
        "source_rows": row_count,
        "source_pairs": len(groups),
        "source_sha256": digest,
        "pair_validation": {
            "exactly_two_rows_per_id": True,
            "one_answerable_and_one_unanswerable": True,
            "same_question_answer_decomposition": True,
            "alias_lists_allowed_to_differ": True,
            "pairs_with_different_alias_lists": aliases_differ,
        },
        "eligibility_definition": {
            "paragraphs": "C and M each have exactly 20 unique slots numbered 0..19",
            "complete_support": (
                "every C decomposition support index points to a supporting paragraph, "
                "and the referenced-index set equals the C supporting-flag set"
            ),
            "missing_support": "M omits at least one raw-text SHA256 from C support",
            "missing_subanswer": (
                "NFKC -> casefold -> Unicode regex \\w+ tokens; at least one decomposition "
                "answer token sequence is absent from the concatenated M title+paragraph stream"
            ),
            "raw_substring_rule_rejected": (
                "A prior diagnostic count of 18,800 used raw substring matching and falsely "
                "treated forms such as America/American as matches; it is not used."
            ),
        },
        "eligibility_reason_counts": dict(sorted(reason_counts.items())),
        "strict_eligible_orbits": len(eligible),
        "missing_complete_support_count_histogram": {
            str(key): value for key, value in sorted(missing_support_histogram.items())
        },
        "missing_subanswer_count_histogram": {
            str(key): value for key, value in sorted(missing_subanswer_histogram.items())
        },
    }
    if len(eligible) != 18_834:
        raise RuntimeError(f"strict eligible count {len(eligible)}, expected 18,834")
    return eligible, ledger, audit


def _components(orbits: Sequence[Orbit]) -> tuple[list[Component], dict[str, int]]:
    dsu = _DisjointSet(len(orbits))
    owner: dict[str, int] = {}
    for index, orbit in enumerate(orbits):
        for key in orbit.leakage_keys:
            previous = owner.get(key)
            if previous is None:
                owner[key] = index
            else:
                dsu.union(index, previous)
    members: dict[int, list[Orbit]] = collections.defaultdict(list)
    for index, orbit in enumerate(orbits):
        members[dsu.find(index)].append(orbit)
    output: list[Component] = []
    for values in members.values():
        ordered = tuple(sorted(values, key=lambda value: value.orbit_id))
        component_id = "component_" + _sha256_text("\n".join(v.orbit_id for v in ordered))[:20]
        keys = frozenset(key for orbit in ordered for key in orbit.leakage_keys)
        output.append(Component(component_id=component_id, orbits=ordered, leakage_keys=keys))
    output.sort(key=lambda value: value.component_id)
    histogram = collections.Counter(component.size for component in output)
    return output, {str(size): count for size, count in sorted(histogram.items())}


def _choose_subset(
    pool: Sequence[Component], target: int, seed: str, split: str
) -> tuple[Component, ...]:
    order = sorted(
        range(len(pool)),
        key=lambda index: (
            _sha256_text(f"{seed}\n{split}\n{pool[index].component_id}"),
            pool[index].component_id,
        ),
    )
    previous: list[tuple[int, int] | None] = [None] * (target + 1)
    previous[0] = (-1, -1)
    for index in order:
        weight = pool[index].size
        if weight > target:
            continue
        for total in range(target - weight, -1, -1):
            if previous[total] is not None and previous[total + weight] is None:
                previous[total + weight] = (total, index)
        if previous[target] is not None:
            break
    if previous[target] is None:
        raise RuntimeError(f"cannot assign whole components to exact {split} size {target}")
    selected: list[Component] = []
    total = target
    while total:
        predecessor = previous[total]
        assert predecessor is not None
        prior_total, index = predecessor
        selected.append(pool[index])
        total = prior_total
    return tuple(selected)


def _assign_splits(
    components: Sequence[Component], config: BuildConfig
) -> tuple[dict[str, list[Orbit]], dict[str, tuple[Component, ...]], set[str]]:
    pool = list(components)
    assignments: dict[str, tuple[Component, ...]] = {}
    for split, target in (
        ("train", config.train_orbits),
        ("dev", config.dev_orbits),
        ("shadow", config.shadow_orbits),
    ):
        selected = _choose_subset(pool, target, config.seed, split)
        assignments[split] = selected
        selected_ids = {component.component_id for component in selected}
        pool = [component for component in pool if component.component_id not in selected_ids]
    split_orbits: dict[str, list[Orbit]] = {}
    for split, selected in assignments.items():
        values = [orbit for component in selected for orbit in component.orbits]
        split_orbits[split] = sorted(
            values,
            key=lambda orbit: (
                _sha256_text(f"{config.seed}\n{split}\n{orbit.orbit_id}"),
                orbit.orbit_id,
            ),
        )
    keys = {
        split: frozenset(key for orbit in values for key in orbit.leakage_keys)
        for split, values in split_orbits.items()
    }
    for first, second in itertools.combinations(keys, 2):
        overlap = keys[first] & keys[second]
        if overlap:
            raise RuntimeError(f"cross-split leakage keys between {first}/{second}: {len(overlap)}")
    selected_ids = {
        orbit.orbit_id for values in split_orbits.values() for orbit in values
    }
    return split_orbits, assignments, selected_ids


def _load_selected_rows(source: Path, orbits: Iterable[Orbit]) -> dict[str, LoadedOrbit]:
    headers: list[RowHeader] = []
    by_orbit: dict[str, Orbit] = {}
    for orbit in orbits:
        by_orbit[orbit.orbit_id] = orbit
        headers.extend((orbit.complete, orbit.missing))
    payloads: dict[tuple[str, bool], Mapping[str, Any]] = {}
    with source.open("rb") as handle:
        for header in sorted(headers, key=lambda value: value.byte_offset):
            handle.seek(header.byte_offset)
            raw_line = handle.read(header.byte_length)
            if _sha256_bytes(raw_line) != header.row_sha256:
                raise RuntimeError(f"source row changed at line {header.line_number}")
            payload = json.loads(raw_line)
            payloads[(header.pair_id, header.answerable)] = payload
    return {
        orbit_id: LoadedOrbit(
            orbit=orbit,
            complete=payloads[(orbit.pair_id, True)],
            missing=payloads[(orbit.pair_id, False)],
        )
        for orbit_id, orbit in by_orbit.items()
    }


def _ordered_paragraphs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(paragraph)
        for paragraph in sorted(payload["paragraphs"], key=lambda value: value["idx"])
    ]


class _DonorIndex:
    def __init__(self, donors: Iterable[Donor]) -> None:
        self.donors = sorted(donors, key=lambda value: (value.token_count, value.stable_key))
        self.lengths = [donor.token_count for donor in self.donors]
        postings: dict[str, list[int]] = collections.defaultdict(list)
        for position, donor in enumerate(self.donors):
            for token in donor.content_tokens:
                postings[token].append(position)
        self.postings = dict(postings)

    def candidate_positions(
        self,
        *,
        token_count: int,
        ratio: float,
        question_tokens: frozenset[str],
        salt: str,
    ) -> list[int]:
        minimum = max(1, math.ceil(token_count / ratio))
        maximum = max(minimum, math.floor(token_count * ratio))
        left = bisect.bisect_left(self.lengths, minimum)
        right = bisect.bisect_right(self.lengths, maximum)
        width = right - left
        if width <= 0:
            return []
        positions: set[int] = set()
        center = bisect.bisect_left(self.lengths, token_count, left, right)
        positions.update(range(max(left, center - 4), min(right, center + 4)))
        digest = int(_sha256_text(salt), 16)
        start = digest % width
        stride = (digest // max(1, width)) % width or 1
        while math.gcd(stride, width) != 1:
            stride += 1
            if stride >= width:
                stride = 1
                break
        for step in range(min(56, width)):
            positions.add(left + ((start + step * stride) % width))
        # The fixed-size sample is intentionally outcome-blind.  Question
        # overlap is used only to rank these candidates, never to scan the full
        # donor pool or select examples by state label.
        return sorted(positions)


def _build_donor_index(
    split: str,
    values: Sequence[LoadedOrbit],
    *,
    selected_support_hashes: frozenset[str],
) -> _DonorIndex:
    donors: list[Donor] = []
    for loaded in values:
        support = set(loaded.orbit.complete.support_indices)
        for paragraph in _ordered_paragraphs(loaded.complete):
            slot = int(paragraph["idx"])
            if slot in support or paragraph["is_supporting"]:
                continue
            content = _paragraph_content(paragraph)
            tokens = _tokens(content)
            normalized_title = normalize_text(str(paragraph["title"]))
            normalized_text = normalize_text(str(paragraph["paragraph_text"]))
            text_hash = _raw_text_hash(str(paragraph["paragraph_text"]))
            # A paragraph used as C support anywhere in the selected release is
            # never eligible as a donor, even when it is non-support in this
            # particular source orbit.
            if text_hash in selected_support_hashes:
                continue
            donors.append(
                Donor(
                    split=split,
                    source_orbit_id=loaded.orbit.orbit_id,
                    source_pair_id_sha256=_sha256_text(loaded.orbit.pair_id),
                    source_slot=slot,
                    paragraph=paragraph,
                    text_hash=text_hash,
                    normalized_content=normalize_text(content),
                    normalized_title=normalized_title,
                    normalized_text=normalized_text,
                    content_tokens=frozenset(tokens),
                    title_tokens=frozenset(normalized_title.split()),
                    text_tokens=frozenset(normalized_text.split()),
                    token_count=max(1, len(tokens)),
                    stable_key=_sha256_text(
                        f"{loaded.orbit.orbit_id}\n{slot}\n{text_hash}"
                    ),
                )
            )
    return _DonorIndex(donors)


def _local_profile(
    paragraph: Mapping[str, Any],
    other: Sequence[Mapping[str, Any]],
    question_tokens: frozenset[str],
) -> dict[str, float]:
    title = normalize_text(str(paragraph["title"]))
    text = normalize_text(str(paragraph["paragraph_text"]))
    title_tokens = token_set(title)
    text_tokens = token_set(text)
    content_tokens = token_set(f"{title} {text}")
    other_titles = [normalize_text(str(value["title"])) for value in other]
    other_texts = [normalize_text(str(value["paragraph_text"])) for value in other]
    return {
        "title_duplicate_rate": sum(title == value for value in other_titles) / max(1, len(other)),
        "text_duplicate_rate": sum(text == value for value in other_texts) / max(1, len(other)),
        "question_document_jaccard": jaccard(question_tokens, content_tokens),
        "question_title_jaccard": jaccard(question_tokens, title_tokens),
        "title_coherence": statistics.fmean(
            [jaccard(title_tokens, token_set(str(value["title"]))) for value in other]
        )
        if other
        else 0.0,
        "text_coherence": statistics.fmean(
            [jaccard(text_tokens, token_set(str(value["paragraph_text"]))) for value in other]
        )
        if other
        else 0.0,
        "token_count": float(max(1, len(_tokens(_paragraph_content(paragraph))))),
    }


def _donor_as_target_paragraph(donor: Donor, target_slot: int) -> dict[str, Any]:
    return {
        "idx": target_slot,
        "title": str(donor.paragraph["title"]),
        "paragraph_text": str(donor.paragraph["paragraph_text"]),
        "is_supporting": False,
    }


def _forbidden_phrases(loaded: LoadedOrbit) -> tuple[tuple[str, ...], ...]:
    values = (
        str(loaded.complete["answer"]),
        *[str(value) for value in loaded.complete["answer_aliases"]],
        *[str(value) for value in loaded.missing["answer_aliases"]],
        *[str(step["answer"]) for step in loaded.complete["question_decomposition"]],
    )
    return tuple(dict.fromkeys(_tokens(value) for value in values if _tokens(value)))


def _donor_has_leakage(donor: Donor, phrases: Sequence[Sequence[str]]) -> bool:
    content = tuple(donor.normalized_content.split())
    return any(_contains_token_sequence(content, phrase) for phrase in phrases)


def _rank_slot_proposals(
    *,
    loaded: LoadedOrbit,
    split: str,
    paragraphs: Sequence[Mapping[str, Any]],
    slot: int,
    donor_index: _DonorIndex,
    forbidden: Sequence[Sequence[str]],
    support_hash_split: Mapping[str, frozenset[str]],
    config: BuildConfig,
    limit: int,
) -> list[DonorProposal]:
    old = paragraphs[slot]
    other = [value for index, value in enumerate(paragraphs) if index != slot]
    question_tokens = token_set(str(loaded.complete["question"]))
    old_profile = _local_profile(old, other, question_tokens)
    old_tokens = max(1, len(_tokens(_paragraph_content(old))))
    old_content_tokens = token_set(_paragraph_content(old))
    other_titles = [normalize_text(str(value["title"])) for value in other]
    other_texts = [normalize_text(str(value["paragraph_text"])) for value in other]
    other_title_tokens = [frozenset(value.split()) for value in other_titles]
    other_text_tokens = [frozenset(value.split()) for value in other_texts]
    positions = donor_index.candidate_positions(
        token_count=old_tokens,
        ratio=config.donor_length_ratio,
        question_tokens=question_tokens,
        salt=f"{config.seed}\n{loaded.orbit.orbit_id}\n{slot}",
    )
    cheap_candidates: list[
        tuple[float, Donor, float, float, float, float, float, float]
    ] = []
    for position in positions:
        donor = donor_index.donors[position]
        if donor.split != split or donor.source_orbit_id == loaded.orbit.orbit_id:
            continue
        if donor.text_hash == _raw_text_hash(str(old["paragraph_text"])):
            continue
        owners = support_hash_split.get(donor.text_hash, frozenset())
        if owners:
            continue
        ratio = max(old_tokens, donor.token_count) / min(old_tokens, donor.token_count)
        if ratio > config.donor_length_ratio + 1e-12:
            continue
        if _donor_has_leakage(donor, forbidden):
            continue
        title_duplicate_rate = sum(
            donor.normalized_title == value for value in other_titles
        ) / max(1, len(other_titles))
        text_duplicate_rate = sum(
            donor.normalized_text == value for value in other_texts
        ) / max(1, len(other_texts))
        duplicate_delta = abs(
            old_profile["title_duplicate_rate"] - title_duplicate_rate
        ) + abs(old_profile["text_duplicate_rate"] - text_duplicate_rate)
        candidate_question_jaccard = jaccard(question_tokens, donor.content_tokens)
        q_delta = abs(
            old_profile["question_document_jaccard"]
            - candidate_question_jaccard
        )
        candidate_title_question = jaccard(question_tokens, donor.title_tokens)
        title_q_delta = abs(
            old_profile["question_title_jaccard"] - candidate_title_question
        )
        length_delta = abs(math.log(donor.token_count / old_tokens))
        source_similarity = jaccard(old_content_tokens, donor.content_tokens)
        cheap_cost = (
            50.0 * duplicate_delta
            + 10.0 * q_delta
            + 6.0 * title_q_delta
            + 2.0 * length_delta
            - 0.5 * source_similarity
        )
        cheap_candidates.append(
            (
                cheap_cost,
                donor,
                duplicate_delta,
                q_delta,
                title_q_delta,
                ratio,
                length_delta,
                source_similarity,
            )
        )
    cheap_candidates.sort(
        key=lambda value: (
            value[0],
            _sha256_text(
                f"{config.seed}\n{loaded.orbit.orbit_id}\n{slot}\n{value[1].stable_key}"
            ),
        )
    )
    proposals: list[DonorProposal] = []
    # Coherence is the expensive term.  It is evaluated only after a frozen
    # cheap shortlist, without changing any hard donor eligibility condition.
    for (
        cheap_cost,
        donor,
        duplicate_delta,
        q_delta,
        title_q_delta,
        ratio,
        length_delta,
        source_similarity,
    ) in cheap_candidates[:16]:
        title_coherence = statistics.fmean(
            jaccard(donor.title_tokens, value) for value in other_title_tokens
        )
        text_coherence = statistics.fmean(
            jaccard(donor.text_tokens, value) for value in other_text_tokens
        )
        coherence_delta = abs(
            old_profile["title_coherence"] - title_coherence
        ) + abs(old_profile["text_coherence"] - text_coherence)
        cost = cheap_cost + 3.0 * coherence_delta
        proposals.append(
            DonorProposal(
                target_slot=slot,
                donor=donor,
                local_cost=cost,
                diagnostics={
                    "duplicate_delta": duplicate_delta,
                    "question_overlap_delta": q_delta,
                    "title_question_overlap_delta": title_q_delta,
                    "coherence_delta": coherence_delta,
                    "length_ratio": ratio,
                    "source_token_jaccard": source_similarity,
                },
            )
        )
    proposals.sort(
        key=lambda value: (
            value.local_cost,
            _sha256_text(
                f"{config.seed}\n{loaded.orbit.orbit_id}\n{slot}\n{value.donor.stable_key}"
            ),
        )
    )
    return proposals[:limit]


def _construct_distractor(
    *,
    loaded: LoadedOrbit,
    split: str,
    donor_index: _DonorIndex,
    support_hash_split: Mapping[str, frozenset[str]],
    config: BuildConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paragraphs = _ordered_paragraphs(loaded.complete)
    support = set(loaded.orbit.complete.support_indices)
    nonsupport = [slot for slot in range(20) if slot not in support]
    question_tokens = token_set(str(loaded.complete["question"]))
    median_tokens = statistics.median(
        max(1, len(_tokens(_paragraph_content(paragraph)))) for paragraph in paragraphs
    )

    def slot_rank(slot: int) -> tuple[float, ...] | tuple[float, ..., str]:
        old = paragraphs[slot]
        other = [value for index, value in enumerate(paragraphs) if index != slot]
        profile = _local_profile(old, other, question_tokens)
        duplicates = profile["title_duplicate_rate"] + profile["text_duplicate_rate"]
        question_overlap = profile["question_document_jaccard"]
        coherence = profile["title_coherence"] + profile["text_coherence"]
        median_delta = abs(profile["token_count"] - median_tokens) / max(1.0, median_tokens)
        tie = _sha256_text(f"{config.seed}\n{loaded.orbit.orbit_id}\nslot\n{slot}")
        return (duplicates, question_overlap, coherence, median_delta, tie)

    ordered_slots = sorted(nonsupport, key=slot_rank)
    forbidden = _forbidden_phrases(loaded)
    proposal_groups: dict[int, list[DonorProposal]] = {}
    for slot in ordered_slots:
        values = _rank_slot_proposals(
            loaded=loaded,
            split=split,
            paragraphs=paragraphs,
            slot=slot,
            donor_index=donor_index,
            forbidden=forbidden,
            support_hash_split=support_hash_split,
            config=config,
            limit=8,
        )
        if values:
            proposal_groups[slot] = values
        if len(proposal_groups) >= 2:
            break
    proposals = [value for values in proposal_groups.values() for value in values]
    pairs: list[tuple[float, DonorProposal, DonorProposal]] = []
    for first, second in itertools.combinations(proposals, 2):
        if first.target_slot == second.target_slot:
            continue
        if first.donor.stable_key == second.donor.stable_key:
            continue
        pairs.append((first.local_cost + second.local_cost, first, second))
    if not pairs:
        raise RuntimeError(f"no two valid donor replacements for {loaded.orbit.orbit_id}")
    pairs.sort(
        key=lambda value: (
            value[0],
            _sha256_text(
                f"{config.seed}\n{loaded.orbit.orbit_id}\n"
                f"{value[1].target_slot}:{value[1].donor.stable_key}\n"
                f"{value[2].target_slot}:{value[2].donor.stable_key}"
            ),
        )
    )
    c_surface = surface_features(paragraphs)
    c_semantic = semantic_features(str(loaded.complete["question"]), paragraphs)
    finalists: list[tuple[float, list[dict[str, Any]], tuple[DonorProposal, DonorProposal]]] = []
    for local_sum, first, second in pairs[:8]:
        distractor = copy.deepcopy(paragraphs)
        distractor[first.target_slot] = _donor_as_target_paragraph(first.donor, first.target_slot)
        distractor[second.target_slot] = _donor_as_target_paragraph(second.donor, second.target_slot)
        d_surface = surface_features(distractor)
        d_semantic = semantic_features(str(loaded.complete["question"]), distractor)
        global_surface = relative_feature_distance(c_surface, d_surface)
        global_question = relative_feature_distance(
            c_semantic[len(SURFACE_FEATURE_NAMES) :],
            d_semantic[len(SURFACE_FEATURE_NAMES) :],
        )
        score = global_surface + global_question + 0.01 * local_sum
        finalists.append((score, distractor, (first, second)))
    finalists.sort(
        key=lambda value: (
            value[0],
            _sha256_text(
                f"{config.seed}\n{loaded.orbit.orbit_id}\n"
                f"{value[2][0].donor.stable_key}\n{value[2][1].donor.stable_key}"
            ),
        )
    )
    score, distractor, chosen = finalists[0]
    donor_metadata = []
    for proposal in sorted(chosen, key=lambda value: value.target_slot):
        donor_metadata.append(
            {
                "target_slot": proposal.target_slot,
                "source_split": proposal.donor.split,
                "source_orbit_id": proposal.donor.source_orbit_id,
                "source_pair_id_sha256": proposal.donor.source_pair_id_sha256,
                "source_slot": proposal.donor.source_slot,
                "source_text_sha256": proposal.donor.text_hash,
                "local_cost": proposal.local_cost,
                "matching_diagnostics": dict(proposal.diagnostics),
                "joint_cost": score,
            }
        )
    for slot in support:
        if distractor[slot] != paragraphs[slot]:
            raise AssertionError("support paragraph changed during D construction")
    changed = [slot for slot in range(20) if distractor[slot] != paragraphs[slot]]
    if changed != [value["target_slot"] for value in donor_metadata]:
        raise AssertionError("D must change exactly the two recorded non-support slots")
    return distractor, donor_metadata


def _user_prompt(question: str, paragraphs: Sequence[Mapping[str, Any]]) -> str:
    rendered = [f"Question: {question}", "", "Paragraphs:"]
    for paragraph in paragraphs:
        rendered.append(
            f"[{paragraph['idx']}] Title: {paragraph['title']}\n{paragraph['paragraph_text']}"
        )
    return "\n".join(rendered)


def _record(
    *,
    loaded: LoadedOrbit,
    split: str,
    state: str,
    paragraphs: Sequence[Mapping[str, Any]],
    answerable: bool,
    target: str,
    support_indices: Sequence[int],
    decomposition: Sequence[Mapping[str, Any]],
    donor_metadata: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    canonical_paragraphs = [
        Paragraph(
            index=int(paragraph["idx"]),
            title=str(paragraph["title"]),
            text=str(paragraph["paragraph_text"]),
        )
        for paragraph in paragraphs
    ]
    prompt = render_user_content(str(loaded.complete["question"]), canonical_paragraphs)
    canonical_target = (
        supported_target(support_indices, str(loaded.complete["answer"]))
        if state in {"C", "D"}
        else FIXED_UNANSWERABLE_TARGET
    )
    if state == "D" and target != str(loaded.complete["answer"]):
        raise AssertionError("D answer target drifted from C")
    # Evaluation aliases remain state-specific official fields.  In
    # particular, M-only aliases must never enlarge the C/D answer gold set.
    aliases = list(
        loaded.complete["answer_aliases"]
        if state in {"C", "D"}
        else loaded.missing["answer_aliases"]
    )
    common = {
        "schema_version": SCHEMA_VERSION,
        "id": f"{loaded.orbit.orbit_id}::{state}",
        "orbit_id": loaded.orbit.orbit_id,
        "source_id": loaded.orbit.pair_id,
        "source_id_sha256": _sha256_text(loaded.orbit.pair_id),
        "split": split,
        "state": state,
        "question": str(loaded.complete["question"]),
        "answer": str(loaded.complete["answer"]),
        "answer_aliases": aliases,
        "gold_support_idxs": list(support_indices),
        "answerable": answerable,
        "paragraphs": list(paragraphs),
        "question_decomposition": list(decomposition),
        "target": canonical_target,
        "prompt": prompt,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": canonical_target},
        ],
        "sealed": split == "shadow",
        "training_read_allowed": split != "shadow",
        "source_dataset": "MuSiQue full v1.0 train",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_license": SOURCE_LICENSE,
    }
    if state == "D":
        common.update(
            {
                "machine_generated": True,
                "generation_boundary": (
                    "paragraph-only deterministic replacement of exactly two C non-support "
                    "slots; every C support paragraph and the C target are byte-preserved"
                ),
                "donors": list(donor_metadata),
            }
        )
    else:
        header = loaded.orbit.complete if state == "C" else loaded.orbit.missing
        common.update(
            {
                "machine_generated": False,
                "source_line_number": header.line_number,
                "source_row_sha256": header.row_sha256,
            }
        )
    return common


def _make_records(
    split_orbits: Mapping[str, Sequence[Orbit]],
    loaded: Mapping[str, LoadedOrbit],
    config: BuildConfig,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    loaded_by_split = {
        split: [loaded[orbit.orbit_id] for orbit in orbits]
        for split, orbits in split_orbits.items()
    }
    support_hash_split_mutable: dict[str, set[str]] = collections.defaultdict(set)
    for split, values in split_orbits.items():
        for orbit in values:
            for text_hash in orbit.complete.support_text_hashes:
                support_hash_split_mutable[text_hash].add(split)
    support_hash_split = {
        key: frozenset(values) for key, values in support_hash_split_mutable.items()
    }
    selected_support_hashes = frozenset(support_hash_split)
    output: dict[str, list[dict[str, Any]]] = {}
    invariant_counts: collections.Counter[str] = collections.Counter()
    length_ratios: list[float] = []
    joint_costs: list[float] = []
    alias_policy_by_split: dict[str, dict[str, int]] = {}
    for split in ("train", "dev", "shadow"):
        different_alias_lists = 0
        m_only_alias_pairs = 0
        m_only_alias_values = 0
        c_only_alias_pairs = 0
        c_only_alias_values = 0
        for value in loaded_by_split[split]:
            complete_aliases = {
                normalize_text(str(alias))
                for alias in value.complete["answer_aliases"]
                if normalize_text(str(alias))
            }
            missing_aliases = {
                normalize_text(str(alias))
                for alias in value.missing["answer_aliases"]
                if normalize_text(str(alias))
            }
            different_alias_lists += value.complete["answer_aliases"] != value.missing[
                "answer_aliases"
            ]
            m_only = missing_aliases - complete_aliases
            c_only = complete_aliases - missing_aliases
            m_only_alias_pairs += bool(m_only)
            m_only_alias_values += len(m_only)
            c_only_alias_pairs += bool(c_only)
            c_only_alias_values += len(c_only)
        alias_policy_by_split[split] = {
            "orbits_with_different_raw_alias_lists": different_alias_lists,
            "orbits_with_M_only_normalized_aliases": m_only_alias_pairs,
            "M_only_normalized_alias_values": m_only_alias_values,
            "orbits_with_C_only_normalized_aliases": c_only_alias_pairs,
            "C_only_normalized_alias_values": c_only_alias_values,
        }
        donor_index = _build_donor_index(
            split,
            loaded_by_split[split],
            selected_support_hashes=selected_support_hashes,
        )
        records: list[dict[str, Any]] = []
        for loaded_orbit in loaded_by_split[split]:
            c_paragraphs = _ordered_paragraphs(loaded_orbit.complete)
            m_paragraphs = _ordered_paragraphs(loaded_orbit.missing)
            d_paragraphs, donor_metadata = _construct_distractor(
                loaded=loaded_orbit,
                split=split,
                donor_index=donor_index,
                support_hash_split=support_hash_split,
                config=config,
            )
            c_target = str(loaded_orbit.complete["answer"])
            c_support = loaded_orbit.orbit.complete.support_indices
            m_support = loaded_orbit.orbit.missing.support_indices
            records.extend(
                (
                    _record(
                        loaded=loaded_orbit,
                        split=split,
                        state="C",
                        paragraphs=c_paragraphs,
                        answerable=True,
                        target=c_target,
                        support_indices=c_support,
                        decomposition=loaded_orbit.complete["question_decomposition"],
                    ),
                    _record(
                        loaded=loaded_orbit,
                        split=split,
                        state="D",
                        paragraphs=d_paragraphs,
                        answerable=True,
                        target=c_target,
                        support_indices=c_support,
                        decomposition=loaded_orbit.complete["question_decomposition"],
                        donor_metadata=donor_metadata,
                    ),
                    _record(
                        loaded=loaded_orbit,
                        split=split,
                        state="M",
                        paragraphs=m_paragraphs,
                        answerable=False,
                        target=FIXED_REFUSAL,
                        support_indices=m_support,
                        decomposition=loaded_orbit.missing["question_decomposition"],
                    ),
                )
            )
            changed = [
                slot for slot in range(20) if c_paragraphs[slot] != d_paragraphs[slot]
            ]
            if len(changed) != 2 or any(slot in c_support for slot in changed):
                raise AssertionError("D changed a support slot or did not change exactly two slots")
            invariant_counts["orbits_with_exactly_two_nonsupport_replacements"] += 1
            invariant_counts["orbits_with_byte_identical_support_in_C_and_D"] += all(
                c_paragraphs[slot] == d_paragraphs[slot] for slot in c_support
            )
            invariant_counts["orbits_with_exact_C_and_D_target"] += 1
            invariant_counts["orbits_with_same_split_donors"] += all(
                donor["source_split"] == split for donor in donor_metadata
            )
            invariant_counts["orbits_with_no_C_support_text_as_donor"] += all(
                donor["source_text_sha256"] not in selected_support_hashes
                for donor in donor_metadata
            )
            invariant_counts["orbits_with_state_specific_official_aliases"] += (
                records[-3]["answer_aliases"]
                == list(loaded_orbit.complete["answer_aliases"])
                and records[-2]["answer_aliases"]
                == list(loaded_orbit.complete["answer_aliases"])
                and records[-1]["answer_aliases"]
                == list(loaded_orbit.missing["answer_aliases"])
            )
            length_ratios.extend(
                float(donor["matching_diagnostics"]["length_ratio"])
                for donor in donor_metadata
            )
            joint_costs.extend(float(donor["joint_cost"]) for donor in donor_metadata[:1])
        output[split] = records
    total_orbits = sum(len(values) for values in split_orbits.values())
    expected = {
        "orbits_with_exactly_two_nonsupport_replacements": total_orbits,
        "orbits_with_byte_identical_support_in_C_and_D": total_orbits,
        "orbits_with_exact_C_and_D_target": total_orbits,
        "orbits_with_same_split_donors": total_orbits,
        "orbits_with_no_C_support_text_as_donor": total_orbits,
        "orbits_with_state_specific_official_aliases": total_orbits,
    }
    if dict(invariant_counts) != expected:
        raise AssertionError(f"construction invariant mismatch: {invariant_counts} != {expected}")
    return output, {
        "invariant_counts": dict(sorted(invariant_counts.items())),
        "all_invariants_pass": True,
        "donor_length_ratio": {
            "ceiling": config.donor_length_ratio,
            "max": max(length_ratios, default=0.0),
            "mean": statistics.fmean(length_ratios) if length_ratios else 0.0,
            "violations": sum(value > config.donor_length_ratio + 1e-12 for value in length_ratios),
        },
        "donor_joint_cost": {
            "mean": statistics.fmean(joint_costs) if joint_costs else 0.0,
            "max": max(joint_costs, default=0.0),
        },
        "answer_alias_policy": {
            "C_and_D": "exact complete-row answer_aliases only",
            "M": "exact missing-row answer_aliases only",
            "leakage_components": "union of complete and missing normalized aliases",
            "counts_by_split": alias_policy_by_split,
        },
        "donor_matching_cost_formula": (
            "local=50*duplicate_delta + 10*question_doc_jaccard_delta + "
            "6*question_title_jaccard_delta + 3*coherence_delta + "
            "2*abs(log(token_length_ratio)) - 0.5*old_donor_token_jaccard; "
            "joint=relative_surface_distance + relative_question_overlap_distance + "
            "0.01*sum(local)"
        ),
    }


def _json_bytes(payload: Any, *, pretty: bool) -> bytes:
    if pretty:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    else:
        text = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
    return text.encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.write_bytes(payload)
    return {"sha256": _sha256_bytes(payload), "bytes": len(payload)}


def _write_json(path: Path, payload: Any) -> dict[str, Any]:
    return _write_bytes(path, _json_bytes(payload, pretty=True))


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    with path.open("wb") as handle:
        for record in records:
            raw = _json_bytes(record, pretty=False)
            handle.write(raw)
            digest.update(raw)
            byte_count += len(raw)
            record_count += 1
    return {"sha256": digest.hexdigest(), "bytes": byte_count, "records": record_count}


def _clean_output_files(destination: Path) -> None:
    """Remove only builder-owned files from a prior deterministic run."""

    names = (
        "train.jsonl",
        "dev.jsonl",
        "shadow.sealed.jsonl",
        "train.jsonl.sha256",
        "dev.jsonl.sha256",
        "shadow.sealed.jsonl.sha256",
        "rejection_ledger.json",
        "audit.json",
        "SHADOW_SEALED.json",
        "manifest.json",
        "SHA256SUMS",
    )
    for name in names:
        path = destination / name
        if path.is_file():
            path.unlink()


def build_dataset(
    source: str | Path = EXPECTED_SOURCE_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT,
    config: BuildConfig | None = None,
) -> BuildResult:
    """Build deterministic train/dev/sealed-shadow artifacts on CPU only."""

    config = config or BuildConfig()
    config.validate()
    source_path = validate_source_path(source)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _clean_output_files(destination)

    eligible, rejection_entries, source_audit = _scan_source(source_path)
    components, component_histogram = _components(eligible)
    split_orbits, split_components, selected_ids = _assign_splits(components, config)
    component_by_orbit = {
        orbit.orbit_id: component.component_id
        for component in components
        for orbit in component.orbits
    }
    for orbit in eligible:
        if orbit.orbit_id not in selected_ids:
            rejection_entries.append(
                {
                    "pair_id_sha256": _sha256_text(orbit.pair_id),
                    "orbit_id": orbit.orbit_id,
                    "component_id": component_by_orbit[orbit.orbit_id],
                    "reason": "eligible_component_reserve_not_selected",
                }
            )

    loaded = _load_selected_rows(
        source_path, (orbit for values in split_orbits.values() for orbit in values)
    )
    records_by_split, construction_audit = _make_records(split_orbits, loaded, config)
    all_records = [
        record
        for split in ("train", "dev", "shadow")
        for record in records_by_split[split]
    ]
    surface_probe = grouped_oof_auc(
        all_records, semantic=False, seed=f"{config.seed}\nsurface-probe"
    )
    semantic_probe = grouped_oof_auc(
        all_records, semantic=True, seed=f"{config.seed}\nsemantic-probe"
    )
    tokenizer_audit = tokenizer_length_audit(
        all_records, TOKENIZER_PATH, config.max_sequence_tokens
    )
    surface_pass = all(
        value["separability_auc"] <= config.surface_auc_ceiling
        for value in surface_probe["pairs"].values()
    )
    semantic_cd_pass = (
        semantic_probe["pairs"]["C-D"]["separability_auc"]
        <= config.semantic_cd_auc_ceiling
    )
    release_ready = (
        construction_audit["all_invariants_pass"]
        and surface_pass
        and semantic_cd_pass
        and tokenizer_audit["zero_truncation_pass"]
    )

    split_key_sets = {
        split: frozenset(key for orbit in values for key in orbit.leakage_keys)
        for split, values in split_orbits.items()
    }
    cross_split_overlaps = {
        f"{first}-{second}": len(split_key_sets[first] & split_key_sets[second])
        for first, second in itertools.combinations(("train", "dev", "shadow"), 2)
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "source_validation": source_audit,
        "component_isolation": {
            "component_key_families": [
                "decomposition source IDs",
                "raw C support paragraph-text SHA256",
                "NFKC/casefold/token-normalized answers, both alias lists, and subanswers",
            ],
            "components": len(components),
            "component_size_histogram": component_histogram,
            "split_component_counts": {
                split: len(values) for split, values in split_components.items()
            },
            "cross_split_key_overlap_counts": cross_split_overlaps,
            "whole_components_only": True,
        },
        "construction": construction_audit,
        "separability_probes": {
            "policy": {
                "surface_only": (
                    f"C-D, D-M, and C-M separability AUC must each be <= "
                    f"{config.surface_auc_ceiling:.2f}"
                ),
                "semantic": (
                    f"C-D separability AUC must be <= {config.semantic_cd_auc_ceiling:.2f}; "
                    "D-M and C-M are diagnostic because missing support is a real semantic signal"
                ),
                "separability_auc": "max(raw ROC AUC, 1 - raw ROC AUC)",
                "shadow": "sealed shadow records excluded from every learned probe",
            },
            "surface_only": surface_probe,
            "semantic_question_overlap": semantic_probe,
            "surface_gate_pass": surface_pass,
            "semantic_C_D_gate_pass": semantic_cd_pass,
        },
        "tokenizer_length": tokenizer_audit,
        "release_gate": {
            "status": "READY" if release_ready else "HOLD",
            "construction_invariants_pass": construction_audit["all_invariants_pass"],
            "surface_probe_pass": surface_pass,
            "semantic_C_D_probe_pass": semantic_cd_pass,
            "zero_truncation_pass": tokenizer_audit["zero_truncation_pass"],
        },
    }

    ledger_counts = collections.Counter(entry["reason"] for entry in rejection_entries)
    rejection_ledger = {
        "schema_version": SCHEMA_VERSION,
        "source_pairs": EXPECTED_SOURCE_PAIRS,
        "strict_eligible_orbits": len(eligible),
        "selected_orbits": len(selected_ids),
        "reason_counts": dict(sorted(ledger_counts.items())),
        "entries": sorted(
            rejection_entries,
            key=lambda value: (
                str(value["reason"]),
                str(value.get("pair_id_sha256", "")),
            ),
        ),
    }

    artifact_meta: dict[str, dict[str, Any]] = {}
    filenames = {
        "train": "train.jsonl",
        "dev": "dev.jsonl",
        "shadow": "shadow.sealed.jsonl",
    }
    # The shadow body is streamed once here.  All subsequent metadata uses the
    # digest returned by this writer and never reopens the sealed artifact.
    for split in ("train", "dev", "shadow"):
        name = filenames[split]
        artifact_meta[name] = _write_jsonl(destination / name, records_by_split[split])
        sidecar_name = f"{name}.sha256"
        sidecar = f"{artifact_meta[name]['sha256']}  {name}\n".encode("utf-8")
        artifact_meta[sidecar_name] = _write_bytes(destination / sidecar_name, sidecar)

    artifact_meta["rejection_ledger.json"] = _write_json(
        destination / "rejection_ledger.json", rejection_ledger
    )
    artifact_meta["audit.json"] = _write_json(destination / "audit.json", audit)
    shadow_marker = {
        "schema_version": SCHEMA_VERSION,
        "sealed": True,
        "training_read_allowed": False,
        "artifact": "shadow.sealed.jsonl",
        "sha256": artifact_meta["shadow.sealed.jsonl"]["sha256"],
        "bytes": artifact_meta["shadow.sealed.jsonl"]["bytes"],
        "orbit_count": config.shadow_orbits,
        "record_count": artifact_meta["shadow.sealed.jsonl"]["records"],
        "policy": (
            "Generated once; body must not be opened for training, model selection, "
            "threshold tuning, prompt tuning, or learned data diagnostics."
        ),
    }
    artifact_meta["SHADOW_SEALED.json"] = _write_json(
        destination / "SHADOW_SEALED.json", shadow_marker
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "selection_seed": config.seed,
        "source": {
            "path": str(source_path),
            "sha256": EXPECTED_SOURCE_SHA256,
            "rows": EXPECTED_SOURCE_ROWS,
            "pairs": EXPECTED_SOURCE_PAIRS,
            "dataset": "MuSiQue full v1.0 train",
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "license": SOURCE_LICENSE,
        },
        "input_policy": {
            "allowed": str(EXPECTED_SOURCE_PATH.resolve()),
            "forbidden": ["archive body", "official dev", "official test", "network"],
            "source_path_and_sha_enforced": True,
        },
        "construction": {
            "C": "official answerable row; official answer target",
            "M": "official paired unanswerable row; fixed exact refusal target",
            "D": (
                "copy C and replace exactly two non-support slots with leak-filtered, "
                "length/coherence/overlap-matched C-nonsupport donors from the same split; "
                "copy C target unchanged"
            ),
            "states_per_orbit": 3,
            "paragraph_slots_per_state": 20,
            "fixed_refusal": FIXED_REFUSAL,
            "model_generated_labels": False,
        },
        "leakage_policy": audit["component_isolation"],
        "counts": {
            "strict_eligible_orbits": len(eligible),
            "selected_orbits": len(selected_ids),
            "train_orbits": config.train_orbits,
            "dev_orbits": config.dev_orbits,
            "shadow_orbits": config.shadow_orbits,
            "train_records": len(records_by_split["train"]),
            "dev_records": len(records_by_split["dev"]),
            "shadow_records": len(records_by_split["shadow"]),
            "minimum_gate": {
                "train_orbits": config.minimum_train_orbits,
                "dev_orbits": config.minimum_dev_orbits,
                "shadow_orbits": config.minimum_shadow_orbits,
            },
            "minimum_gate_exceeded": True,
        },
        "record_contract": {
            "unique_id": "orbit_id::state",
            "state_values": ["C", "D", "M"],
            "required_eval_keys": [
                "id",
                "orbit_id",
                "source_id",
                "state",
                "answer",
                "answer_aliases",
                "gold_support_idxs",
                "answerable",
            ],
        },
        "shadow_policy": shadow_marker,
        "tokenizer_audit": tokenizer_audit,
        "release_gate": audit["release_gate"],
        "artifact_sha256": {
            name: metadata["sha256"] for name, metadata in sorted(artifact_meta.items())
        },
    }
    artifact_meta["manifest.json"] = _write_json(destination / "manifest.json", manifest)
    checksum_lines = "".join(
        f"{metadata['sha256']}  {name}\n"
        for name, metadata in sorted(artifact_meta.items())
    )
    _write_bytes(destination / "SHA256SUMS", checksum_lines.encode("utf-8"))
    return BuildResult(output_dir=destination, manifest=manifest)
