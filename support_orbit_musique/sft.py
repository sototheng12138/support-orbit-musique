"""Strict completion-only encoding and orbit-major dynamic collation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from .formatting import (
    FIXED_UNANSWERABLE_TARGET,
    RenderedState,
    STATE_ORDER,
    State,
    render_prepared_record,
)


IGNORE_INDEX = -100
ACTION_TOKEN_IDS = {"S": 50, "U": 52}


@dataclass(frozen=True)
class EncodedState:
    orbit_id: str
    state: State
    assistant_content: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    prompt_tokens: int
    completion_ids: tuple[int, ...]
    prediction_positions: tuple[int, ...]


@dataclass(frozen=True)
class EncodedOrbit:
    """Exactly one C/D/M group; batching treats this as the atomic sample."""

    orbit_id: str
    complete: EncodedState
    distractor: EncodedState
    missing: EncodedState

    @property
    def states(self) -> tuple[EncodedState, EncodedState, EncodedState]:
        return (self.complete, self.distractor, self.missing)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("tokenizer must return a mapping containing input_ids")
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], Sequence):
        if len(ids) != 1:
            raise ValueError("tokenizer unexpectedly returned a batch")
        ids = ids[0]
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise TypeError("tokenizer input_ids must be a sequence")
    result = [int(token_id) for token_id in ids]
    if any(token_id < 0 for token_id in result):
        raise ValueError("tokenizer returned a negative token ID")
    return result


def validate_action_token_ids(tokenizer: Any) -> None:
    """Fail if the frozen Qwen S/U action-token assumption has drifted."""

    for action, expected in ACTION_TOKEN_IDS.items():
        actual = _token_ids(tokenizer, action)
        if actual != [expected]:
            raise ValueError(f"action {action!r} tokenized as {actual}, expected [{expected}]")


def encode_state(
    tokenizer: Any,
    rendered: RenderedState,
    *,
    max_length: int,
) -> EncodedState:
    """Mask every prompt token and supervise every native assistant token.

    Prompt and assistant suffix are tokenized separately and their concatenation
    must exactly equal tokenizing the full native conversation.  Overlength
    examples are rejected; this function never truncates.
    """

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    prompt_ids = _token_ids(tokenizer, rendered.prompt)
    completion_ids = _token_ids(tokenizer, rendered.completion)
    full_ids = _token_ids(tokenizer, rendered.full_text)
    if not prompt_ids or not completion_ids:
        raise ValueError(f"{rendered.orbit_id}/{rendered.state}: empty tokenization")
    if [*prompt_ids, *completion_ids] != full_ids:
        raise ValueError(
            f"{rendered.orbit_id}/{rendered.state}: separate/full token boundary drift"
        )
    if len(full_ids) > max_length:
        raise ValueError(
            f"{rendered.orbit_id}/{rendered.state}: {len(full_ids)} tokens exceed "
            f"max_length={max_length}; truncation is forbidden"
        )
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}
    if not eos_ids or eos_ids.isdisjoint(completion_ids):
        raise ValueError(f"{rendered.orbit_id}/{rendered.state}: EOS is not supervised")
    expected_action = ACTION_TOKEN_IDS["U" if rendered.state == "M" else "S"]
    if completion_ids[0] != expected_action:
        raise ValueError(
            f"{rendered.orbit_id}/{rendered.state}: first completion token is "
            f"{completion_ids[0]}, expected action token {expected_action}"
        )
    prompt_size = len(prompt_ids)
    prediction_positions = tuple(range(prompt_size - 1, len(full_ids) - 1))
    if len(prediction_positions) != len(completion_ids):
        raise AssertionError("causal prediction-position construction failed")
    return EncodedState(
        orbit_id=rendered.orbit_id,
        state=rendered.state,
        assistant_content=rendered.assistant_content,
        input_ids=tuple(full_ids),
        labels=tuple([IGNORE_INDEX] * prompt_size + completion_ids),
        attention_mask=tuple([1] * len(full_ids)),
        prompt_tokens=prompt_size,
        completion_ids=tuple(completion_ids),
        prediction_positions=prediction_positions,
    )


def encode_orbit(states: Sequence[EncodedState]) -> EncodedOrbit:
    """Validate and bind an exact C/D/M orbit in canonical order."""

    if len(states) != 3:
        raise ValueError("an orbit must contain exactly three states")
    if tuple(state.state for state in states) != STATE_ORDER:
        raise ValueError(f"states must be orbit-major in order {STATE_ORDER}")
    orbit_ids = {state.orbit_id for state in states}
    if len(orbit_ids) != 1:
        raise ValueError("C/D/M states must share one orbit_id")
    complete, distractor, missing = states
    if complete.assistant_content != distractor.assistant_content:
        raise ValueError("C and D assistant targets must be byte-identical")
    if complete.completion_ids != distractor.completion_ids:
        raise ValueError("C and D completion token ordinals must be exactly shared")
    if missing.assistant_content != FIXED_UNANSWERABLE_TARGET:
        raise ValueError("M must use the frozen U target")
    return EncodedOrbit(
        orbit_id=complete.orbit_id,
        complete=complete,
        distractor=distractor,
        missing=missing,
    )


class PreparedOrbitDataset(Dataset[EncodedOrbit]):
    """Eagerly validate canonical consecutive C/D/M triples before model load."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
    ) -> None:
        if not records or len(records) % 3:
            raise ValueError("prepared records must be non-empty complete C/D/M triples")
        validate_action_token_ids(tokenizer)
        orbits: list[EncodedOrbit] = []
        seen: set[str] = set()
        for start in range(0, len(records), 3):
            group = records[start : start + 3]
            states = tuple(record.get("state") for record in group)
            orbit_ids = tuple(record.get("orbit_id") for record in group)
            if states != STATE_ORDER:
                raise ValueError(f"records {start}:{start + 3} are not ordered C/D/M")
            if not isinstance(orbit_ids[0], str) or len(set(orbit_ids)) != 1:
                raise ValueError(f"records {start}:{start + 3} do not share one orbit_id")
            if orbit_ids[0] in seen:
                raise ValueError(f"duplicate prepared orbit_id: {orbit_ids[0]}")
            seen.add(orbit_ids[0])
            encoded = [
                encode_state(
                    tokenizer,
                    render_prepared_record(tokenizer, record),
                    max_length=max_length,
                )
                for record in group
            ]
            orbits.append(encode_orbit(encoded))
        self._orbits = tuple(orbits)
        self.stats = {
            "orbits": len(orbits),
            "states": len(orbits) * 3,
            "tokens": sum(len(state.input_ids) for orbit in orbits for state in orbit.states),
            "completion_tokens": sum(
                len(state.completion_ids) for orbit in orbits for state in orbit.states
            ),
            "max_length": max(
                len(state.input_ids) for orbit in orbits for state in orbit.states
            ),
            "truncated": 0,
        }

    def __len__(self) -> int:
        return len(self._orbits)

    def __getitem__(self, index: int) -> EncodedOrbit:
        return self._orbits[index]


class OrbitMajorCollator:
    """Flatten atomic orbits as C,D,M and build Qwen union-logit indices."""

    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: int = 8) -> None:
        if pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")
        if pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, orbits: Sequence[EncodedOrbit]) -> dict[str, Any]:
        if not orbits:
            raise ValueError("cannot collate an empty orbit batch")
        rows: list[EncodedState] = []
        for orbit in orbits:
            if not isinstance(orbit, EncodedOrbit):
                raise TypeError("OrbitMajorCollator accepts EncodedOrbit values")
            if tuple(state.state for state in orbit.states) != STATE_ORDER:
                raise ValueError("orbit state order drifted from C/D/M")
            rows.extend(orbit.states)

        lengths = [len(row.input_ids) for row in rows]
        width = math.ceil(max(lengths) / self.pad_to_multiple_of) * self.pad_to_multiple_of
        batch_size = len(rows)
        input_ids = torch.full((batch_size, width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
        labels = torch.full((batch_size, width), IGNORE_INDEX, dtype=torch.long)
        max_targets = max(len(row.completion_ids) for row in rows)
        target_ids = torch.zeros((batch_size, max_targets), dtype=torch.long)
        target_mask = torch.zeros((batch_size, max_targets), dtype=torch.bool)
        absolute_positions = torch.full((batch_size, max_targets), -1, dtype=torch.long)

        for index, row in enumerate(rows):
            size = len(row.input_ids)
            targets = len(row.completion_ids)
            if (
                len(row.labels) != size
                or len(row.attention_mask) != size
                or any(value != 1 for value in row.attention_mask)
            ):
                raise ValueError("encoded input, labels, and attention mask are inconsistent")
            if tuple(row.labels[: row.prompt_tokens]) != (IGNORE_INDEX,) * row.prompt_tokens:
                raise ValueError("encoded prompt labels are not fully masked")
            if tuple(row.labels[row.prompt_tokens :]) != row.completion_ids:
                raise ValueError("encoded completion labels differ from completion IDs")
            expected_positions = tuple(range(row.prompt_tokens - 1, size - 1))
            if row.prediction_positions != expected_positions or targets != len(expected_positions):
                raise ValueError("encoded causal prediction positions are inconsistent")
            input_ids[index, :size] = torch.tensor(row.input_ids, dtype=torch.long)
            attention_mask[index, :size] = 1
            labels[index, :size] = torch.tensor(row.labels, dtype=torch.long)
            target_ids[index, :targets] = torch.tensor(row.completion_ids, dtype=torch.long)
            target_mask[index, :targets] = True
            absolute_positions[index, :targets] = torch.tensor(
                row.prediction_positions, dtype=torch.long
            )

        union = torch.unique(absolute_positions[target_mask], sorted=True)
        if union.numel() == 0 or int(union[0]) < 0 or int(union[-1]) >= width:
            raise ValueError("invalid completion prediction-position union")
        # searchsorted is an exact absolute-position -> returned-logit mapping.
        union_indices = torch.zeros_like(absolute_positions)
        union_indices[target_mask] = torch.searchsorted(union, absolute_positions[target_mask])
        if not torch.equal(union[union_indices[target_mask]], absolute_positions[target_mask]):
            raise AssertionError("logits_to_keep union mapping is not exact")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "prediction_positions": absolute_positions,
            "prediction_union_indices": union_indices,
            "logits_to_keep": union,
            "state_ids": torch.tensor([0, 1, 2] * len(orbits), dtype=torch.long),
            "orbit_ids": tuple(orbit.orbit_id for orbit in orbits),
            "row_orbit_ids": tuple(row.orbit_id for row in rows),
        }
