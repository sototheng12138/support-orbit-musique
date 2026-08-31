"""Small, dependency-free adapter matching the official MuSiQue metrics.

The normalization and scoring rules intentionally mirror
``vendor/musique/metrics/{answer,support}.py``.  Keeping the functions here
avoids importing the vendored evaluator through its historical top-level
``metrics`` package path while preserving its exact semantics.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OfficialGold:
    instance_id: str
    orbit_id: str
    state: str
    answers: tuple[str, ...]
    support_indices: tuple[int, ...]
    answerable: bool


def normalize_answer(text: str) -> str:
    """Apply the official SQuAD/MuSiQue lowercase/article/punctuation rules."""

    lowered = text.lower()
    no_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation, flags=re.UNICODE)
    return " ".join(no_articles.split())


def answer_exact(gold: str, prediction: str) -> float:
    return float(normalize_answer(gold) == normalize_answer(prediction))


def answer_f1(gold: str, prediction: str) -> float:
    gold_tokens = normalize_answer(gold).split() if gold else []
    predicted_tokens = normalize_answer(prediction).split() if prediction else []
    common = Counter(gold_tokens) & Counter(predicted_tokens)
    same = sum(common.values())
    if not gold_tokens or not predicted_tokens:
        return float(gold_tokens == predicted_tokens)
    if same == 0:
        return 0.0
    precision = same / len(predicted_tokens)
    recall = same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def answer_scores(prediction: str, ground_truths: Sequence[str]) -> tuple[float, float]:
    """Return official max-over-primary-and-aliases EM and token F1."""

    if not ground_truths:
        raise ValueError("ground_truths must contain the primary answer")
    return (
        max(answer_exact(gold, prediction) for gold in ground_truths),
        max(answer_f1(gold, prediction) for gold in ground_truths),
    )


def support_scores(
    predicted_indices: Sequence[int], gold_indices: Sequence[int]
) -> tuple[float, float, float, float]:
    """Return official support EM, F1, precision, and recall."""

    predicted = set(map(int, predicted_indices))
    gold = set(map(int, gold_indices))
    true_positive = len(predicted & gold)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    precision = true_positive / (true_positive + false_positive) if predicted else 0.0
    recall = true_positive / (true_positive + false_negative) if gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = float(not false_positive and not false_negative)
    if not predicted and not gold:
        exact = f1 = 1.0
    return exact, f1, precision, recall


def _support_from_record(record: Mapping[str, Any]) -> tuple[int, ...]:
    if "gold_support_idxs" in record:
        values = record["gold_support_idxs"]
    elif "support_indices" in record:
        values = record["support_indices"]
    elif "paragraphs" in record:
        paragraphs = record["paragraphs"]
        if not isinstance(paragraphs, list):
            raise ValueError("paragraphs must be a list")
        values = [paragraph["idx"] for paragraph in paragraphs if paragraph["is_supporting"]]
    else:
        raise ValueError("gold record has no support indices")
    if not isinstance(values, list):
        raise ValueError("support indices must be a list")
    indices = tuple(int(value) for value in values)
    if indices != tuple(sorted(set(indices))):
        raise ValueError("gold support indices must be unique and sorted")
    if any(index < 0 or index > 19 for index in indices):
        raise ValueError("gold support index outside 0..19")
    return indices


def adapt_official_gold(record: Mapping[str, Any]) -> OfficialGold:
    """Validate the minimal official-plus-orbit gold schema."""

    instance_id = str(record["id"])
    orbit_id = str(record["orbit_id"])
    state = str(record["state"])
    if state not in {"C", "D", "M"}:
        raise ValueError(f"invalid state for {instance_id}: {state!r}")
    if not isinstance(record["answerable"], bool):
        raise ValueError(f"answerable must be bool for {instance_id}")
    answerable = record["answerable"]
    answer = record["answer"]
    aliases = record.get("answer_aliases", [])
    if not isinstance(answer, str) or not answer:
        raise ValueError(f"answer must be a nonempty string for {instance_id}")
    if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
        raise ValueError(f"answer_aliases must be a string list for {instance_id}")
    return OfficialGold(
        instance_id=instance_id,
        orbit_id=orbit_id,
        state=state,
        answers=(answer, *aliases),
        support_indices=_support_from_record(record),
        answerable=answerable,
    )


__all__ = [
    "OfficialGold",
    "adapt_official_gold",
    "answer_exact",
    "answer_f1",
    "answer_scores",
    "normalize_answer",
    "support_scores",
]
