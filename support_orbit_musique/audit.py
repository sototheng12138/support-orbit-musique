"""Frozen surface-separability and tokenizer-length audits.

The surface feature family deliberately excludes the question, answers, support
flags, provenance, and state labels.  The semantic diagnostic appends only
question-to-document lexical overlap.  No answer-derived feature is used by
either classifier.
"""

from __future__ import annotations

import hashlib
import math
import re
import string
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SURFACE_FEATURE_NAMES = (
    "document_count",
    "total_characters",
    "mean_document_characters",
    "std_document_characters",
    "min_document_characters",
    "max_document_characters",
    "total_word_tokens",
    "mean_document_tokens",
    "std_document_tokens",
    "mean_title_tokens",
    "std_title_tokens",
    "exact_title_duplicate_fraction",
    "exact_text_duplicate_fraction",
    "mean_pairwise_title_jaccard",
    "mean_pairwise_text_jaccard",
    "digit_character_fraction",
    "punctuation_character_fraction",
    "slot_length_slope",
)

SEMANTIC_EXTRA_FEATURE_NAMES = (
    "question_document_jaccard_mean",
    "question_document_jaccard_max",
    "question_document_jaccard_top2_sum",
    "question_title_jaccard_max",
    "question_overlap_document_fraction",
)

SEMANTIC_FEATURE_NAMES = SURFACE_FEATURE_NAMES + SEMANTIC_EXTRA_FEATURE_NAMES
PAIR_NAMES = (("C", "D"), ("D", "M"), ("C", "M"))


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def token_set(text: str) -> frozenset[str]:
    value = normalize_text(text)
    return frozenset(value.split()) if value else frozenset()


def jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    if not first and not second:
        return 0.0
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _duplicate_fraction(values: Sequence[str]) -> float:
    if len(values) < 2:
        return 0.0
    pairs = len(values) * (len(values) - 1) // 2
    duplicates = sum(
        values[left] == values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return duplicates / pairs


def _mean_pairwise_jaccard(values: Sequence[frozenset[str]]) -> float:
    if len(values) < 2:
        return 0.0
    similarities = [
        jaccard(values[left], values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    return _mean(similarities)


def surface_features(paragraphs: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    """Return the frozen 18-dimensional structural/surface vector."""

    texts = [str(paragraph["paragraph_text"]) for paragraph in paragraphs]
    titles = [str(paragraph["title"]) for paragraph in paragraphs]
    combined = [f"{title} {text}" for title, text in zip(titles, texts, strict=True)]
    char_lengths = [float(len(value)) for value in combined]
    text_tokens = [token_set(value) for value in texts]
    title_tokens = [token_set(value) for value in titles]
    word_lengths = [float(len(token_set(value))) for value in combined]
    title_lengths = [float(len(value)) for value in title_tokens]
    normalized_titles = [normalize_text(value) for value in titles]
    normalized_texts = [normalize_text(value) for value in texts]
    all_characters = "".join(combined)
    denominator = max(1, len(all_characters))
    digit_fraction = sum(character.isdigit() for character in all_characters) / denominator
    punctuation_fraction = (
        sum(character in string.punctuation for character in all_characters) / denominator
    )
    count = len(char_lengths)
    if count > 1:
        mean_slot = (count - 1) / 2.0
        mean_length = _mean(char_lengths)
        numerator = sum(
            (slot - mean_slot) * (length - mean_length)
            for slot, length in enumerate(char_lengths)
        )
        slot_denominator = sum((slot - mean_slot) ** 2 for slot in range(count))
        slot_slope = numerator / slot_denominator if slot_denominator else 0.0
    else:
        slot_slope = 0.0
    return (
        float(count),
        sum(char_lengths),
        _mean(char_lengths),
        _std(char_lengths),
        min(char_lengths, default=0.0),
        max(char_lengths, default=0.0),
        sum(word_lengths),
        _mean(word_lengths),
        _std(word_lengths),
        _mean(title_lengths),
        _std(title_lengths),
        _duplicate_fraction(normalized_titles),
        _duplicate_fraction(normalized_texts),
        _mean_pairwise_jaccard(title_tokens),
        _mean_pairwise_jaccard(text_tokens),
        digit_fraction,
        punctuation_fraction,
        slot_slope,
    )


def semantic_features(
    question: str, paragraphs: Sequence[Mapping[str, Any]]
) -> tuple[float, ...]:
    """Append question-document overlap to the surface-only vector."""

    question_tokens = token_set(question)
    document_tokens = [
        token_set(f"{paragraph['title']} {paragraph['paragraph_text']}")
        for paragraph in paragraphs
    ]
    title_tokens = [token_set(str(paragraph["title"])) for paragraph in paragraphs]
    overlaps = [jaccard(question_tokens, value) for value in document_tokens]
    title_overlaps = [jaccard(question_tokens, value) for value in title_tokens]
    descending = sorted(overlaps, reverse=True)
    return surface_features(paragraphs) + (
        _mean(overlaps),
        max(overlaps, default=0.0),
        sum(descending[:2]),
        max(title_overlaps, default=0.0),
        sum(value > 0.0 for value in overlaps) / max(1, len(overlaps)),
    )


def relative_feature_distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Symmetric, scale-free distance used only to rank donor pairs."""

    return sum(
        abs(left - right) / (1.0 + abs(left) + abs(right))
        for left, right in zip(first, second, strict=True)
    )


def _stable_folds(orbit_ids: Sequence[str], seed: str, folds: int = 5) -> dict[str, int]:
    ordered = sorted(
        set(orbit_ids),
        key=lambda value: (
            hashlib.sha256(f"{seed}\n{value}".encode()).hexdigest(),
            value,
        ),
    )
    return {orbit_id: rank % folds for rank, orbit_id in enumerate(ordered)}


def grouped_oof_auc(
    records: Sequence[Mapping[str, Any]],
    *,
    semantic: bool,
    seed: str,
) -> dict[str, Any]:
    """Fit a fixed logistic probe with orbit-grouped deterministic 5-fold OOF."""

    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    usable = [record for record in records if record["split"] != "shadow"]
    orbit_ids = [str(record["orbit_id"]) for record in usable]
    fold_by_orbit = _stable_folds(orbit_ids, seed)
    feature_function = semantic_features if semantic else surface_features
    names = SEMANTIC_FEATURE_NAMES if semantic else SURFACE_FEATURE_NAMES
    pair_results: dict[str, Any] = {}
    for first, second in PAIR_NAMES:
        pair = [record for record in usable if record["state"] in {first, second}]
        features = np.asarray(
            [
                feature_function(record["question"], record["paragraphs"])
                if semantic
                else feature_function(record["paragraphs"])
                for record in pair
            ],
            dtype=np.float64,
        )
        labels = np.asarray([record["state"] == second for record in pair], dtype=np.int64)
        folds = np.asarray(
            [fold_by_orbit[str(record["orbit_id"])] for record in pair], dtype=np.int64
        )
        predictions = np.zeros(len(pair), dtype=np.float64)
        fold_counts: dict[str, int] = {}
        for fold in range(5):
            train = folds != fold
            held_out = folds == fold
            fold_counts[str(fold)] = int(held_out.sum() // 2)
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=1729,
                ),
            )
            estimator.fit(features[train], labels[train])
            predictions[held_out] = estimator.predict_proba(features[held_out])[:, 1]
        raw_auc = float(roc_auc_score(labels, predictions))
        separability = max(raw_auc, 1.0 - raw_auc)
        pair_results[f"{first}-{second}"] = {
            "raw_auc": raw_auc,
            "separability_auc": separability,
            "examples": len(pair),
            "orbits": len(pair) // 2,
            "fold_orbit_counts": fold_counts,
        }
    return {
        "feature_family": "semantic_question_overlap" if semantic else "surface_only",
        "feature_names": list(names),
        "feature_count": len(names),
        "classifier": {
            "pipeline": "StandardScaler -> LogisticRegression",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 2000,
            "random_state": 1729,
            "folds": 5,
            "grouping": "all states of an orbit remain in one fold",
            "fold_assignment": "SHA256(seed + newline + orbit_id), sorted, round-robin",
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
        },
        "shadow_excluded": True,
        "pairs": pair_results,
    }


def tokenizer_length_audit(
    records: Sequence[Mapping[str, Any]],
    tokenizer_path: Path,
    max_tokens: int,
) -> dict[str, Any]:
    """Run the canonical formatter and strict encoder over every train/dev state."""

    import transformers
    from transformers import AutoTokenizer

    from .formatting import Paragraph, render_state
    from .sft import encode_state, validate_action_token_ids

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=False
    )
    validate_action_token_ids(tokenizer)
    usable = [record for record in records if record["split"] != "shadow"]
    lengths: list[int] = []
    split_max: dict[str, int] = {}
    over_limit = 0
    record_contract_mismatches = 0
    for record in usable:
        paragraphs = [
            Paragraph(
                index=int(paragraph["idx"]),
                title=str(paragraph["title"]),
                text=str(paragraph["paragraph_text"]),
            )
            for paragraph in record["paragraphs"]
        ]
        state = str(record["state"])
        kwargs: dict[str, Any] = {}
        if state in {"C", "D"}:
            kwargs = {
                "support_indices": record["gold_support_idxs"],
                "answer": record["answer"],
            }
        rendered = render_state(
            tokenizer,
            orbit_id=str(record["orbit_id"]),
            state=state,
            question=str(record["question"]),
            paragraphs=paragraphs,
            **kwargs,
        )
        record_contract_mismatches += not (
            record["prompt"] == rendered.user_content
            and record["target"] == rendered.assistant_content
            and record["messages"]
            == [
                {"role": "user", "content": rendered.user_content},
                {"role": "assistant", "content": rendered.assistant_content},
            ]
        )
        try:
            encoded = encode_state(tokenizer, rendered, max_length=max_tokens)
            length = len(encoded.input_ids)
        except ValueError as exc:
            if "tokens exceed max_length" not in str(exc):
                raise
            tokenized = tokenizer(
                rendered.full_text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            length = len(tokenized)
            over_limit += 1
        lengths.append(length)
        split = str(record["split"])
        split_max[split] = max(split_max.get(split, 0), length)
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        if not ordered:
            return 0
        return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]

    identity_files = ["tokenizer.json", "tokenizer_config.json", "config.json"]
    file_hashes: dict[str, str] = {}
    for name in identity_files:
        path = tokenizer_path / name
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            file_hashes[name] = digest.hexdigest()
    return {
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": transformers.__version__,
        "identity_sha256": file_hashes,
        "format": "formatting.render_state -> sft.encode_state",
        "native_chat_template": {
            "add_generation_prompt": True,
            "enable_thinking": False,
            "completion_only_labels": True,
        },
        "truncation": "forbidden; strict encoder raises over max_length",
        "max_allowed_tokens": max_tokens,
        "records": len(usable),
        "splits": ["train", "dev"],
        "shadow_excluded": True,
        "min_tokens": min(lengths, default=0),
        "p50_tokens": percentile(0.50),
        "p95_tokens": percentile(0.95),
        "p99_tokens": percentile(0.99),
        "max_tokens": max(lengths, default=0),
        "split_max_tokens": dict(sorted(split_max.items())),
        "over_limit_records": over_limit,
        "prepared_vs_canonical_mismatches": record_contract_mismatches,
        "prepared_contract_byte_identical": record_contract_mismatches == 0,
        "zero_truncation_pass": over_limit == 0 and record_contract_mismatches == 0,
    }
