"""Strict parser for the frozen Support-Orbit generation format."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
_HORIZONTAL_SPACE = r"[ \t]*"
_EVIDENCE_ITEM = r"P(?:0[0-9]|1[0-9])"
_OUTPUT_RE = re.compile(
    rf"\A{_HORIZONTAL_SPACE}"
    rf"(?P<status>[SU]){_HORIZONTAL_SPACE}\|{_HORIZONTAL_SPACE}"
    rf"evidence{_HORIZONTAL_SPACE}={_HORIZONTAL_SPACE}"
    rf"\[(?P<evidence>{_HORIZONTAL_SPACE}(?:{_EVIDENCE_ITEM}"
    rf"(?:{_HORIZONTAL_SPACE},{_HORIZONTAL_SPACE}{_EVIDENCE_ITEM})*)?"
    rf"{_HORIZONTAL_SPACE})\]"
    rf"{_HORIZONTAL_SPACE}\|{_HORIZONTAL_SPACE}"
    rf"answer{_HORIZONTAL_SPACE}={_HORIZONTAL_SPACE}"
    rf"(?P<answer>[^|\r\n]*?){_HORIZONTAL_SPACE}\Z"
)


@dataclass(frozen=True, slots=True)
class ParsedPrediction:
    """A parse result; malformed generations remain data rather than exceptions."""

    raw: str
    parse_valid: bool
    status: Literal["S", "U"] | None = None
    support_indices: tuple[int, ...] = ()
    answer: str = ""
    error: str | None = None

    @property
    def predicted_answerable(self) -> bool | None:
        if not self.parse_valid:
            return None
        return self.status == "S"

    @property
    def refused(self) -> bool:
        return self.parse_valid and self.status == "U"


def _invalid(raw: object, error: str) -> ParsedPrediction:
    return ParsedPrediction(raw=raw if isinstance(raw, str) else repr(raw), parse_valid=False, error=error)


def parse_prediction(raw: object) -> ParsedPrediction:
    """Parse one generation without repairing, truncating, or extracting substrings.

    Horizontal whitespace is tolerated around delimiters.  Newlines, prose before
    or after the record, duplicate/unsorted evidence indices, inconsistent
    sufficiency labels, and indices outside P00--P19 are rejected.
    """

    if not isinstance(raw, str):
        return _invalid(raw, "prediction_not_string")
    match = _OUTPUT_RE.fullmatch(raw)
    if match is None:
        return _invalid(raw, "grammar_mismatch")

    evidence_text = match.group("evidence").strip()
    support_indices = tuple(
        int(item.strip()[1:]) for item in evidence_text.split(",") if item.strip()
    )
    if support_indices != tuple(sorted(set(support_indices))):
        return _invalid(raw, "evidence_not_unique_strictly_sorted")

    status = match.group("status")
    answer = match.group("answer").strip()
    if not answer:
        return _invalid(raw, "empty_answer")
    if status == "S":
        if not support_indices:
            return _invalid(raw, "answerable_without_evidence")
        if answer == INSUFFICIENT_EVIDENCE:
            return _invalid(raw, "answerable_with_sentinel")
    else:
        if support_indices:
            return _invalid(raw, "unanswerable_with_evidence")
        if answer != INSUFFICIENT_EVIDENCE:
            return _invalid(raw, "unanswerable_without_sentinel")

    return ParsedPrediction(
        raw=raw,
        parse_valid=True,
        status=status,  # type: ignore[arg-type]
        support_indices=support_indices,
        answer=answer,
    )


__all__ = ["INSUFFICIENT_EVIDENCE", "ParsedPrediction", "parse_prediction"]
