"""Canonical MuSiQue prompts and native-Qwen assistant boundaries.

The formatter deliberately accepts explicit semantic fields instead of coupling
the training code to a particular JSONL layout.  Source-only annotations (for
example ``is_supporting`` and decomposition answers) therefore cannot leak into
the rendered prompt by accident.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


State = Literal["C", "D", "M"]
STATE_ORDER: tuple[State, ...] = ("C", "D", "M")
FIXED_UNANSWERABLE_TARGET = "U | evidence=[] | answer=INSUFFICIENT_EVIDENCE"
_CHAT_CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Paragraph:
    """One visible paragraph in the fixed P00--P19 registry."""

    index: int
    title: str
    text: str


@dataclass(frozen=True)
class RenderedState:
    """One state after native chat rendering, before tokenization."""

    orbit_id: str
    state: State
    user_content: str
    assistant_content: str
    prompt: str
    full_text: str
    completion: str
    prompt_sha256: str
    completion_sha256: str


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _one_line(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if any(token in value for token in _CHAT_CONTROL_TOKENS):
        raise ValueError(f"{field} contains a raw chat control token")
    return _SPACE_RE.sub(" ", value).strip()


def _paragraphs(paragraphs: Sequence[Paragraph]) -> tuple[Paragraph, ...]:
    if len(paragraphs) != 20:
        raise ValueError(f"exactly 20 paragraphs are required, found {len(paragraphs)}")
    normalized: list[Paragraph] = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, Paragraph):
            raise TypeError("paragraphs must contain Paragraph values")
        normalized.append(
            Paragraph(
                index=paragraph.index,
                title=_one_line(paragraph.title, f"P{paragraph.index:02d}.title"),
                text=_one_line(paragraph.text, f"P{paragraph.index:02d}.text"),
            )
        )
    indices = [paragraph.index for paragraph in normalized]
    if indices != list(range(20)):
        raise ValueError("paragraphs must be ordered exactly P00 through P19")
    return tuple(normalized)


def render_user_content(question: str, paragraphs: Sequence[Paragraph]) -> str:
    """Render only fields allowed at inference time.

    In particular, this function has no parameters for gold answerability,
    answers, subanswers, or support flags.
    """

    clean_question = _one_line(question, "question")
    clean_paragraphs = _paragraphs(paragraphs)
    registry = "\n".join(
        f"[P{paragraph.index:02d}] {paragraph.title}: {paragraph.text}"
        for paragraph in clean_paragraphs
    )
    return (
        "Use only the paragraph registry to answer the question. Select the complete "
        "set of supporting paragraph IDs.\n"
        "Return exactly one line in one of these forms:\n"
        "S | evidence=[P00,P01] | answer=<answer>\n"
        f"{FIXED_UNANSWERABLE_TARGET}\n\n"
        f"Question: {clean_question}\n"
        f"Paragraph registry:\n{registry}"
    )


def supported_target(support_indices: Sequence[int], answer: str) -> str:
    """Return the canonical supported-answer line."""

    if not support_indices:
        raise ValueError("a supported target requires at least one evidence index")
    indices = [int(index) for index in support_indices]
    if any(index < 0 or index >= 20 for index in indices):
        raise ValueError("support indices must lie in [0, 19]")
    if len(set(indices)) != len(indices):
        raise ValueError("support indices must be unique")
    if indices != sorted(indices):
        raise ValueError("support indices must be in ascending registry order")
    if not isinstance(answer, str):
        raise ValueError("answer must be a non-empty string")
    if "|" in answer or "\n" in answer or "\r" in answer:
        raise ValueError("answer contains a reserved delimiter or line break")
    clean_answer = _one_line(answer, "answer")
    evidence = ",".join(f"P{index:02d}" for index in indices)
    return f"S | evidence=[{evidence}] | answer={clean_answer}"


def native_chat_boundary(
    tokenizer: Any,
    *,
    orbit_id: str,
    state: State,
    user_content: str,
    assistant_content: str,
) -> RenderedState:
    """Apply the tokenizer-owned Qwen template and freeze an exact boundary."""

    clean_orbit = _one_line(orbit_id, "orbit_id")
    if state not in STATE_ORDER:
        raise ValueError(f"state must be one of {STATE_ORDER}")
    if not isinstance(user_content, str) or not user_content.strip():
        raise ValueError("user_content must be non-empty")
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        raise ValueError("assistant_content must be non-empty")
    if any(token in user_content for token in _CHAT_CONTROL_TOKENS):
        raise ValueError("user_content contains a raw chat control token")
    if any(token in assistant_content for token in _CHAT_CONTROL_TOKENS):
        raise ValueError("assistant_content contains a raw chat control token")
    messages = [{"role": "user", "content": user_content}]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    full_text = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": assistant_content}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("tokenizer returned an empty generation prompt")
    if not isinstance(full_text, str) or not full_text:
        raise ValueError("tokenizer returned an empty full conversation")
    if not full_text.startswith(prompt):
        raise ValueError("native full conversation does not have the prompt as an exact prefix")
    completion = full_text[len(prompt) :]
    if not completion or not completion.startswith(assistant_content):
        raise ValueError("native assistant suffix is empty or changed the target prefix")
    return RenderedState(
        orbit_id=clean_orbit,
        state=state,
        user_content=user_content,
        assistant_content=assistant_content,
        prompt=prompt,
        full_text=full_text,
        completion=completion,
        prompt_sha256=_sha256_text(prompt),
        completion_sha256=_sha256_text(completion),
    )


def render_state(
    tokenizer: Any,
    *,
    orbit_id: str,
    state: State,
    question: str,
    paragraphs: Sequence[Paragraph],
    support_indices: Sequence[int] | None = None,
    answer: str | None = None,
) -> RenderedState:
    """Render one C, D, or M state with its canonical target."""

    user_content = render_user_content(question, paragraphs)
    if state in ("C", "D"):
        if support_indices is None or answer is None:
            raise ValueError(f"state {state} requires support_indices and answer")
        target = supported_target(support_indices, answer)
    elif state == "M":
        if support_indices is not None or answer is not None:
            raise ValueError("state M must not receive gold support indices or answer")
        target = FIXED_UNANSWERABLE_TARGET
    else:
        raise ValueError(f"state must be one of {STATE_ORDER}")
    return native_chat_boundary(
        tokenizer,
        orbit_id=orbit_id,
        state=state,
        user_content=user_content,
        assistant_content=target,
    )


def render_prepared_record(
    tokenizer: Any,
    record: Mapping[str, Any],
    *,
    require_training_read: bool = True,
) -> RenderedState:
    """Whitelist-adapt one prepared-data record to the canonical formatter.

    The adapter intentionally ignores pre-rendered ``prompt``, ``messages``,
    ``target``, decomposition fields, and paragraph support flags.  They remain
    useful for provenance/evaluation but are not model inputs.
    """

    if not isinstance(record, Mapping):
        raise TypeError("prepared record must be a mapping")
    if require_training_read and (
        record.get("training_read_allowed") is not True or record.get("sealed") is True
    ):
        raise ValueError("prepared record is not authorized for training reads")
    orbit_id = record.get("orbit_id")
    state = record.get("state")
    question = record.get("question")
    raw_paragraphs = record.get("paragraphs")
    if not isinstance(orbit_id, str) or state not in STATE_ORDER or not isinstance(question, str):
        raise ValueError("prepared record has invalid orbit_id/state/question")
    if not isinstance(raw_paragraphs, Sequence) or isinstance(raw_paragraphs, (str, bytes)):
        raise ValueError("prepared record paragraphs must be a sequence")
    converted: list[Paragraph] = []
    for value in raw_paragraphs:
        if not isinstance(value, Mapping):
            raise ValueError("prepared paragraph must be a mapping")
        index, title, text = value.get("idx"), value.get("title"), value.get("paragraph_text")
        if not isinstance(index, int) or not isinstance(title, str) or not isinstance(text, str):
            raise ValueError("prepared paragraph lacks idx/title/paragraph_text")
        converted.append(Paragraph(index=index, title=title, text=text))
    if state in ("C", "D"):
        support = record.get("gold_support_idxs")
        answer = record.get("answer")
        if not isinstance(support, Sequence) or isinstance(support, (str, bytes)):
            raise ValueError("supported prepared record lacks gold_support_idxs")
        if not isinstance(answer, str):
            raise ValueError("supported prepared record lacks answer")
        return render_state(
            tokenizer,
            orbit_id=orbit_id,
            state=state,
            question=question,
            paragraphs=converted,
            support_indices=[int(value) for value in support],
            answer=answer,
        )
    return render_state(
        tokenizer,
        orbit_id=orbit_id,
        state="M",
        question=question,
        paragraphs=converted,
    )
