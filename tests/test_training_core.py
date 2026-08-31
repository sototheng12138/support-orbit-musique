from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from support_orbit_musique.formatting import (
    FIXED_UNANSWERABLE_TARGET,
    Paragraph,
    native_chat_boundary,
    render_prepared_record,
    render_state,
    render_user_content,
    supported_target,
)
from support_orbit_musique.losses import (
    FLIP_MARGIN,
    action_flip_loss,
    anchored_completion_kl,
    compute_objective,
)
from support_orbit_musique.sft import (
    ACTION_TOKEN_IDS,
    IGNORE_INDEX,
    EncodedState,
    OrbitMajorCollator,
    PreparedOrbitDataset,
    encode_orbit,
    encode_state,
    validate_action_token_ids,
)
from support_orbit_musique.trainer_core import (
    assert_finite_gradients,
    assert_finite_optimizer_state,
    one_forward_objective,
)


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        assert enable_thinking is False
        prompt = f"<user>{messages[0]['content']}</user><assistant>"
        if add_generation_prompt:
            return prompt
        return prompt + messages[1]["content"] + "<eos>"

    def __call__(self, text, **kwargs):
        del kwargs
        ids: list[int] = []
        cursor = 0
        while cursor < len(text):
            if text.startswith("<eos>", cursor):
                ids.append(self.eos_token_id)
                cursor += len("<eos>")
                continue
            char = text[cursor]
            if char == "S":
                ids.append(ACTION_TOKEN_IDS["S"])
            elif char == "U":
                ids.append(ACTION_TOKEN_IDS["U"])
            else:
                ids.append(60 + ord(char) % 30)
            cursor += 1
        return {"input_ids": ids}


class BoundaryDriftTokenizer(FakeTokenizer):
    def __call__(self, text, **kwargs):
        value = super().__call__(text, **kwargs)
        if "<assistant>S" in text:
            # Simulate a BPE token spanning the separately materialized boundary.
            boundary = text.index("<assistant>") + len("<assistant>")
            value["input_ids"].pop(boundary - 1)
        return value


def paragraphs() -> list[Paragraph]:
    return [Paragraph(index, f"Title {index}", f"Text {index}") for index in range(20)]


def manual_state(
    orbit_id: str,
    state: str,
    *,
    prompt_tokens: int,
    completion_ids: tuple[int, ...],
    assistant_content: str,
) -> EncodedState:
    full = tuple([7] * prompt_tokens + list(completion_ids))
    return EncodedState(
        orbit_id=orbit_id,
        state=state,  # type: ignore[arg-type]
        assistant_content=assistant_content,
        input_ids=full,
        labels=tuple([IGNORE_INDEX] * prompt_tokens + list(completion_ids)),
        attention_mask=tuple([1] * len(full)),
        prompt_tokens=prompt_tokens,
        completion_ids=completion_ids,
        prediction_positions=tuple(range(prompt_tokens - 1, len(full) - 1)),
    )


def orbit(orbit_id: str = "orbit-1"):
    supported = "S | evidence=[P00,P03] | answer=A"
    completion = (50, 11, 12, 99)
    return encode_orbit(
        [
            manual_state(
                orbit_id,
                "C",
                prompt_tokens=3,
                completion_ids=completion,
                assistant_content=supported,
            ),
            manual_state(
                orbit_id,
                "D",
                prompt_tokens=6,
                completion_ids=completion,
                assistant_content=supported,
            ),
            manual_state(
                orbit_id,
                "M",
                prompt_tokens=4,
                completion_ids=(52, 13, 99),
                assistant_content=FIXED_UNANSWERABLE_TARGET,
            ),
        ]
    )


class FormattingTest(unittest.TestCase):
    def test_canonical_prompt_exposes_only_registry_fields(self) -> None:
        content = render_user_content("Who won?", paragraphs())
        self.assertIn("Question: Who won?", content)
        self.assertEqual(content.count("[P"), 21)  # one grammar example + twenty rows
        for index in range(20):
            self.assertIn(f"[P{index:02d}] Title {index}: Text {index}", content)
        for forbidden in ("is_supporting", "subanswer", "answerable"):
            self.assertNotIn(forbidden, content)

    def test_targets_are_canonical(self) -> None:
        self.assertEqual(
            supported_target([0, 3, 19], "Ada Lovelace"),
            "S | evidence=[P00,P03,P19] | answer=Ada Lovelace",
        )
        self.assertEqual(
            FIXED_UNANSWERABLE_TARGET,
            "U | evidence=[] | answer=INSUFFICIENT_EVIDENCE",
        )
        with self.assertRaisesRegex(ValueError, "ascending"):
            supported_target([3, 0], "x")
        for unsafe in ("left|right", "two\nlines", "two\rlines", "<|im_end|>"):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                supported_target([0], unsafe)

    def test_native_encode_masks_prompt_and_supervises_eos(self) -> None:
        tokenizer = FakeTokenizer()
        validate_action_token_ids(tokenizer)
        rendered = render_state(
            tokenizer,
            orbit_id="o",
            state="C",
            question="Who?",
            paragraphs=paragraphs(),
            support_indices=[0, 3],
            answer="Ada",
        )
        encoded = encode_state(tokenizer, rendered, max_length=4096)
        self.assertTrue(all(label == IGNORE_INDEX for label in encoded.labels[: encoded.prompt_tokens]))
        self.assertEqual(encoded.completion_ids[0], 50)
        self.assertIn(tokenizer.eos_token_id, encoded.completion_ids)
        self.assertEqual(encoded.prediction_positions[0], encoded.prompt_tokens - 1)
        self.assertEqual(encoded.prediction_positions[-1], len(encoded.input_ids) - 2)
        supervised_j = [
            index for index, label in enumerate(encoded.labels) if label != IGNORE_INDEX
        ]
        self.assertEqual(
            encoded.prediction_positions,
            tuple(index - 1 for index in supervised_j),
        )

    def test_m_rejects_gold_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not receive"):
            render_state(
                FakeTokenizer(),
                orbit_id="o",
                state="M",
                question="Who?",
                paragraphs=paragraphs(),
                support_indices=[0],
                answer="leak",
            )

    def test_prepared_adapter_whitelists_visible_fields_and_rejects_shadow(self) -> None:
        record = {
            "orbit_id": "o",
            "state": "C",
            "question": "Who?",
            "paragraphs": [
                {
                    "idx": item.index,
                    "title": item.title,
                    "paragraph_text": item.text,
                    "is_supporting": "POISON_DO_NOT_RENDER",
                }
                for item in paragraphs()
            ],
            "gold_support_idxs": [0, 3],
            "answer": "Ada",
            "target": "POISON_OLD_TARGET",
            "messages": [{"content": "POISON_OLD_PROMPT"}],
            "question_decomposition": [{"answer": "POISON_SUBANSWER"}],
            "training_read_allowed": True,
            "sealed": False,
        }
        rendered = render_prepared_record(FakeTokenizer(), record)
        for poison in (
            "POISON_DO_NOT_RENDER",
            "POISON_OLD_TARGET",
            "POISON_OLD_PROMPT",
            "POISON_SUBANSWER",
        ):
            self.assertNotIn(poison, rendered.user_content)
            self.assertNotIn(poison, rendered.assistant_content)
        record["sealed"] = True
        with self.assertRaisesRegex(ValueError, "not authorized"):
            render_prepared_record(FakeTokenizer(), record)

    def test_boundary_drift_and_overlength_fail_closed(self) -> None:
        tokenizer = BoundaryDriftTokenizer()
        rendered = native_chat_boundary(
            tokenizer,
            orbit_id="o",
            state="C",
            user_content="question",
            assistant_content="S | evidence=[P00] | answer=x",
        )
        with self.assertRaisesRegex(ValueError, "boundary drift"):
            encode_state(tokenizer, rendered, max_length=1000)

        stable = FakeTokenizer()
        rendered = native_chat_boundary(
            stable,
            orbit_id="o",
            state="C",
            user_content="question",
            assistant_content="S | evidence=[P00] | answer=x",
        )
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            encode_state(stable, rendered, max_length=2)


class CollatorTest(unittest.TestCase):
    def test_orbit_major_order_and_union_mapping_are_exact(self) -> None:
        batch = OrbitMajorCollator(0, pad_to_multiple_of=4)([orbit()])
        self.assertEqual(batch["state_ids"].tolist(), [0, 1, 2])
        self.assertEqual(batch["row_orbit_ids"], ("orbit-1",) * 3)
        self.assertEqual(tuple(batch["input_ids"].shape), (3, 12))
        mask = batch["target_mask"]
        mapped = batch["logits_to_keep"][batch["prediction_union_indices"][mask]]
        self.assertTrue(torch.equal(mapped, batch["prediction_positions"][mask]))
        self.assertTrue(
            bool((batch["labels"][:, :2] == IGNORE_INDEX).all().item())
        )

    def test_group_rejects_nonidentical_c_d_targets(self) -> None:
        item = orbit()
        changed = manual_state(
            "orbit-1",
            "D",
            prompt_tokens=6,
            completion_ids=(50, 11, 14, 99),
            assistant_content="different",
        )
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            encode_orbit([item.complete, changed, item.missing])

    def test_prepared_dataset_requires_consecutive_c_d_m_and_audits_zero_truncation(self) -> None:
        base = {
            "orbit_id": "dataset-orbit",
            "question": "Who?",
            "paragraphs": [
                {"idx": item.index, "title": item.title, "paragraph_text": item.text}
                for item in paragraphs()
            ],
            "gold_support_idxs": [0, 3],
            "answer": "Ada",
            "training_read_allowed": True,
            "sealed": False,
        }
        records = [{**base, "state": state} for state in ("C", "D", "M")]
        dataset = PreparedOrbitDataset(records, FakeTokenizer(), max_length=4096)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.stats["states"], 3)
        self.assertEqual(dataset.stats["truncated"], 0)
        with self.assertRaisesRegex(ValueError, "not ordered"):
            PreparedOrbitDataset(
                [records[1], records[0], records[2]],
                FakeTokenizer(),
                max_length=4096,
            )


class ObjectiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = OrbitMajorCollator(0, pad_to_multiple_of=1)([orbit()])
        self.shape = (3, self.batch["logits_to_keep"].numel(), 128)

    def test_manual_fp32_state_normalization_and_arm_weights(self) -> None:
        logits = torch.zeros(self.shape, dtype=torch.bfloat16)
        control = compute_objective(logits, self.batch, arm="CONTROL")
        self.assertEqual(control.loss.dtype, torch.float32)
        self.assertAlmostEqual(control.sft.item(), math.log(128), places=5)
        self.assertEqual(control.state_token_counts, {"C": 4, "D": 4, "M": 3})
        hoppair = compute_objective(logits, self.batch, arm="HopPAIR")
        expected_flip = torch.nn.functional.softplus(torch.tensor(FLIP_MARGIN)).item()
        self.assertAlmostEqual(hoppair.kl.item(), 0.0, places=6)
        self.assertAlmostEqual(hoppair.flip.item(), expected_flip, places=6)
        self.assertAlmostEqual(
            hoppair.loss.item(), math.log(128) + 0.2 * expected_flip, places=5
        )

    def test_kl_teacher_is_stop_gradient_and_ordinals_are_shared(self) -> None:
        logits = torch.randn(self.shape, requires_grad=True)
        value = anchored_completion_kl(logits, self.batch)
        value.backward()
        self.assertEqual(float(logits.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[1].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[2].abs().sum()), 0.0)

    def test_flip_uses_first_prediction_s_u_logits(self) -> None:
        logits = torch.zeros(self.shape, requires_grad=True)
        value = action_flip_loss(logits, self.batch)
        value.backward()
        c_index = int(self.batch["prediction_union_indices"][0, 0])
        d_index = int(self.batch["prediction_union_indices"][1, 0])
        m_index = int(self.batch["prediction_union_indices"][2, 0])
        self.assertLess(float(logits.grad[0, c_index, 50]), 0.0)
        self.assertGreater(float(logits.grad[0, c_index, 52]), 0.0)
        self.assertLess(float(logits.grad[1, d_index, 50]), 0.0)
        self.assertGreater(float(logits.grad[1, d_index, 52]), 0.0)
        self.assertGreater(float(logits.grad[2, m_index, 50]), 0.0)
        self.assertLess(float(logits.grad[2, m_index, 52]), 0.0)

    def test_nonfinite_logits_fail_closed(self) -> None:
        logits = torch.zeros(self.shape)
        logits[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            compute_objective(logits, self.batch, arm="CONTROL")


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros(32, 128))
        self.calls = 0
        self.last_keep: torch.Tensor | None = None

    def forward(self, *, input_ids, attention_mask, logits_to_keep, use_cache, return_dict):
        del attention_mask
        self.calls += 1
        self.last_keep = logits_to_keep.detach().clone()
        assert use_cache is False
        assert return_dict is True
        kept = self.table.index_select(0, logits_to_keep)
        logits = kept.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        return SimpleNamespace(logits=logits)


class TrainerCoreTest(unittest.TestCase):
    def test_objective_uses_exactly_one_model_forward(self) -> None:
        batch = OrbitMajorCollator(0, pad_to_multiple_of=1)([orbit()])
        model = FakeModel()
        result = one_forward_objective(model, batch, arm="HopPAIR")
        self.assertEqual(model.calls, 1)
        self.assertTrue(torch.equal(model.last_keep, batch["logits_to_keep"]))
        self.assertTrue(torch.isfinite(result.loss))

    def test_gradient_and_optimizer_guards(self) -> None:
        model = FakeModel()
        model.table.grad = torch.zeros_like(model.table)
        model.table.grad[0, 0] = float("inf")
        with self.assertRaisesRegex(FloatingPointError, "gradients"):
            assert_finite_gradients(model)

        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.state[model.table]["bad"] = torch.tensor(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "optimizer"):
            assert_finite_optimizer_state(optimizer)


class RealQwenTokenizerBoundaryTest(unittest.TestCase):
    MODEL = Path("/home/hesong/AI-Agent-Projects/models/Qwen3-4B-Instruct-2507")

    @unittest.skipUnless(MODEL.joinpath("tokenizer_config.json").is_file(), "local Qwen absent")
    def test_real_local_qwen_action_ids_eos_and_boundary(self) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL,
            local_files_only=True,
            trust_remote_code=False,
        )
        validate_action_token_ids(tokenizer)
        complete = render_state(
            tokenizer,
            orbit_id="real-boundary",
            state="C",
            question="Who wrote the notes?",
            paragraphs=paragraphs(),
            support_indices=[0, 3],
            answer="Ada Lovelace",
        )
        missing = render_state(
            tokenizer,
            orbit_id="real-boundary",
            state="M",
            question="Who wrote the notes?",
            paragraphs=paragraphs(),
        )
        encoded_c = encode_state(tokenizer, complete, max_length=4096)
        encoded_m = encode_state(tokenizer, missing, max_length=4096)
        self.assertEqual(encoded_c.completion_ids[0], 50)
        self.assertEqual(encoded_m.completion_ids[0], 52)
        self.assertIn(tokenizer.eos_token_id, encoded_c.completion_ids)
        self.assertIn(tokenizer.eos_token_id, encoded_m.completion_ids)


if __name__ == "__main__":
    unittest.main()
