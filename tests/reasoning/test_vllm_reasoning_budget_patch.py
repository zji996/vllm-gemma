#!/usr/bin/env python3
"""Unit tests for the vLLM reasoning-budget patch helpers."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PATCH_SCRIPT = ROOT / "scripts" / "patch-vllm-openai-reasoning-budget.py"


def load_patch_module():
    spec = importlib.util.spec_from_file_location("vllm_reasoning_budget_patch", PATCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load patch module from {PATCH_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = load_patch_module()


class VllmReasoningBudgetPatchTests(unittest.TestCase):
    def test_reasoning_effort_defaults_map_to_budgets(self) -> None:
        self.assertEqual(
            patch.reasoning_budget_defaults_for_tests("low")[0],
            32,
        )
        self.assertEqual(
            patch.reasoning_budget_defaults_for_tests("medium")[0],
            128,
        )
        self.assertEqual(
            patch.reasoning_budget_defaults_for_tests("high")[0],
            512,
        )
        self.assertEqual(
            patch.reasoning_budget_defaults_for_tests("none"),
            (None, None),
        )

    def test_apply_reasoning_defaults_preserves_explicit_budget(self) -> None:
        budget, extra_args = patch.apply_reasoning_defaults_for_tests(
            "medium",
            thinking_token_budget=77,
        )
        self.assertEqual(budget, 77)
        self.assertIn("reasoning_repetition_detection", extra_args)

    def test_tail_repetition_detection_triggers_after_three_repeats(self) -> None:
        repetition_cfg = {
            "max_pattern_size": 4,
            "min_pattern_size": 4,
            "min_count": 3,
        }
        token_ids = [10, 11, 12, 13] * 3
        self.assertTrue(
            patch.repeated_pattern_detected_while_thinking_for_tests(
                token_ids,
                repetition_cfg,
                in_think=True,
            )
        )

    def test_non_tail_repetition_does_not_trigger(self) -> None:
        repetition_cfg = {
            "max_pattern_size": 4,
            "min_pattern_size": 4,
            "min_count": 3,
        }
        token_ids = [1, 2, 3, 4] * 2 + [8, 9, 10, 11]
        self.assertFalse(
            patch.repeated_pattern_detected_while_thinking_for_tests(
                token_ids,
                repetition_cfg,
                in_think=True,
            )
        )

    def test_repetition_detection_is_disabled_outside_thinking(self) -> None:
        repetition_cfg = {
            "max_pattern_size": 4,
            "min_pattern_size": 4,
            "min_count": 3,
        }
        token_ids = [10, 11, 12, 13] * 3
        self.assertFalse(
            patch.repeated_pattern_detected_while_thinking_for_tests(
                token_ids,
                repetition_cfg,
                in_think=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
