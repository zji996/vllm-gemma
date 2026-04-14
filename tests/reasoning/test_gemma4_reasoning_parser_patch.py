#!/usr/bin/env python3
"""Unit tests for the local Gemma4 reasoning parser patch helpers."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PATCH_SCRIPT = ROOT / "scripts" / "patch-vllm-gemma4-reasoning-parser.py"


def load_patch_module():
    spec = importlib.util.spec_from_file_location("gemma4_reasoning_patch", PATCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load patch module from {PATCH_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = load_patch_module()


class Gemma4ReasoningParserPatchTests(unittest.TestCase):
    def test_non_thinking_output_is_left_as_content(self) -> None:
        reasoning, content = patch.simulate_patched_extract_reasoning(
            "56088",
            expects_reasoning=False,
        )
        self.assertIsNone(reasoning)
        self.assertEqual(content, "56088")

    def test_thought_only_output_becomes_reasoning_only(self) -> None:
        reasoning, content = patch.simulate_patched_extract_reasoning(
            "thought\nThe user asks for multiplication.",
            expects_reasoning=True,
        )
        self.assertEqual(reasoning, "The user asks for multiplication.")
        self.assertIsNone(content)

    def test_thought_prefix_followed_by_channel_becomes_final_content(self) -> None:
        reasoning, content = patch.simulate_patched_extract_reasoning(
            "thought\n<channel|>579",
            expects_reasoning=True,
        )
        self.assertIsNone(reasoning)
        self.assertEqual(content, "579")

    def test_full_channel_output_is_split(self) -> None:
        reasoning, content = patch.simulate_patched_extract_reasoning(
            "<|channel>thought\nWork it out carefully.<channel|>579"
        )
        self.assertEqual(reasoning, "Work it out carefully.")
        self.assertEqual(content, "579")

    def test_finalize_cleans_turn_and_eos_markers(self) -> None:
        reasoning, content = patch.finalize_reasoning_extraction_for_tests(
            "thought\nCheck arithmetic.",
            "579<turn|><eos>",
        )
        self.assertEqual(reasoning, "Check arithmetic.")
        self.assertEqual(content, "579")

    def test_heuristic_final_answer_is_disabled_by_default(self) -> None:
        reasoning, content = patch.simulate_patched_extract_reasoning(
            "thought\nLet's solve it.\nFinal Answer: 56088",
            expects_reasoning=True,
        )
        self.assertEqual(reasoning, "Let's solve it.\nFinal Answer: 56088")
        self.assertIsNone(content)

    def test_heuristic_final_answer_can_salvage_explicit_tail_marker(self) -> None:
        reasoning, content = patch.simulate_patched_extract_reasoning(
            "thought\nLet's solve it.\nFinal Answer: 56088",
            expects_reasoning=True,
            heuristic_enabled=True,
        )
        self.assertEqual(reasoning, "Let's solve it.")
        self.assertEqual(content, "56088")

    def test_heuristic_accepts_multiline_answer_after_marker(self) -> None:
        reasoning, content = patch.simulate_patched_extract_reasoning(
            "thought\nPlan it first.\nFinal Answer:\n579",
            expects_reasoning=True,
            heuristic_enabled=True,
        )
        self.assertEqual(reasoning, "Plan it first.")
        self.assertEqual(content, "579")

    def test_heuristic_rejects_channel_like_text(self) -> None:
        reasoning, content = patch.simulate_patched_extract_reasoning(
            "thought\nPlan it first.\nFinal Answer: <|channel>579",
            expects_reasoning=True,
            heuristic_enabled=True,
        )
        self.assertEqual(reasoning, "Plan it first.\nFinal Answer: <|channel>579")
        self.assertIsNone(content)


if __name__ == "__main__":
    unittest.main()
