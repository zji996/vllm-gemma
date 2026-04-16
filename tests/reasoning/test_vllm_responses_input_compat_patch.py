#!/usr/bin/env python3
"""Unit tests for the local vLLM Responses API compatibility patch."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PATCH_SCRIPT = ROOT / "scripts" / "patch-vllm-responses-input-compat.py"


def load_patch_module():
    spec = importlib.util.spec_from_file_location("vllm_responses_input_compat", PATCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load patch module from {PATCH_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = load_patch_module()


class VllmResponsesInputCompatPatchTests(unittest.TestCase):
    def test_sanitize_drops_undefined_placeholders_but_keeps_text(self) -> None:
        sanitized = patch.sanitize_undefined_placeholders_for_tests(
            {
                "temperature": "[undefined]",
                "user": "undefined",
                "instructions": "[undefined]",
                "text": "[undefined]",
            }
        )
        self.assertEqual(sanitized, {"text": "[undefined]"})

    def test_assistant_shorthand_is_normalized_to_response_message_shape(self) -> None:
        normalized = patch.normalize_responses_assistant_input_for_tests(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "你好！很高兴为你服务。",
                    }
                ],
            }
        )
        self.assertEqual(normalized["type"], "message")
        self.assertEqual(normalized["status"], "completed")
        self.assertTrue(normalized["id"].startswith("msg_"))
        self.assertEqual(
            normalized["content"],
            [
                {
                    "type": "output_text",
                    "text": "你好！很高兴为你服务。",
                    "annotations": [],
                }
            ],
        )

    def test_end_to_end_normalizer_handles_real_payload_shape(self) -> None:
        normalized = patch.normalize_responses_request_for_tests(
            {
                "model": "gemma",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "你好"}],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "你好！很高兴为你服务。请问有什么我可以帮你的吗？",
                            }
                        ],
                        "id": "[undefined]",
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "你好"}],
                    },
                ],
                "temperature": "[undefined]",
                "top_p": "[undefined]",
                "max_output_tokens": "[undefined]",
                "tools": "[undefined]",
                "tool_choice": "[undefined]",
                "store": False,
                "stream": True,
            }
        )

        self.assertNotIn("temperature", normalized)
        self.assertNotIn("top_p", normalized)
        self.assertNotIn("max_output_tokens", normalized)
        self.assertNotIn("tools", normalized)
        self.assertNotIn("tool_choice", normalized)

        assistant_item = normalized["input"][1]
        self.assertEqual(assistant_item["role"], "assistant")
        self.assertEqual(assistant_item["type"], "message")
        self.assertEqual(assistant_item["status"], "completed")
        self.assertTrue(assistant_item["id"].startswith("msg_"))
        self.assertEqual(assistant_item["content"][0]["annotations"], [])


if __name__ == "__main__":
    unittest.main()
