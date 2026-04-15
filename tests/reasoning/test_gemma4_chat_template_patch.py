#!/usr/bin/env python3
"""Unit tests for the local Gemma4 chat template patch."""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import textwrap
import unittest

import jinja2


ROOT = pathlib.Path(__file__).resolve().parents[2]
PATCH_SCRIPT = ROOT / "scripts" / "patch-modelscope-gemma4-chat-template.py"


def load_patch_module():
    spec = importlib.util.spec_from_file_location("gemma4_template_patch", PATCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load patch module from {PATCH_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = load_patch_module()


TEMPLATE_FIXTURE = textwrap.dedent(
    """\
    {%- set ns = namespace(prev_message_type=None) -%}
    {{- bos_token -}}
    {%- if (enable_thinking is defined and enable_thinking) or tools or messages[0]['role'] in ['system', 'developer'] -%}
        {{- '<|turn>system\\n' -}}
        {%- if enable_thinking is defined and enable_thinking -%}
            {{- '<|think|>\\n' -}}
        {%- endif -%}
        {{- '<turn|>\\n' -}}
    {%- endif %}
    {%- if add_generation_prompt -%}
        {%- if not enable_thinking | default(false) and ns.prev_message_type != 'tool_call' -%}
            {{- '<|channel>thought\\n<channel|>' -}}
        {%- endif -%}
    {%- endif -%}
    """
)

LEGACY_PATCHED_TEMPLATE_FIXTURE = textwrap.dedent(
    """\
    {%- set ns = namespace(prev_message_type=None) -%}
    {%- set ns_request = namespace(enable_thinking=false) -%}
    {%- if enable_thinking is defined -%}
        {%- set ns_request.enable_thinking = enable_thinking -%}
    {%- elif reasoning_effort is defined and reasoning_effort in ['medium', 'high'] -%}
        {%- set ns_request.enable_thinking = true -%}
    {%- endif -%}
    {{- bos_token -}}
    {%- if ns_request.enable_thinking or tools or messages[0]['role'] in ['system', 'developer'] -%}
        {{- '<|turn>system\\n' -}}
        {%- if ns_request.enable_thinking -%}
            {{- '<|think|>\\n' -}}
            {%- if not tools -%}
                {{- 'Keep the reasoning extremely brief, at most 2 short lines. Do not repeat calculations. Then end with a final line formatted exactly as Final Answer: <answer>.\\n' -}}
            {%- endif -%}
        {%- endif -%}
        {{- '<turn|>\\n' -}}
    {%- endif %}
    {%- if add_generation_prompt -%}
        {%- if not ns_request.enable_thinking and ns.prev_message_type != 'tool_call' -%}
            {{- '<|channel>thought\\n<channel|>' -}}
        {%- endif -%}
    {%- endif -%}
    """
)

LEGACY_DUPLICATED_TEMPLATE_FIXTURE = textwrap.dedent(
    """\
    {%- set ns = namespace(prev_message_type=None) -%}
    {%- set ns_request = namespace(enable_thinking=false, reasoning_effort='none') -%}
    {%- if reasoning_effort is defined and reasoning_effort in ['none', 'low', 'medium', 'high'] -%}
        {%- set ns_request.reasoning_effort = reasoning_effort -%}
    {%- endif -%}
    {%- if enable_thinking is defined -%}
        {%- set ns_request.enable_thinking = enable_thinking -%}
        {%- if ns_request.enable_thinking and ns_request.reasoning_effort == 'none' -%}
            {%- set ns_request.reasoning_effort = 'medium' -%}
        {%- endif -%}
    {%- elif ns_request.reasoning_effort in ['low', 'medium', 'high'] -%}
        {%- set ns_request.enable_thinking = true -%}
    {%- endif -%}
    {%- set ns_request = namespace(enable_thinking=false) -%}
    {%- if enable_thinking is defined -%}
        {%- set ns_request.enable_thinking = enable_thinking -%}
    {%- elif reasoning_effort is defined and reasoning_effort in ['medium', 'high'] -%}
        {%- set ns_request.enable_thinking = true -%}
    {%- endif -%}
    {{- bos_token -}}
    {%- if ns_request.enable_thinking or tools or messages[0]['role'] in ['system', 'developer'] -%}
        {{- '<|turn>system\\n' -}}
        {%- if ns_request.enable_thinking -%}
            {{- '<|think|>\\n' -}}
            {%- if not tools -%}
                {{- 'Keep the reasoning extremely brief, at most 2 short lines. Do not repeat calculations. Then end with a final line formatted exactly as Final Answer: <answer>.\\n' -}}
            {%- endif -%}
        {%- endif -%}
        {{- '<turn|>\\n' -}}
    {%- endif %}
    """
)


def render_template(source: str, **kwargs: object) -> str:
    environment = jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.from_string(source)
    render_kwargs = {
        "bos_token": "<bos>",
        "messages": [{"role": "user", "content": "What is 17 * 19?"}],
        "tools": None,
        "add_generation_prompt": True,
    }
    render_kwargs.update(kwargs)
    return template.render(
        **render_kwargs,
    )


class Gemma4ChatTemplatePatchTests(unittest.TestCase):
    def patch_fixture(self) -> tuple[pathlib.Path, str]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        template_path = pathlib.Path(temp_dir.name) / "chat_template.jinja"
        template_path.write_text(TEMPLATE_FIXTURE, encoding="utf-8")

        changed = patch.patch_template(template_path)
        self.assertTrue(changed)
        return template_path, template_path.read_text(encoding="utf-8")

    def test_patch_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = pathlib.Path(temp_dir) / "chat_template.jinja"
            template_path.write_text(TEMPLATE_FIXTURE, encoding="utf-8")
            self.assertTrue(patch.patch_template(template_path))
            self.assertFalse(patch.patch_template(template_path))

    def test_patch_upgrades_legacy_patched_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = pathlib.Path(temp_dir) / "chat_template.jinja"
            template_path.write_text(LEGACY_PATCHED_TEMPLATE_FIXTURE, encoding="utf-8")
            self.assertTrue(patch.patch_template(template_path))
            upgraded = template_path.read_text(encoding="utf-8")
            self.assertIn("reasoning_effort='none'", upgraded)
            self.assertIn("Keep the reasoning brief.", upgraded)
            self.assertNotIn("at most 2 short lines", upgraded)
            self.assertEqual(upgraded.count("ns_request = namespace"), 1)

    def test_patch_removes_legacy_duplicate_ns_block(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = pathlib.Path(temp_dir) / "chat_template.jinja"
            template_path.write_text(LEGACY_DUPLICATED_TEMPLATE_FIXTURE, encoding="utf-8")
            self.assertTrue(patch.patch_template(template_path))
            upgraded = template_path.read_text(encoding="utf-8")
            self.assertEqual(upgraded.count("ns_request = namespace"), 1)

    def test_reasoning_effort_medium_enables_thinking(self) -> None:
        _, patched_source = self.patch_fixture()
        rendered = render_template(patched_source, reasoning_effort="medium")
        self.assertIn("<|think|>", rendered)
        self.assertIn("Final Answer: <answer>", rendered)
        self.assertIn("Keep the reasoning brief.", rendered)
        self.assertNotIn("<|channel>thought\n<channel|>", rendered)

    def test_reasoning_effort_low_stays_non_thinking(self) -> None:
        _, patched_source = self.patch_fixture()
        rendered = render_template(patched_source, reasoning_effort="low")
        self.assertNotIn("<|think|>", rendered)
        self.assertNotIn("Final Answer: <answer>", rendered)
        self.assertIn("<|channel>thought\n<channel|>", rendered)

    def test_reasoning_effort_none_stays_non_thinking(self) -> None:
        _, patched_source = self.patch_fixture()
        rendered = render_template(patched_source, reasoning_effort="none")
        self.assertNotIn("<|think|>", rendered)
        self.assertNotIn("Final Answer: <answer>", rendered)
        self.assertIn("<|channel>thought\n<channel|>", rendered)

    def test_reasoning_effort_high_enables_thinking(self) -> None:
        _, patched_source = self.patch_fixture()
        rendered = render_template(patched_source, reasoning_effort="high")
        self.assertIn("<|think|>", rendered)
        self.assertIn("Keep the reasoning brief.", rendered)
        self.assertIn("Final Answer: <answer>", rendered)

    def test_explicit_enable_thinking_takes_priority(self) -> None:
        _, patched_source = self.patch_fixture()
        rendered = render_template(
            patched_source,
            enable_thinking=True,
            reasoning_effort="none",
        )
        self.assertIn("<|think|>", rendered)
        self.assertIn("Keep the reasoning brief.", rendered)
        self.assertNotIn("<|channel>thought\n<channel|>", rendered)

    def test_hint_is_skipped_when_tools_are_present(self) -> None:
        _, patched_source = self.patch_fixture()
        rendered = render_template(
            patched_source,
            reasoning_effort="medium",
            tools=[{"type": "function"}],
        )
        self.assertIn("<|think|>", rendered)
        self.assertNotIn("Final Answer: <answer>", rendered)


if __name__ == "__main__":
    unittest.main()
