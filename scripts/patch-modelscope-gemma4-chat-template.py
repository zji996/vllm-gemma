#!/usr/bin/env python3
"""
Patch a local Gemma 4 chat template so vLLM's OpenAI-compatible
`reasoning_effort` request field can drive thinking mode.

Behavior after patch:
  - `enable_thinking=true` still has highest priority.
  - `reasoning_effort=medium|high` enables `<|think|>`.
  - `reasoning_effort=none|low` keeps non-thinking mode.

This patch is idempotent and only touches `chat_template.jinja`.
"""

from __future__ import annotations

from pathlib import Path
import sys


PATCH_HEADER = """{%- set ns = namespace(prev_message_type=None) -%}
{%- set ns_request = namespace(enable_thinking=false) -%}
{%- if enable_thinking is defined -%}
    {%- set ns_request.enable_thinking = enable_thinking -%}
{%- elif reasoning_effort is defined and reasoning_effort in ['medium', 'high'] -%}
    {%- set ns_request.enable_thinking = true -%}
{%- endif -%}
"""
THINKING_HINT_BLOCK = """        {%- if not tools -%}
            {{- 'Keep the reasoning extremely brief, at most 2 short lines. Do not repeat calculations. Then end with a final line formatted exactly as Final Answer: <answer>.\\n' -}}
        {%- endif -%}
"""
OLD_THINKING_HINT = (
    "Keep reasoning brief. When you are ready to answer directly, end with a "
    "final line formatted exactly as Final Answer: <answer>."
)
NEW_THINKING_HINT = (
    "Keep the reasoning extremely brief, at most 2 short lines. Do not repeat "
    "calculations. Then end with a final line formatted exactly as Final Answer: "
    "<answer>."
)


def patch_template(template_path: Path) -> bool:
    text = template_path.read_text(encoding="utf-8")
    changed = False

    if "ns_request = namespace(enable_thinking=false)" not in text:
        original_header = "{%- set ns = namespace(prev_message_type=None) -%}\n"
        if original_header not in text:
            raise RuntimeError("Could not find template namespace header to patch.")

        text = text.replace(original_header, PATCH_HEADER, 1)
        text = text.replace(
            "{%- if (enable_thinking is defined and enable_thinking) or tools or messages[0]['role'] in ['system', 'developer'] -%}",
            "{%- if ns_request.enable_thinking or tools or messages[0]['role'] in ['system', 'developer'] -%}",
            1,
        )
        text = text.replace(
            "{%- if enable_thinking is defined and enable_thinking -%}",
            "{%- if ns_request.enable_thinking -%}",
            1,
        )
        text = text.replace(
            "{%- if not enable_thinking | default(false) and ns.prev_message_type != 'tool_call' -%}",
            "{%- if not ns_request.enable_thinking and ns.prev_message_type != 'tool_call' -%}",
            1,
        )
        changed = True

    if OLD_THINKING_HINT in text and NEW_THINKING_HINT not in text:
        text = text.replace(OLD_THINKING_HINT, NEW_THINKING_HINT, 1)
        changed = True

    if "Final Answer: <answer>" not in text:
        thinking_anchor = "        {{- '<|think|>\\n' -}}\n"
        if thinking_anchor not in text:
            raise RuntimeError("Could not find thinking anchor to patch.")
        text = text.replace(thinking_anchor, thinking_anchor + THINKING_HINT_BLOCK, 1)
        changed = True

    if changed:
        template_path.write_text(text, encoding="utf-8")
    return changed


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: patch-modelscope-gemma4-chat-template.py <model-dir>",
            file=sys.stderr,
        )
        return 2

    model_dir = Path(sys.argv[1]).expanduser().resolve()
    template_path = model_dir / "chat_template.jinja"
    if not template_path.is_file():
        print(f"chat_template.jinja not found: {template_path}", file=sys.stderr)
        return 1

    changed = patch_template(template_path)
    status = "patched" if changed else "already-patched"
    print(f"{status}: {template_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
