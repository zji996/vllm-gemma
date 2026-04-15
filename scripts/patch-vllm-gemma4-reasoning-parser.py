#!/usr/bin/env python3
"""
Patch vLLM's Gemma4 reasoning parser so chat completions can split
Gemma thinking text out of `message.content` more robustly.

Why this patch is needed on vLLM v0.19.0:

- In non-streaming chat completions, `output.text` is often decoded with
  `skip_special_tokens=True`.
- Gemma4's opening `<|channel>` token may therefore be absent from the
  string passed into `Gemma4ReasoningParser.extract_reasoning()`.
- The stock parser then falls back to treating the whole payload as
  `content`, which leaks `thought\\n...` into the final answer.

This patch teaches the parser to handle the three real shapes we have
observed locally:

1. `thought\\n...`
   -> reasoning only, no final content yet
2. `thought\\n<channel|>579`
   -> empty reasoning, final content `579`
3. `<|channel>thought\\n...<channel|>579`
   -> reasoning + final content

An optional heuristic can also salvage explicit tail markers like
`Final Answer:` when the model omits `<channel|>`. It is disabled by
default and only activates when `SERVE_GEMMA4_HEURISTIC_FINAL_ANSWER=true`.
"""

from __future__ import annotations

import os
from pathlib import Path


VLLM_ROOT_CANDIDATES = [
    Path("/opt/vllm-src/vllm"),
    Path("/usr/local/lib/python3.12/dist-packages/vllm"),
]
TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
HEURISTIC_FINAL_ANSWER_MARKERS = (
    "final answer:",
    "final answer：",
    "最终答案:",
    "最终答案：",
)


def resolve_vllm_root() -> Path:
    for root in VLLM_ROOT_CANDIDATES:
        if (root / "reasoning/gemma4_reasoning_parser.py").exists():
            return root
    raise FileNotFoundError(
        "Could not find vLLM install root. Checked: "
        + ", ".join(str(p) for p in VLLM_ROOT_CANDIDATES)
    )


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Failed to find expected anchor for {label}")
    return text.replace(old, new, 1)


def is_truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUTHY_ENV_VALUES


def heuristic_final_answer_enabled() -> bool:
    return is_truthy_env(os.environ.get("SERVE_GEMMA4_HEURISTIC_FINAL_ANSWER"))


def request_expects_reasoning_for_tests(
    *,
    enable_thinking: bool = False,
    reasoning_effort: str | None = None,
    reasoning_cfg_effort: str | None = None,
) -> bool:
    if enable_thinking:
        return True
    if reasoning_effort in ("medium", "high"):
        return True
    return reasoning_cfg_effort in ("medium", "high")


def strip_thought_label_for_tests(text: str) -> str:
    if text.startswith("thought\n"):
        return text[len("thought\n") :]
    return text


def clean_content_for_tests(text: str | None) -> str | None:
    if text is None:
        return None

    cleaned = text.strip()
    while True:
        updated = cleaned
        if updated.endswith("<turn|>"):
            updated = updated[: -len("<turn|>")].rstrip()
        if updated.endswith("<eos>"):
            updated = updated[:-5].rstrip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned or None


def normalize_heuristic_line(text: str) -> str:
    candidate = text.strip()
    candidate = candidate.lstrip("-*#> ").strip()
    candidate = candidate.replace("**", "").replace("__", "")
    return candidate.strip()


def looks_like_final_answer_candidate(text: str | None) -> bool:
    if not text:
        return False

    candidate = text.strip()
    if not candidate:
        return False
    if len(candidate) > 4000:
        return False
    if candidate.count("\n") > 3:
        return False
    for marker in ("thought\n", "<|channel>", "<channel|>", "<turn|>"):
        if marker in candidate:
            return False
    return True


def extract_heuristic_final_answer(reasoning: str | None) -> tuple[str | None, str | None]:
    if reasoning is None:
        return None, None

    cleaned_reasoning = reasoning.strip()
    if not cleaned_reasoning:
        return None, None

    lines = [line.rstrip() for line in cleaned_reasoning.splitlines()]
    scan_start = max(0, len(lines) - 4)
    for line_idx in range(scan_start, len(lines)):
        trailing_lines = [line for line in lines[line_idx:] if line.strip()]
        if not trailing_lines:
            continue

        first_line = normalize_heuristic_line(trailing_lines[0])
        for marker in HEURISTIC_FINAL_ANSWER_MARKERS:
            source = first_line.lower() if marker.isascii() else first_line
            target = marker.lower() if marker.isascii() else marker
            if not source.startswith(target):
                continue

            answer = first_line[len(marker) :].strip()
            if not answer and len(trailing_lines) > 1:
                answer = "\n".join(
                    normalize_heuristic_line(line)
                    for line in trailing_lines[1:]
                    if line.strip()
                ).strip()

            if not looks_like_final_answer_candidate(answer):
                continue

            head = "\n".join(lines[:line_idx]).strip() or None
            return head, answer

    return cleaned_reasoning, None


def finalize_reasoning_extraction_for_tests(
    reasoning: str | None,
    content: str | None,
    *,
    heuristic_enabled: bool = False,
) -> tuple[str | None, str | None]:
    cleaned_reasoning = None
    if reasoning is not None:
        cleaned_reasoning = strip_thought_label_for_tests(reasoning).strip() or None

    cleaned_content = clean_content_for_tests(content)
    if heuristic_enabled and cleaned_content is None and cleaned_reasoning is not None:
        cleaned_reasoning, cleaned_content = extract_heuristic_final_answer(
            cleaned_reasoning
        )

    return cleaned_reasoning, cleaned_content


def simulate_patched_extract_reasoning(
    model_output: str,
    *,
    expects_reasoning: bool = False,
    heuristic_enabled: bool = False,
    full_start_token: str = "<|channel>thought\n",
    end_token: str = "<channel|>",
) -> tuple[str | None, str | None]:
    if full_start_token in model_output:
        reasoning_block, _, remainder = model_output.partition(full_start_token)
        del reasoning_block
        reasoning_text, has_end_token, content = remainder.partition(end_token)
        return finalize_reasoning_extraction_for_tests(
            reasoning_text,
            content if has_end_token else None,
            heuristic_enabled=heuristic_enabled,
        )

    if end_token in model_output:
        reasoning_block, _, content = model_output.partition(end_token)
        return finalize_reasoning_extraction_for_tests(
            reasoning_block,
            content,
            heuristic_enabled=heuristic_enabled,
        )

    if expects_reasoning:
        stripped = strip_thought_label_for_tests(model_output)
        if stripped != model_output:
            return finalize_reasoning_extraction_for_tests(
                stripped,
                None,
                heuristic_enabled=heuristic_enabled,
            )

    return None, model_output


OLD_EXTRACT_REASONING = '''    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Extract reasoning, stripping the ``thought\\\\n`` role label."""
        if self.start_token not in model_output and self.end_token not in model_output:
            # Default to content history if no tags are present
            # (or if they were stripped)
            return None, model_output

        reasoning, content = super().extract_reasoning(model_output, request)
        if reasoning is not None:
            reasoning = _strip_thought_label(reasoning)
        return reasoning, content
'''


NEW_EXTRACT_REASONING = '''    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Extract reasoning from Gemma4 output, tolerating stripped start tags.

        In vLLM's non-streaming chat path, `output.text` is commonly decoded with
        `skip_special_tokens=True`, so the opening `<|channel>` marker may be
        missing even though the model did produce a thinking block.

        Gemma4 still leaves behind one of these stable textual shapes:

        - `thought\\n...`
        - `thought\\n<channel|>final answer`
        - `<|channel>thought\\n...<channel|>final answer`

        We normalize all three into `(reasoning, content)` so `message.content`
        only contains the final answer.
        """
        if self.start_token in model_output:
            reasoning, content = super().extract_reasoning(model_output, request)
            return _finalize_reasoning_extraction(reasoning, content)

        if self.end_token in model_output:
            reasoning_block, _, content = model_output.partition(self.end_token)
            return _finalize_reasoning_extraction(reasoning_block, content)

        if _request_expects_reasoning(request):
            stripped = _strip_thought_label(model_output)
            if stripped != model_output:
                reasoning = stripped.strip()
                return _finalize_reasoning_extraction(reasoning or None, None)

        return None, model_output
'''


OLD_STRIP_HELPER = '''def _strip_thought_label(text: str) -> str:
    """Remove the ``thought\\\\n`` role label from the beginning of text.

    Mirrors ``vllm.reasoning.gemma4_utils._strip_thought_label`` from the
    offline parser.
    """
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text
'''


NEW_STRIP_HELPER = '''def _strip_thought_label(text: str) -> str:
    """Remove the ``thought\\\\n`` role label from the beginning of text.

    Mirrors ``vllm.reasoning.gemma4_utils._strip_thought_label`` from the
    offline parser.
    """
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text


def _request_expects_reasoning(request: "ChatCompletionRequest | ResponsesRequest") -> bool:
    chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
    if chat_template_kwargs.get("enable_thinking"):
        return True

    reasoning_effort = getattr(request, "reasoning_effort", None)
    if reasoning_effort in ("low", "medium", "high"):
        return True

    reasoning_cfg = getattr(request, "reasoning", None)
    reasoning_cfg_effort = getattr(reasoning_cfg, "effort", None)
    return reasoning_cfg_effort in ("low", "medium", "high")


def _clean_content(text: str | None) -> str | None:
    if text is None:
        return None

    cleaned = text.strip()
    while True:
        updated = cleaned
        if updated.endswith("<turn|>"):
            updated = updated[: -len("<turn|>")].rstrip()
        if updated.endswith("<eos>"):
            updated = updated[:-5].rstrip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned or None


def _heuristic_final_answer_enabled() -> bool:
    value = __import__("os").environ.get("SERVE_GEMMA4_HEURISTIC_FINAL_ANSWER", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_heuristic_line(text: str) -> str:
    candidate = text.strip()
    candidate = candidate.lstrip("-*#> ").strip()
    candidate = candidate.replace("**", "").replace("__", "")
    return candidate.strip()


def _looks_like_final_answer_candidate(text: str | None) -> bool:
    if not text:
        return False

    candidate = text.strip()
    if not candidate:
        return False
    if len(candidate) > 4000:
        return False
    if candidate.count("\\n") > 3:
        return False
    for marker in ("thought\\n", "<|channel>", "<channel|>", "<turn|>"):
        if marker in candidate:
            return False
    return True


def _maybe_extract_heuristic_final_answer(
    reasoning: str | None,
) -> tuple[str | None, str | None]:
    if reasoning is None:
        return None, None

    cleaned_reasoning = reasoning.strip()
    if not cleaned_reasoning:
        return None, None

    lines = [line.rstrip() for line in cleaned_reasoning.splitlines()]
    markers = (
        "final answer:",
        "final answer：",
        "最终答案:",
        "最终答案：",
    )
    scan_start = max(0, len(lines) - 4)
    for line_idx in range(scan_start, len(lines)):
        trailing_lines = [line for line in lines[line_idx:] if line.strip()]
        if not trailing_lines:
            continue

        first_line = _normalize_heuristic_line(trailing_lines[0])
        for marker in markers:
            source = first_line.lower() if marker.isascii() else first_line
            target = marker.lower() if marker.isascii() else marker
            if not source.startswith(target):
                continue

            answer = first_line[len(marker) :].strip()
            if not answer and len(trailing_lines) > 1:
                answer = "\\n".join(
                    _normalize_heuristic_line(line)
                    for line in trailing_lines[1:]
                    if line.strip()
                ).strip()

            if not _looks_like_final_answer_candidate(answer):
                continue

            head = "\\n".join(lines[:line_idx]).strip() or None
            return head, answer

    return cleaned_reasoning, None


def _finalize_reasoning_extraction(
    reasoning: str | None, content: str | None
) -> tuple[str | None, str | None]:
    cleaned_reasoning = None
    if reasoning is not None:
        cleaned_reasoning = _strip_thought_label(reasoning).strip() or None

    cleaned_content = _clean_content(content)
    if (
        _heuristic_final_answer_enabled()
        and cleaned_content is None
        and cleaned_reasoning is not None
    ):
        cleaned_reasoning, cleaned_content = _maybe_extract_heuristic_final_answer(
            cleaned_reasoning
        )

    return cleaned_reasoning, cleaned_content
'''


def main() -> int:
    reasoning_parser_path = resolve_vllm_root() / "reasoning/gemma4_reasoning_parser.py"

    text = reasoning_parser_path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        OLD_EXTRACT_REASONING,
        NEW_EXTRACT_REASONING,
        "gemma4 extract_reasoning",
    )
    text = replace_once(
        text,
        OLD_STRIP_HELPER,
        NEW_STRIP_HELPER,
        "gemma4 strip helper",
    )

    reasoning_parser_path.write_text(text, encoding="utf-8")
    print(f"Patched {reasoning_parser_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
