#!/usr/bin/env python3
"""
Patch vLLM's OpenAI chat stack to apply harder reasoning-budget controls.

This patch does two things for chat completions:

1. Map `reasoning_effort=low|medium|high` to default `thinking_token_budget`
   values when the request did not specify one explicitly.
2. Reuse vLLM's built-in thinking-budget logits processor to also force the
   reasoning section to end early when the generated thought starts repeating
   the same token pattern several times.

The goal is to avoid pathological Gemma4 outputs that keep looping inside the
thinking channel until `max_tokens` is exhausted.
"""

from __future__ import annotations

from pathlib import Path


VLLM_ROOT_CANDIDATES = [
    Path("/opt/vllm-src/vllm"),
    Path("/usr/local/lib/python3.12/dist-packages/vllm"),
]


def resolve_vllm_root() -> Path:
    for root in VLLM_ROOT_CANDIDATES:
        if (root / "entrypoints/openai/chat_completion/protocol.py").exists():
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


def reasoning_budget_defaults_for_tests(
    reasoning_effort: str | None,
) -> tuple[int | None, dict[str, int] | None]:
    budgets = {
        "low": 32,
        "medium": 128,
        "high": 512,
    }
    budget = budgets.get(reasoning_effort or "")
    repetition_cfg = {
        "max_pattern_size": 64,
        "min_pattern_size": 4,
        "min_count": 3,
    }
    if budget is None:
        return None, None
    return budget, repetition_cfg


def apply_reasoning_defaults_for_tests(
    reasoning_effort: str | None,
    *,
    thinking_token_budget: int | None = None,
    extra_args: dict | None = None,
) -> tuple[int | None, dict]:
    resolved_extra_args = dict(extra_args or {})
    if reasoning_effort not in ("low", "medium", "high"):
        return thinking_token_budget, resolved_extra_args

    default_budget, repetition_cfg = reasoning_budget_defaults_for_tests(
        reasoning_effort
    )
    if thinking_token_budget is None:
        thinking_token_budget = default_budget
    if repetition_cfg is not None and "reasoning_repetition_detection" not in resolved_extra_args:
        resolved_extra_args["reasoning_repetition_detection"] = repetition_cfg
    return thinking_token_budget, resolved_extra_args


def repeated_pattern_detected_for_tests(
    token_ids: list[int],
    repetition_cfg: dict[str, int] | None,
) -> bool:
    if not repetition_cfg:
        return False

    max_pattern_size = repetition_cfg.get("max_pattern_size", 0)
    min_pattern_size = repetition_cfg.get("min_pattern_size", 0) or 1
    min_count = repetition_cfg.get("min_count", 0)

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False

    for pattern_len in range(min_pattern_size, max_pattern_size + 1):
        if pattern_len * min_count > len(token_ids):
            return False
        pattern = token_ids[-pattern_len:]
        if pattern * min_count == token_ids[-(pattern_len * min_count) :]:
            return True
    return False


def repeated_pattern_detected_while_thinking_for_tests(
    token_ids: list[int],
    repetition_cfg: dict[str, int] | None,
    *,
    in_think: bool,
) -> bool:
    if not in_think:
        return False
    return repeated_pattern_detected_for_tests(token_ids, repetition_cfg)


OLD_PROTOCOL_IMPORT = """import json
import time
from typing import Annotated, Any, ClassVar, Literal
"""


NEW_PROTOCOL_IMPORT = """import json
import os
import time
from typing import Annotated, Any, ClassVar, Literal
"""


OLD_PROTOCOL_HELPER_ANCHOR = """_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class ChatMessage(OpenAIBaseModel):
"""


NEW_PROTOCOL_HELPER_BLOCK = """_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _read_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _default_reasoning_token_budget(reasoning_effort: str | None) -> int | None:
    budgets = {
        "low": _read_int_env("SERVE_REASONING_BUDGET_LOW", 32),
        "medium": _read_int_env("SERVE_REASONING_BUDGET_MEDIUM", 128),
        "high": _read_int_env("SERVE_REASONING_BUDGET_HIGH", 512),
    }
    budget = budgets.get(reasoning_effort or "")
    if budget is None or budget <= 0:
        return None
    return budget


def _default_reasoning_repetition_detection() -> dict[str, int] | None:
    max_pattern_size = _read_int_env("SERVE_REASONING_REPEAT_MAX_PATTERN", 64)
    min_pattern_size = _read_int_env("SERVE_REASONING_REPEAT_MIN_PATTERN", 4)
    min_count = _read_int_env("SERVE_REASONING_REPEAT_MIN_COUNT", 3)
    if max_pattern_size <= 0 or min_count < 2:
        return None
    if min_pattern_size <= 0:
        min_pattern_size = 1
    if min_pattern_size > max_pattern_size:
        min_pattern_size = max_pattern_size
    return {
        "max_pattern_size": max_pattern_size,
        "min_pattern_size": min_pattern_size,
        "min_count": min_count,
    }


def _apply_reasoning_sampling_defaults(
    request: "ChatCompletionRequest", extra_args: dict[str, Any]
) -> tuple[int | None, dict[str, Any]]:
    thinking_token_budget = request.thinking_token_budget
    reasoning_effort = getattr(request, "reasoning_effort", None)
    if reasoning_effort not in ("low", "medium", "high"):
        return thinking_token_budget, extra_args

    if thinking_token_budget is None:
        thinking_token_budget = _default_reasoning_token_budget(reasoning_effort)
    if "reasoning_repetition_detection" not in extra_args:
        repetition_cfg = _default_reasoning_repetition_detection()
        if repetition_cfg is not None:
            extra_args["reasoning_repetition_detection"] = repetition_cfg
    return thinking_token_budget, extra_args


class ChatMessage(OpenAIBaseModel):
"""


OLD_PROTOCOL_SAMPLING_BLOCK = """        extra_args: dict[str, Any] = self.vllm_xargs if self.vllm_xargs else {}
        if self.kv_transfer_params:
            # Pass in kv_transfer_params via extra_args
            extra_args["kv_transfer_params"] = self.kv_transfer_params
        return SamplingParams.from_optional(
"""


NEW_PROTOCOL_SAMPLING_BLOCK = """        extra_args: dict[str, Any] = dict(self.vllm_xargs) if self.vllm_xargs else {}
        thinking_token_budget, extra_args = _apply_reasoning_sampling_defaults(
            self, extra_args
        )
        if self.kv_transfer_params:
            # Pass in kv_transfer_params via extra_args
            extra_args["kv_transfer_params"] = self.kv_transfer_params
        return SamplingParams.from_optional(
"""


OLD_PROTOCOL_THINKING_FIELD = """            bad_words=self.bad_words,
            thinking_token_budget=self.thinking_token_budget,
            allowed_token_ids=self.allowed_token_ids,
            extra_args=extra_args or None,
"""


NEW_PROTOCOL_THINKING_FIELD = """            bad_words=self.bad_words,
            thinking_token_budget=thinking_token_budget,
            allowed_token_ids=self.allowed_token_ids,
            extra_args=extra_args or None,
"""


OLD_BUILTIN_IMPORT = """from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
"""


NEW_BUILTIN_IMPORT = """from vllm import SamplingParams
from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.core.sched.utils import check_sequence_repetition
from vllm.v1.sample.logits_processor.interface import (
"""


OLD_INIT_STATE_ENTRY = """    def _init_state_entry(
        self, prompt_tok_ids: list[int] | None, thinking_token_budget: int
    ) -> dict[str, Any]:
        \"\"\"Initializes the tracking state for a given sequence index.\"\"\"
"""


NEW_INIT_STATE_ENTRY = """    def _init_state_entry(
        self,
        prompt_tok_ids: list[int] | None,
        thinking_token_budget: int,
        reasoning_repetition_detection: RepetitionDetectionParams | None,
    ) -> dict[str, Any]:
        \"\"\"Initializes the tracking state for a given sequence index.\"\"\"
"""


OLD_INIT_STATE_RETURN = """        return {
            "in_think": in_think,  # Currently in thinking mode
            "in_end": in_think and thinking_token_budget == 0,
            "check_count_down": thinking_token_budget,
            "think_count": think_count,  # Number of tokens in thinking section
            "end_count": 0,  # Number of end tokens forced so far
            "prompt_tok_ids": prompt_tok_ids,
            "output_tok_ids": [],
            "thinking_token_budget": thinking_token_budget,
            "prev_output_length": 0,
            # Track previous output length for incremental updates
        }
"""


NEW_INIT_STATE_RETURN = """        return {
            "in_think": in_think,  # Currently in thinking mode
            "in_end": in_think and thinking_token_budget == 0,
            "check_count_down": thinking_token_budget,
            "think_count": think_count,  # Number of tokens in thinking section
            "end_count": 0,  # Number of end tokens forced so far
            "prompt_tok_ids": prompt_tok_ids,
            "output_tok_ids": [],
            "thinking_token_budget": thinking_token_budget,
            "reasoning_repetition_detection": reasoning_repetition_detection,
            "prev_output_length": 0,
            # Track previous output length for incremental updates
        }
"""


OLD_UPDATE_BUDGET_CHECK = """            # Check if need to transition to end mode
            if (
                state["in_think"]
                and state["think_count"] >= state["thinking_token_budget"]
            ):
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0
                state["check_count_down"] = state["thinking_token_budget"]
        else:
"""


NEW_UPDATE_BUDGET_CHECK = """            if self._maybe_force_end_for_repetition(state):
                return

            # Check if need to transition to end mode
            if (
                state["in_think"]
                and state["think_count"] >= state["thinking_token_budget"]
            ):
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0
                state["check_count_down"] = state["thinking_token_budget"]
        else:
"""


OLD_IS_ARGMAX = """    def is_argmax_invariant(self) -> bool:
        \"\"\"This logits processor can change the outcome of
        greedy sampling by forcing that the thinking section
        ends after a certain number of tokens.\"\"\"
        return False
"""


NEW_IS_ARGMAX = """    def _maybe_force_end_for_repetition(self, state: dict[str, Any]) -> bool:
        repetition_detection = state.get("reasoning_repetition_detection")
        if not state.get("in_think") or repetition_detection is None:
            return False

        output_tok_ids = state.get("output_tok_ids", [])
        if not output_tok_ids:
            return False

        if not check_sequence_repetition(output_tok_ids, repetition_detection):
            return False

        state["in_think"] = False
        state["in_end"] = True
        state["end_count"] = 0
        state["check_count_down"] = state["thinking_token_budget"]
        return True

    def is_argmax_invariant(self) -> bool:
        \"\"\"This logits processor can change the outcome of
        greedy sampling by forcing that the thinking section
        ends after a certain number of tokens.\"\"\"
        return False
"""


OLD_UPDATE_STATE_ADDED = """            for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
                thinking_token_budget = params.thinking_token_budget

                if thinking_token_budget is not None:
                    self._state[index] = self._init_state_entry(
                        prompt_tok_ids, thinking_token_budget
                    )
                    self._state[index]["output_tok_ids"] = output_tok_ids
                else:
                    # Remove state if no thinking budget
                    self._state.pop(index, None)
"""


NEW_UPDATE_STATE_ADDED = """            for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
                thinking_token_budget = params.thinking_token_budget
                reasoning_repetition_detection = None
                extra_args = params.extra_args or {}
                repetition_config = (
                    extra_args.get("reasoning_repetition_detection")
                    if isinstance(extra_args, dict)
                    else None
                )
                if isinstance(repetition_config, dict):
                    try:
                        reasoning_repetition_detection = RepetitionDetectionParams(
                            max_pattern_size=int(
                                repetition_config.get("max_pattern_size", 0)
                            ),
                            min_pattern_size=int(
                                repetition_config.get("min_pattern_size", 0)
                            ),
                            min_count=int(repetition_config.get("min_count", 0)),
                        )
                    except ValueError:
                        reasoning_repetition_detection = None

                if thinking_token_budget is not None:
                    self._state[index] = self._init_state_entry(
                        prompt_tok_ids,
                        thinking_token_budget,
                        reasoning_repetition_detection,
                    )
                    self._state[index]["output_tok_ids"] = output_tok_ids
                else:
                    # Remove state if no thinking budget
                    self._state.pop(index, None)
"""


def patch_protocol(chat_protocol_path: Path) -> None:
    text = chat_protocol_path.read_text(encoding="utf-8")
    text = replace_once(
        text, OLD_PROTOCOL_IMPORT, NEW_PROTOCOL_IMPORT, "chat protocol import"
    )
    text = replace_once(
        text,
        OLD_PROTOCOL_HELPER_ANCHOR,
        NEW_PROTOCOL_HELPER_BLOCK,
        "chat protocol helper block",
    )
    text = replace_once(
        text,
        OLD_PROTOCOL_SAMPLING_BLOCK,
        NEW_PROTOCOL_SAMPLING_BLOCK,
        "chat protocol sampling block",
    )
    text = replace_once(
        text,
        OLD_PROTOCOL_THINKING_FIELD,
        NEW_PROTOCOL_THINKING_FIELD,
        "chat protocol thinking field",
    )
    chat_protocol_path.write_text(text, encoding="utf-8")


def patch_builtin_processor(builtin_path: Path) -> None:
    text = builtin_path.read_text(encoding="utf-8")
    text = replace_once(
        text, OLD_BUILTIN_IMPORT, NEW_BUILTIN_IMPORT, "builtin import block"
    )
    text = replace_once(
        text, OLD_INIT_STATE_ENTRY, NEW_INIT_STATE_ENTRY, "thinking init signature"
    )
    text = replace_once(
        text, OLD_INIT_STATE_RETURN, NEW_INIT_STATE_RETURN, "thinking init state"
    )
    text = replace_once(
        text,
        OLD_UPDATE_BUDGET_CHECK,
        NEW_UPDATE_BUDGET_CHECK,
        "thinking repetition force-end hook",
    )
    text = replace_once(
        text, OLD_IS_ARGMAX, NEW_IS_ARGMAX, "thinking repetition helper"
    )
    text = replace_once(
        text,
        OLD_UPDATE_STATE_ADDED,
        NEW_UPDATE_STATE_ADDED,
        "thinking update_state added block",
    )
    builtin_path.write_text(text, encoding="utf-8")


def main() -> int:
    root = resolve_vllm_root()
    patch_protocol(root / "entrypoints/openai/chat_completion/protocol.py")
    patch_builtin_processor(root / "v1/sample/logits_processor/builtin.py")
    print(f"Patched reasoning budget controls under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
