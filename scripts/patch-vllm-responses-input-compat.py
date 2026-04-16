#!/usr/bin/env python3
"""
Patch vLLM's Responses API request model to be more forgiving with common
frontend payload shapes.

This patch targets two integration issues we have observed in practice:

1. Some web clients serialize omitted optional fields as the literal string
   "[undefined]". vLLM's strict Pydantic schema rejects those values for typed
   fields such as temperature/top_p/tools/tool_choice.
2. Multi-turn assistant history is often sent in a shorthand form like:
     {"role": "assistant", "content": [{"type": "output_text", "text": "..."}]}
   vLLM expects a full ResponseOutputMessage-like item instead, including
   "type", "status", "id", and "annotations".

The patch adds a request pre-normalization step before ResponsesRequest's
existing validators run, keeping the rest of vLLM's parsing path unchanged.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4


VLLM_ROOT_CANDIDATES = [
    Path("/opt/vllm-src/vllm"),
    Path("/usr/local/lib/python3.12/dist-packages/vllm"),
]


def resolve_vllm_root() -> Path:
    for root in VLLM_ROOT_CANDIDATES:
        if (root / "entrypoints/openai/responses/protocol.py").exists():
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


UNDEFINED_SENTINELS = frozenset({"[undefined]", "undefined"})


def is_placeholder_undefined_for_tests(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in UNDEFINED_SENTINELS


def make_response_message_id_for_tests() -> str:
    return f"msg_{uuid4().hex}"


def sanitize_undefined_placeholders_for_tests(value: object) -> object:
    if isinstance(value, list):
        return [sanitize_undefined_placeholders_for_tests(item) for item in value]

    if not isinstance(value, dict):
        return value

    sanitized: dict[object, object] = {}
    for key, item in value.items():
        if key != "text" and is_placeholder_undefined_for_tests(item):
            continue
        sanitized[key] = sanitize_undefined_placeholders_for_tests(item)
    return sanitized


def normalize_responses_assistant_input_for_tests(value: object) -> object:
    if isinstance(value, list):
        return [normalize_responses_assistant_input_for_tests(item) for item in value]

    if not isinstance(value, dict):
        return value

    normalized = {
        key: normalize_responses_assistant_input_for_tests(item)
        for key, item in value.items()
    }

    if normalized.get("role") != "assistant":
        return normalized

    content = normalized.get("content")
    if isinstance(content, str):
        content = [{"type": "output_text", "text": content, "annotations": []}]
    elif isinstance(content, list):
        normalized_parts = []
        for part in content:
            if isinstance(part, str):
                normalized_parts.append(
                    {"type": "output_text", "text": part, "annotations": []}
                )
                continue

            if (
                isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                part_copy = dict(part)
                part_copy.setdefault("annotations", [])
                normalized_parts.append(part_copy)
                continue

            return normalized
        content = normalized_parts
    else:
        return normalized

    normalized["type"] = "message"
    normalized["status"] = normalized.get("status") or "completed"
    normalized["id"] = normalized.get("id") or make_response_message_id_for_tests()
    normalized["content"] = content
    return normalized


def normalize_responses_request_for_tests(data: object) -> object:
    sanitized = sanitize_undefined_placeholders_for_tests(data)
    return normalize_responses_assistant_input_for_tests(sanitized)


OLD_HELPER_ANCHOR = """_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class InputTokensDetails(OpenAIBaseModel):
"""


NEW_HELPER_BLOCK = """_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

_UNDEFINED_SENTINELS = frozenset({"[undefined]", "undefined"})


def _is_placeholder_undefined(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in _UNDEFINED_SENTINELS


def _sanitize_undefined_placeholders(value: object) -> object:
    if isinstance(value, list):
        return [_sanitize_undefined_placeholders(item) for item in value]

    if not isinstance(value, dict):
        return value

    sanitized: dict[object, object] = {}
    for key, item in value.items():
        if key != "text" and _is_placeholder_undefined(item):
            continue
        sanitized[key] = _sanitize_undefined_placeholders(item)
    return sanitized


def _normalize_responses_assistant_input(value: object) -> object:
    if isinstance(value, list):
        return [_normalize_responses_assistant_input(item) for item in value]

    if not isinstance(value, dict):
        return value

    normalized = {
        key: _normalize_responses_assistant_input(item)
        for key, item in value.items()
    }

    if normalized.get("role") != "assistant":
        return normalized

    content = normalized.get("content")
    if isinstance(content, str):
        content = [{"type": "output_text", "text": content, "annotations": []}]
    elif isinstance(content, list):
        normalized_parts = []
        for part in content:
            if isinstance(part, str):
                normalized_parts.append(
                    {"type": "output_text", "text": part, "annotations": []}
                )
                continue

            if (
                isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                part_copy = dict(part)
                part_copy.setdefault("annotations", [])
                normalized_parts.append(part_copy)
                continue

            return normalized
        content = normalized_parts
    else:
        return normalized

    normalized["type"] = "message"
    normalized["status"] = normalized.get("status") or "completed"
    normalized["id"] = normalized.get("id") or f"msg_{random_uuid()}"
    normalized["content"] = content
    return normalized


def _normalize_responses_request_input(data: object) -> object:
    sanitized = _sanitize_undefined_placeholders(data)
    return _normalize_responses_assistant_input(sanitized)


class InputTokensDetails(OpenAIBaseModel):
"""


OLD_VALIDATOR_ANCHOR = """    @model_validator(mode="before")
    @classmethod
    def validate_background(cls, data):
"""


NEW_NORMALIZER_VALIDATOR = """    @model_validator(mode="before")
    @classmethod
    def normalize_compat_input(cls, data):
        if not isinstance(data, dict):
            return data
        return _normalize_responses_request_input(data)

    @model_validator(mode="before")
    @classmethod
    def validate_background(cls, data):
"""


def patch_responses_protocol(text: str) -> str:
    text = replace_once(
        text,
        OLD_HELPER_ANCHOR,
        NEW_HELPER_BLOCK,
        "responses protocol helpers",
    )
    text = replace_once(
        text,
        OLD_VALIDATOR_ANCHOR,
        NEW_NORMALIZER_VALIDATOR,
        "responses request compat validator",
    )
    return text


def main() -> None:
    vllm_root = resolve_vllm_root()
    protocol_path = vllm_root / "entrypoints/openai/responses/protocol.py"
    original = protocol_path.read_text()
    patched = patch_responses_protocol(original)
    if patched != original:
        protocol_path.write_text(patched)
        print(f"Patched {protocol_path}")
    else:
        print(f"No changes needed in {protocol_path}")


if __name__ == "__main__":
    main()
