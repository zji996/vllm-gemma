#!/usr/bin/env python3
"""Smoke-check Gemma4 reasoning/content splitting on the local OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Any


def post_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ensure_clean_content(label: str, content: str | None) -> None:
    if content is None:
        return
    if "thought\n" in content or "<channel|>" in content or "<|channel>" in content:
        raise AssertionError(f"{label}: leaked reasoning markers in content: {content!r}")


def ensure_reasoning_absent(label: str, message: dict[str, Any]) -> None:
    reasoning = message.get("reasoning")
    if reasoning not in (None, ""):
        raise AssertionError(f"{label}: expected no reasoning, got {reasoning!r}")


def run_non_thinking_case(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.0,
        "max_tokens": 64,
        "stream": False,
    }
    response = post_json(f"{base_url.rstrip('/')}/v1/chat/completions", api_key, payload)
    message = response["choices"][0]["message"]
    ensure_clean_content("non-thinking", message.get("content"))
    if not (message.get("content") or "").strip():
        raise AssertionError("non-thinking: expected non-empty content")
    ensure_reasoning_absent("non-thinking", message)
    return response


def run_low_effort_case(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Solve 123 * 456. Return only the final number.",
            }
        ],
        "reasoning_effort": "low",
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": False,
    }
    response = post_json(f"{base_url.rstrip('/')}/v1/chat/completions", api_key, payload)
    message = response["choices"][0]["message"]
    ensure_clean_content("low-effort", message.get("content"))
    if "56088" not in (message.get("content") or ""):
        raise AssertionError(
            f"low-effort: unexpected final content: {message.get('content')!r}"
        )
    ensure_reasoning_absent("low-effort", message)
    return response


def run_medium_case(
    base_url: str,
    api_key: str,
    model: str,
    *,
    expect_final_content: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Solve 123 * 456. Return only the final number.",
            }
        ],
        "reasoning_effort": "medium",
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": False,
    }
    response = post_json(f"{base_url.rstrip('/')}/v1/chat/completions", api_key, payload)
    message = response["choices"][0]["message"]
    ensure_clean_content("medium", message.get("content"))
    if not message.get("reasoning"):
        raise AssertionError("medium: expected reasoning to be split into message.reasoning")
    if expect_final_content and not (message.get("content") or "").strip():
        raise AssertionError(
            "medium: expected non-empty final content when heuristic extraction is enabled"
        )
    return response


def run_tool_case(base_url: str, api_key: str, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]
    first_payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Call add for 123 and 456. Then wait for the tool result.",
            }
        ],
        "tools": tools,
        "tool_choice": "auto",
        "reasoning_effort": "medium",
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False,
    }
    first = post_json(f"{base_url.rstrip('/')}/v1/chat/completions", api_key, first_payload)
    assistant = first["choices"][0]["message"]
    tool_calls = assistant.get("tool_calls") or []
    if not tool_calls:
        raise AssertionError(f"tool-first: expected tool_calls, got {json.dumps(first, ensure_ascii=False)}")

    second_payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Call add for 123 and 456. Then wait for the tool result.",
            },
            {
                "role": "assistant",
                "content": assistant.get("content"),
                "tool_calls": tool_calls,
            },
            {
                "role": "tool",
                "tool_call_id": tool_calls[0]["id"],
                "content": "579",
            },
        ],
        "tools": tools,
        "tool_choice": "auto",
        "reasoning_effort": "medium",
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": False,
    }
    second = post_json(
        f"{base_url.rstrip('/')}/v1/chat/completions", api_key, second_payload
    )
    message = second["choices"][0]["message"]
    ensure_clean_content("tool-second", message.get("content"))
    if "579" not in (message.get("content") or ""):
        raise AssertionError(
            f"tool-second: unexpected final content: {message.get('content')!r}"
        )
    return first, second


def run_heuristic_e2e_case(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    last_response = None
    for attempt in range(1, 4):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "When thinking is enabled, keep the reasoning extremely brief, "
                        "at most 2 short lines. Then end with a final line formatted "
                        "exactly as Final Answer: <answer>."
                    ),
                },
                {
                    "role": "user",
                    "content": "Solve 123 * 456. Return only the final number.",
                },
            ],
            "reasoning_effort": "medium",
            "temperature": 0.0,
            "max_tokens": 768,
            "stream": False,
        }
        response = post_json(
            f"{base_url.rstrip('/')}/v1/chat/completions", api_key, payload
        )
        last_response = response
        message = response["choices"][0]["message"]
        ensure_clean_content(f"heuristic-e2e[{attempt}]", message.get("content"))
        if message.get("content") == "56088" and message.get("reasoning"):
            return response

    assert last_response is not None
    last_message = last_response["choices"][0]["message"]
    raise AssertionError(
        "heuristic-e2e: expected salvaged final content '56088', "
        f"got {last_message.get('content')!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    host = os.environ.get("VLLM_HOST", "127.0.0.1")
    port = os.environ.get("VLLM_HOST_PORT", "8000")
    parser.add_argument("--base-url", default=f"http://{host}:{port}")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "gemma"))
    parser.add_argument(
        "--expect-medium-final-content",
        action="store_true",
        help="Require the medium-effort basic case to return non-empty content.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw responses after validation.",
    )
    parser.add_argument(
        "--check-heuristic-sample",
        action="store_true",
        help="Run an end-to-end case that requires heuristic final-answer extraction.",
    )
    args = parser.parse_args()

    non_thinking = run_non_thinking_case(args.base_url, args.api_key, args.model)
    low_effort = run_low_effort_case(args.base_url, args.api_key, args.model)
    medium = run_medium_case(
        args.base_url,
        args.api_key,
        args.model,
        expect_final_content=args.expect_medium_final_content,
    )
    tool_first, tool_second = run_tool_case(args.base_url, args.api_key, args.model)
    heuristic_e2e = None
    if args.check_heuristic_sample:
        heuristic_e2e = run_heuristic_e2e_case(args.base_url, args.api_key, args.model)

    print("non_thinking: clean content and no reasoning")
    print("low_effort: remains on non-thinking path")
    print("medium: reasoning/content split OK")
    print("tool_roundtrip: reasoning markers removed from final content")
    if heuristic_e2e is not None:
        print("heuristic_e2e: explicit Final Answer marker salvaged into content")

    if args.json:
        payload = {
            "non_thinking": non_thinking,
            "low_effort": low_effort,
            "medium": medium,
            "tool_first": tool_first,
            "tool_second": tool_second,
        }
        if heuristic_e2e is not None:
            payload["heuristic_e2e"] = heuristic_e2e
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
