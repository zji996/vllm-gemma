#!/usr/bin/env python3
"""Benchmark Gemma4 thinking and tool roundtrip behavior on the local OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


def post_json(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> tuple[dict[str, Any], float]:
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
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        parsed = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    return parsed, elapsed


def contains_reasoning_markers(text: str | None) -> bool:
    if not text:
        return False
    return any(marker in text for marker in ("thought\n", "<channel|>", "<|channel>"))


def contains_expected_fragment(text: str | None, expected: str | None) -> bool:
    if not text or not expected:
        return False
    return expected in text


def summarize_runs(samples: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(sample["latency_s"] for sample in samples)
    completion_tokens = [sample["completion_tokens"] for sample in samples]
    prompt_tokens = [sample["prompt_tokens"] for sample in samples]
    finish_reasons = Counter(sample["finish_reason"] for sample in samples)
    content_nonempty = sum(1 for sample in samples if sample["content_nonempty"])
    reasoning_present = sum(1 for sample in samples if sample["reasoning_present"])
    leaked_markers = sum(1 for sample in samples if sample["content_has_reasoning_markers"])
    exact_matches = sum(1 for sample in samples if sample.get("content_exact_match"))

    return {
        "rounds": len(samples),
        "latency_avg_s": statistics.fmean(latencies) if latencies else math.nan,
        "latency_p50_s": latencies[len(latencies) // 2] if latencies else math.nan,
        "latency_max_s": max(latencies) if latencies else math.nan,
        "prompt_tokens_avg": statistics.fmean(prompt_tokens) if prompt_tokens else math.nan,
        "completion_tokens_avg": (
            statistics.fmean(completion_tokens) if completion_tokens else math.nan
        ),
        "content_nonempty_rounds": content_nonempty,
        "reasoning_present_rounds": reasoning_present,
        "content_with_reasoning_markers_rounds": leaked_markers,
        "content_exact_match_rounds": exact_matches,
        "finish_reasons": dict(finish_reasons),
    }


def run_thinking_case(
    *,
    label: str,
    base_url: str,
    api_key: str,
    model: str,
    user_content: str,
    rounds: int,
    max_tokens: int,
    timeout: float,
    expected_content: str | None = None,
) -> dict[str, Any]:
    samples = []
    for round_idx in range(1, rounds + 1):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "reasoning_effort": "medium",
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response, latency_s = post_json(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            api_key,
            payload,
            timeout=timeout,
        )
        message = response["choices"][0]["message"]
        usage = response.get("usage") or {}
        content = message.get("content")
        sample = {
            "round": round_idx,
            "latency_s": latency_s,
            "finish_reason": response["choices"][0].get("finish_reason"),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "content": content,
            "reasoning": message.get("reasoning"),
            "content_nonempty": bool((content or "").strip()),
            "reasoning_present": bool((message.get("reasoning") or "").strip()),
            "content_has_reasoning_markers": contains_reasoning_markers(content),
            "content_exact_match": expected_content is not None
            and (content or "").strip() == expected_content,
        }
        samples.append(sample)

    return {
        "scenario": label,
        "type": "thinking",
        "user_content": user_content,
        "max_tokens": max_tokens,
        "expected_content": expected_content,
        "summary": summarize_runs(samples),
        "samples": samples,
    }


def run_tool_roundtrip_case(
    *,
    label: str,
    base_url: str,
    api_key: str,
    model: str,
    rounds: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
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
    samples = []
    for round_idx in range(1, rounds + 1):
        user_content = f"Round {round_idx}: Call add for 123 and 456. Then wait for the tool result."
        first_payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "tools": tools,
            "tool_choice": "auto",
            "reasoning_effort": "medium",
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        first_response, first_latency_s = post_json(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            api_key,
            first_payload,
            timeout=timeout,
        )
        assistant = first_response["choices"][0]["message"]
        tool_calls = assistant.get("tool_calls") or []
        usage1 = first_response.get("usage") or {}

        second_response = None
        second_latency_s = math.nan
        second_message: dict[str, Any] = {}
        usage2: dict[str, Any] = {}
        if tool_calls:
            second_payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": user_content},
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
                "max_tokens": max_tokens,
                "stream": False,
            }
            second_response, second_latency_s = post_json(
                f"{base_url.rstrip('/')}/v1/chat/completions",
                api_key,
                second_payload,
                timeout=timeout,
            )
            second_message = second_response["choices"][0]["message"]
            usage2 = second_response.get("usage") or {}

        content = second_message.get("content")
        sample = {
            "round": round_idx,
            "first_latency_s": first_latency_s,
            "second_latency_s": second_latency_s,
            "latency_s": first_latency_s + (second_latency_s if not math.isnan(second_latency_s) else 0.0),
            "finish_reason": (
                second_response["choices"][0].get("finish_reason")
                if second_response is not None
                else first_response["choices"][0].get("finish_reason")
            ),
            "prompt_tokens": int(usage1.get("prompt_tokens", 0)) + int(usage2.get("prompt_tokens", 0)),
            "completion_tokens": int(usage1.get("completion_tokens", 0))
            + int(usage2.get("completion_tokens", 0)),
            "tool_calls_count": len(tool_calls),
            "tool_call_name": tool_calls[0]["function"]["name"] if tool_calls else None,
            "content": content,
            "reasoning": second_message.get("reasoning"),
            "content_nonempty": bool((content or "").strip()),
            "reasoning_present": bool((second_message.get("reasoning") or "").strip()),
            "content_has_reasoning_markers": contains_reasoning_markers(content),
            "content_exact_match": contains_expected_fragment(content, "579"),
        }
        samples.append(sample)

    summary = summarize_runs(samples)
    summary["tool_call_rounds"] = sum(1 for sample in samples if sample["tool_calls_count"] > 0)
    return {
        "scenario": label,
        "type": "tool_roundtrip",
        "max_tokens": max_tokens,
        "expected_content": "579",
        "summary": summary,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    host = os.environ.get("VLLM_HOST", "127.0.0.1")
    port = os.environ.get("VLLM_HOST_PORT", "8000")
    parser.add_argument("--base-url", default=f"http://{host}:{port}")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "gemma"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    payload = {
        "base_url": args.base_url,
        "model": args.model,
        "rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "scenarios": [],
    }

    scenarios = [
        run_thinking_case(
            label="thinking_greeting_medium",
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            user_content="你好",
            rounds=args.rounds,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            expected_content="你好！请问有什么我可以帮您的吗？",
        ),
        run_thinking_case(
            label="thinking_math_medium",
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            user_content="Solve 123 * 456. Return only the final number.",
            rounds=args.rounds,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            expected_content="56088",
        ),
        run_tool_roundtrip_case(
            label="tool_roundtrip_medium",
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            rounds=args.rounds,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        ),
    ]

    payload["scenarios"] = scenarios

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    for scenario in scenarios:
        summary = scenario["summary"]
        print(
            "[bench] "
            f"{scenario['scenario']} "
            f"rounds={summary['rounds']} "
            f"content={summary['content_nonempty_rounds']}/{summary['rounds']} "
            f"exact={summary['content_exact_match_rounds']}/{summary['rounds']} "
            f"reasoning={summary['reasoning_present_rounds']}/{summary['rounds']} "
            f"p50={summary['latency_p50_s']:.2f}s "
            f"max={summary['latency_max_s']:.2f}s "
            f"finish={summary['finish_reasons']}"
        )
    print(f"[saved] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
