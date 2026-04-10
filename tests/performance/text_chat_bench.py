#!/usr/bin/env python3
"""Simple OpenAI-compatible chat completion benchmark for local vLLM baselines."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "请用两到三句话概括流水线并行的优缺点，并给一个简短结论。"
    "为了避免缓存偏置，请保留这段请求编号：{request_id}。"
)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[int(pos)]
    lower_val = values[lower]
    upper_val = values[upper]
    return lower_val + (upper_val - lower_val) * (pos - lower)


def make_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt_template: str,
    request_id: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    prompt = prompt_template.format(request_id=request_id)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        response_body = resp.read().decode("utf-8")
    elapsed = time.perf_counter() - started
    parsed = json.loads(response_body)
    usage = parsed.get("usage") or {}
    content = parsed["choices"][0]["message"].get("content", "")
    return {
        "latency_s": elapsed,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "chars": len(content),
        "finish_reason": parsed["choices"][0].get("finish_reason"),
    }


def run_round(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt_template: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    concurrency: int,
    num_requests: int,
    start_request_id: int,
) -> tuple[dict[str, Any], int]:
    latencies: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    chars = 0
    failures: list[str] = []
    completed = 0

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                make_request,
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt_template=prompt_template,
                request_id=start_request_id + idx,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            for idx in range(num_requests)
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                failures.append(f"HTTP {exc.code}: {body[:300]}")
                continue
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            latencies.append(result["latency_s"])
            prompt_tokens += result["prompt_tokens"]
            completion_tokens += result["completion_tokens"]
            total_tokens += result["total_tokens"]
            chars += result["chars"]
            completed += 1
    wall_s = time.perf_counter() - wall_started
    latencies.sort()

    summary = {
        "concurrency": concurrency,
        "requests": num_requests,
        "completed": completed,
        "failed": len(failures),
        "wall_s": wall_s,
        "req_per_s": completed / wall_s if wall_s else math.nan,
        "prompt_tok_per_s": prompt_tokens / wall_s if wall_s else math.nan,
        "completion_tok_per_s": completion_tokens / wall_s if wall_s else math.nan,
        "total_tok_per_s": total_tokens / wall_s if wall_s else math.nan,
        "latency_avg_s": statistics.fmean(latencies) if latencies else math.nan,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p90_s": percentile(latencies, 0.90),
        "latency_p95_s": percentile(latencies, 0.95),
        "latency_max_s": max(latencies) if latencies else math.nan,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "chars": chars,
        "failures": failures,
    }
    return summary, start_request_id + num_requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_host = os.environ.get("VLLM_HOST", "127.0.0.1")
    default_port = os.environ.get("VLLM_HOST_PORT", "8000")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", f"http://{default_host}:{default_port}"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "abc123"))
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "gemma"))
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--requests-per-level", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help="Repeat the prompt template this many times to simulate long prefill.",
    )
    parser.add_argument("--output-json", help="Write summary JSON to this path")
    args = parser.parse_args()

    levels = [int(item.strip()) for item in args.concurrency.split(",") if item.strip()]
    if not levels:
        raise SystemExit("No concurrency levels provided")
    if args.prompt_repeat < 1:
        raise SystemExit("--prompt-repeat must be >= 1")

    prompt_template = " ".join([args.prompt] * args.prompt_repeat)

    request_id = 1
    if args.warmup > 0:
        print(f"[warmup] sending {args.warmup} request(s)...", flush=True)
        _, request_id = run_round(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt_template=prompt_template,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            concurrency=1,
            num_requests=args.warmup,
            start_request_id=request_id,
        )

    results = []
    for level in levels:
        summary, request_id = run_round(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt_template=prompt_template,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            concurrency=level,
            num_requests=args.requests_per_level,
            start_request_id=request_id,
        )
        results.append(summary)
        print(
            "[bench] "
            f"c={summary['concurrency']} "
            f"done={summary['completed']}/{summary['requests']} "
            f"req/s={summary['req_per_s']:.3f} "
            f"out_tok/s={summary['completion_tok_per_s']:.1f} "
            f"p50={summary['latency_p50_s']:.2f}s "
            f"p95={summary['latency_p95_s']:.2f}s "
            f"fail={summary['failed']}"
        )
        if summary["failures"]:
            print("  failures:")
            for failure in summary["failures"]:
                print(f"    - {failure}")

    payload = {
        "base_url": args.base_url,
        "model": args.model,
        "requests_per_level": args.requests_per_level,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "prompt_repeat": args.prompt_repeat,
        "results": results,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"[saved] {output_path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
