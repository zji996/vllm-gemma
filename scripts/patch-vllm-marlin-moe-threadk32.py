#!/usr/bin/env python3
"""
Patch vLLM Marlin MoE so Gemma4 FP8 expert shapes dispatch on Marlin.

The stock v0.19.0 Marlin MoE path only considers:
  - thread_k in {64, 128}
  - thread_n in {64, 128, 256}
  - num_threads >= 128

Under Gemma4 TP=2, we have seen two real gaps:
  - K=352 requires thread_k=32 coverage
  - N=1056 requires thread_n=96 coverage

This patch:
  - lowers the shared Marlin minimum thread_k guard from 64 to 32
  - lowers the Marlin minimum thread count from 128 to 64
  - adds conservative dense + MoE configs for:
      * (thread_k=32, thread_n=64, threads=64)
      * (thread_k=64, thread_n=96, threads=64)
      * (thread_k=128, thread_n=96, threads=128)

The same coverage is applied to both:
  - csrc/quantization/marlin
  - csrc/moe/marlin_moe_wna16
"""

from pathlib import Path
import sys


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Failed to find expected anchor for {label}")
    return text.replace(old, new, 1)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/opt/vllm-src")

    marlin_cuh = root / "csrc/quantization/marlin/marlin.cuh"
    dense_generate = root / "csrc/quantization/marlin/generate_kernels.py"
    dense_ops = root / "csrc/quantization/marlin/marlin.cu"
    moe_generate = root / "csrc/moe/marlin_moe_wna16/generate_kernels.py"
    moe_ops = root / "csrc/moe/marlin_moe_wna16/ops.cu"

    marlin_cuh_text = marlin_cuh.read_text()
    marlin_cuh_text = replace_once(
        marlin_cuh_text,
        "static constexpr int min_thread_k = 64;",
        "static constexpr int min_thread_k = 32;",
        "marlin min_thread_k",
    )
    marlin_cuh.write_text(marlin_cuh_text)
    print(f"Patched {marlin_cuh}")

    dense_generate_text = dense_generate.read_text()
    dense_generate_text = replace_once(
        dense_generate_text,
        "THREAD_CONFIGS = [(128, 128, 256), (64, 256, 256), (64, 128, 128), (128, 64, 128)]",
        "THREAD_CONFIGS = [(128, 128, 256), (64, 256, 256), (64, 128, 128), (128, 64, 128), (64, 96, 64), (128, 96, 128), (32, 64, 64)]",
        "dense generate THREAD_CONFIGS",
    )
    dense_generate.write_text(dense_generate_text)
    print(f"Patched {dense_generate}")

    dense_ops_text = dense_ops.read_text()
    dense_ops_text = replace_once(
        dense_ops_text,
        """thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};""",
        """thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128},
    {64, 96, 64},
    {128, 96, 128},
    {32, 64, 64}};""",
        "dense small_batch_thread_configs",
    )
    dense_ops_text = replace_once(
        dense_ops_text,
        """thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};""",
        """thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128},
    {64, 96, 64},
    {128, 96, 128},
    {32, 64, 64}};""",
        "dense large_batch_thread_configs",
    )
    dense_ops_text = replace_once(
        dense_ops_text,
        """  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {""",
        """  // num_threads must be at least 64 (= 2 warps)
  if (th_config.num_threads < 64) {""",
        "dense min num_threads",
    )
    dense_ops.write_text(dense_ops_text)
    print(f"Patched {dense_ops}")

    moe_generate_text = moe_generate.read_text()
    moe_generate_text = replace_once(
        moe_generate_text,
        "THREAD_CONFIGS = [(128, 128, 256), (64, 256, 256), (64, 128, 128), (128, 64, 128)]",
        "THREAD_CONFIGS = [(128, 128, 256), (64, 256, 256), (64, 128, 128), (128, 64, 128), (64, 96, 64), (128, 96, 128), (32, 64, 64)]",
        "moe generate THREAD_CONFIGS",
    )
    moe_generate.write_text(moe_generate_text)
    print(f"Patched {moe_generate}")

    moe_ops_text = moe_ops.read_text()
    moe_ops_text = replace_once(
        moe_ops_text,
        """thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};""",
        """thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128},
    {64, 96, 64},
    {128, 96, 128},
    {32, 64, 64}};""",
        "moe small_batch_thread_configs",
    )
    moe_ops_text = replace_once(
        moe_ops_text,
        """thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};""",
        """thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128},
    {64, 96, 64},
    {128, 96, 128},
    {32, 64, 64}};""",
        "moe large_batch_thread_configs",
    )
    moe_ops_text = replace_once(
        moe_ops_text,
        """  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {""",
        """  // num_threads must be at least 64 (= 2 warps)
  if (th_config.num_threads < 64) {""",
        "moe min num_threads",
    )
    moe_ops.write_text(moe_ops_text)
    print(f"Patched {moe_ops}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
