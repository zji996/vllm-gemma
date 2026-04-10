#!/usr/bin/env python3
"""
Gemma-4-26B-A4B-it FP8 Weight-Only 量化器
直接操作 safetensors 文件 — 零 llm-compressor 依赖

方案: FP8 Weight-Only, per-tensor scaling
      BF16 权重 → float8_e4m3fn + float32 scale
      输出 compressed-tensors 格式, vLLM 直接加载

原理:
  - 对每个可量化的 weight tensor 计算 absmax → scale = absmax / FP8_MAX
  - 量化: w_q = clamp(w / scale, -FP8_MAX, FP8_MAX).to(float8_e4m3fn)
  - 反量化: w ≈ w_q.float() * scale
  - 内存友好: 逐 tensor 处理, 不需要加载整个模型

硬件说明:
  RTX 3080 = SM86, 无原生 FP8 计算
  vLLM 自动用 Marlin/cutlass W8A16 kernel: FP8 存储, BF16 计算
  主要收益: VRAM 减半 (48GB → ~24GB), 双卡可放下

依赖: torch>=2.1, safetensors>=0.4.0 (仅此两个)

用法:
  python3 scripts/quantize-fp8.py                           # 自动检测模型
  python3 scripts/quantize-fp8.py --model-path /path/to/model
  python3 scripts/quantize-fp8.py --dry-run                 # 仅检查, 不执行
"""

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ============================================================
# 常量
# ============================================================
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
FP8_DTYPE = torch.float8_e4m3fn
OUTPUT_SHARD_MAX_BYTES = 4 * 1024**3  # 4GB per shard

# 不量化的 tensor 名称模式 (保留原精度)
# 原则: 只量化 Linear 权重, 跳过所有敏感/小型组件
SKIP_PATTERNS = [
    r"embed_tokens",           # 嵌入层 (tie_word_embeddings → 也是 lm_head)
    r"layernorm",              # 所有 LayerNorm
    r"\.norm\.",               # final norm 等
    r"norm\.weight$",          # model.language_model.norm.weight
    r"layer_scalar",           # per-layer scalar
    r"router\.",               # MoE router/gating (量化会严重影响路由精度)
    r"vision_tower\.",         # 视觉编码器 (~550M, 量化收益小且影响图像理解)
    r"embed_vision\.",         # 视觉→文本投影
    r"patch_embedder\.",       # vision patch embedding
    r"std_bias$",              # vision normalization
    r"std_scale$",             # vision normalization
    r"position_embedding",     # 位置编码
]

# ============================================================
# 工具函数
# ============================================================
class C:
    CYAN = "\033[0;36m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    NC = "\033[0m"

def info(msg):    print(f"{C.CYAN}▸{C.NC} {msg}", flush=True)
def ok(msg):     print(f"{C.GREEN}✔{C.NC} {msg}", flush=True)
def warn(msg):   print(f"{C.YELLOW}⚠{C.NC} {msg}", flush=True)
def err(msg):    print(f"{C.RED}✖{C.NC} {msg}", file=sys.stderr, flush=True)

def fmt_size(nbytes: int) -> str:
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.2f} GB"
    elif nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.1f} MB"
    return f"{nbytes / 1024:.0f} KB"

def fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds/3600:.1f}h"
    elif seconds >= 60:
        return f"{seconds/60:.1f}m"
    return f"{seconds:.1f}s"

# ============================================================
# 量化核心
# ============================================================
def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    """判断 tensor 是否应该被量化为 FP8"""
    # 必须至少 2D (跳过 1D norm weights, scalars)
    if tensor.ndim < 2:
        return False
    # 检查跳过模式
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, name):
            return False
    return True


def quantize_to_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-tensor FP8 量化

    Args:
        tensor: BF16/FP32 weight tensor, shape [*, out, in]

    Returns:
        (fp8_weight, scale)
        fp8_weight: float8_e4m3fn, same shape
        scale: float32, shape [1] (per-tensor scalar)

    Per-tensor 选择理由:
      - vLLM 在 SM86 上使用 cutlass_scaled_mm, 对 per-tensor scale 兼容性最好
      - MoE packed weights (3D) 也能正确处理
      - 精度损失相比 per-channel 微乎其微 (weight-only 场景)
    """
    t = tensor.float()
    # 全 tensor absmax
    amax = t.abs().max().clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    # 量化
    t_scaled = t / scale
    t_fp8 = t_scaled.clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return t_fp8, scale.reshape(1)


def maybe_quantize_gemma4_moe(
    name: str,
    tensor: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    """
    Gemma4 MoE 权重以 3D packed tensor 存储:
      - experts.gate_up_proj: [E, 2I, H]
      - experts.down_proj:    [E, H, I]

    vLLM 的 Gemma4 loader + FusedMoE 更容易稳定加载逐 expert 的 2D 张量，
    并为每个 expert 提供独立 weight_scale:
      - experts.{i}.gate_proj
      - experts.{i}.up_proj
      - experts.{i}.down_proj
      - *.weight_scale

    返回空列表表示不是需要特殊处理的 Gemma4 MoE tensor。
    """
    if tensor.ndim != 3 or ".experts." not in name:
        return []

    out: list[tuple[str, torch.Tensor]] = []

    if name.endswith(".experts.gate_up_proj"):
        num_experts = tensor.shape[0]
        intermediate_size = tensor.shape[1] // 2
        for expert_id in range(num_experts):
            gate_weight = tensor[expert_id, :intermediate_size, :]
            up_weight = tensor[expert_id, intermediate_size:, :]

            gate_q, gate_s = quantize_to_fp8(gate_weight)
            up_q, up_s = quantize_to_fp8(up_weight)

            base = name.replace(".experts.gate_up_proj", f".experts.{expert_id}")
            out.append((f"{base}.gate_proj", gate_q))
            out.append((f"{base}.gate_proj.weight_scale", gate_s))
            out.append((f"{base}.up_proj", up_q))
            out.append((f"{base}.up_proj.weight_scale", up_s))
        return out

    if name.endswith(".experts.down_proj"):
        num_experts = tensor.shape[0]
        for expert_id in range(num_experts):
            down_weight = tensor[expert_id]
            down_q, down_s = quantize_to_fp8(down_weight)
            base = name.replace(".experts.down_proj", f".experts.{expert_id}.down_proj")
            out.append((base, down_q))
            out.append((f"{base}.weight_scale", down_s))
        return out

    return []


# ============================================================
# 模型路径解析
# ============================================================
def resolve_model_path(model_path: str | None) -> Path:
    """自动查找 BF16 源模型"""
    if model_path:
        p = Path(model_path)
        if not (p / "config.json").is_file():
            err(f"模型路径无效 (缺少 config.json): {p}")
            sys.exit(1)
        return p

    root = Path(__file__).resolve().parent.parent
    env_model_path = os.environ.get("QUANTIZE_MODEL_PATH") or os.environ.get("DOWNLOAD_MODEL_PATH")
    candidates = [
        Path(env_model_path).expanduser() if env_model_path else None,
        root / ".cache" / "modelscope" / "google" / "gemma-4-26B-A4B-it",
        root / ".cache" / "huggingface" / "hub" / "models--google--gemma-4-26B-A4B-it",
        Path.home() / ".cache" / "modelscope" / "google" / "gemma-4-26B-A4B-it",
    ]
    for p in candidates:
        if p is None:
            continue
        if (p / "config.json").is_file():
            info(f"自动检测到模型: {C.BOLD}{p}{C.NC}")
            return p

    err("未找到本地模型, 请用 --model-path 指定")
    sys.exit(1)


def resolve_output_dir(output_dir: str | None, model_path: Path) -> Path:
    if output_dir:
        return Path(output_dir)
    return model_path.parent / (model_path.name + "-FP8")


# ============================================================
# 主流程
# ============================================================
def quantize_model(model_dir: Path, output_dir: Path, dry_run: bool = False):
    """
    FP8 量化主流程

    1. 读 safetensors index → 获取 shard 列表和 tensor→shard 映射
    2. 逐 shard 处理: 读每个 tensor → 量化或保留 → 写新 shard
    3. 生成新的 index.json + config.json (含 quantization_config)
    4. 复制 tokenizer/processor 辅助文件
    """
    t0 = time.time()

    # ---- 读取 safetensors index ----
    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.is_file():
        err(f"找不到 {index_file}")
        sys.exit(1)

    with open(index_file) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    info(f"源模型: {C.BOLD}{model_dir}{C.NC}")
    info(f"输出:   {C.BOLD}{output_dir}{C.NC}")
    info(f"Shards: {len(shard_files)}, Tensors: {len(weight_map)}")
    print()

    # ---- 预扫描: 分类所有 tensor ----
    to_quantize = []
    to_skip = []
    for name, shard in weight_map.items():
        # 需要读 tensor 的 ndim 来决定, 但先按 name pattern 预筛
        skip = False
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, name):
                skip = True
                break
        if skip:
            to_skip.append(name)
        else:
            to_quantize.append(name)  # 暂定, 实际读取时还会检查 ndim

    info(f"预计量化: {C.BOLD}{len(to_quantize)}{C.NC} tensors")
    info(f"预计跳过: {C.BOLD}{len(to_skip)}{C.NC} tensors")
    print()

    # 显示量化目标的模式
    q_patterns = set()
    for n in to_quantize:
        p = re.sub(r'\.\d+\.', '.X.', n)
        q_patterns.add(p)
    info("量化目标模式:")
    for p in sorted(q_patterns):
        info(f"  {C.DIM}→{C.NC} {p}")
    print()

    # 显示跳过的模式
    s_patterns = set()
    for n in to_skip:
        p = re.sub(r'\.\d+\.', '.X.', n)
        s_patterns.add(p)
    info("跳过模式:")
    for p in sorted(s_patterns):
        info(f"  {C.DIM}✗{C.NC} {p}")
    print()

    if dry_run:
        warn("Dry run — 仅打印配置, 不执行量化")
        return

    # ---- 检查输出目录 ----
    if output_dir.exists() and any(output_dir.glob("*.safetensors")):
        warn(f"输出目录已存在: {output_dir}")
        resp = input("  覆盖? [y/N] ").strip().lower()
        if resp != "y":
            info("取消")
            return
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 逐 shard 处理 ----
    new_weight_map = {}
    total_src_bytes = 0
    total_dst_bytes = 0
    quantized_count = 0
    skipped_count = 0
    output_shard_idx = 0
    pending_tensors = {}
    pending_bytes = 0

    def flush_shard():
        """写出当前累积的 tensors 到一个 shard 文件"""
        nonlocal output_shard_idx, pending_tensors, pending_bytes
        if not pending_tensors:
            return
        output_shard_idx += 1
        shard_name = f"model-{output_shard_idx:05d}-of-PLACEHOLDER.safetensors"
        out_path = output_dir / shard_name
        save_file(pending_tensors, str(out_path))
        sz = out_path.stat().st_size
        info(f"  写入 {shard_name}: {fmt_size(sz)} ({len(pending_tensors)} tensors)")
        for tname in pending_tensors:
            new_weight_map[tname] = shard_name
        pending_tensors = {}
        pending_bytes = 0

    for shard_file in shard_files:
        src_path = model_dir / shard_file
        src_size = src_path.stat().st_size
        total_src_bytes += src_size
        info(f"处理 {shard_file} ({fmt_size(src_size)})...")

        with safe_open(str(src_path), framework="pt") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                src_elem_bytes = tensor.nelement() * tensor.element_size()

                gemma4_moe_tensors = maybe_quantize_gemma4_moe(name, tensor)
                if gemma4_moe_tensors:
                    total_dst_bytes_local = 0
                    for out_name, out_tensor in gemma4_moe_tensors:
                        dst_bytes = out_tensor.nelement() * out_tensor.element_size()
                        pending_tensors[out_name] = out_tensor
                        pending_bytes += dst_bytes
                        total_dst_bytes_local += dst_bytes
                        new_weight_map[out_name] = ""

                        if pending_bytes >= OUTPUT_SHARD_MAX_BYTES:
                            flush_shard()

                    total_dst_bytes += total_dst_bytes_local
                    quantized_count += 1
                    del tensor
                    continue

                if should_quantize(name, tensor):
                    # 量化
                    fp8_w, scale = quantize_to_fp8(tensor)
                    scale_name = name + "_scale"

                    dst_bytes = fp8_w.nelement() * fp8_w.element_size()
                    dst_bytes += scale.nelement() * scale.element_size()

                    pending_tensors[name] = fp8_w
                    pending_tensors[scale_name] = scale
                    pending_bytes += dst_bytes
                    total_dst_bytes += dst_bytes
                    quantized_count += 1

                    del tensor, fp8_w, scale
                else:
                    # 保留原精度
                    dst_bytes = tensor.nelement() * tensor.element_size()
                    pending_tensors[name] = tensor
                    pending_bytes += dst_bytes
                    total_dst_bytes += dst_bytes
                    skipped_count += 1

                # 超过 shard 大小限制则 flush
                if pending_bytes >= OUTPUT_SHARD_MAX_BYTES:
                    flush_shard()

        gc.collect()

    # flush 剩余
    flush_shard()

    # ---- 修正 shard 文件名中的 PLACEHOLDER ----
    total_shards = output_shard_idx
    final_weight_map = {}
    for tname, shard_name in new_weight_map.items():
        new_name = shard_name.replace("PLACEHOLDER", f"{total_shards:05d}")
        final_weight_map[tname] = new_name

    # 重命名文件
    for i in range(1, total_shards + 1):
        old_name = f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        old_path = output_dir / old_name
        new_path = output_dir / new_name
        if old_path.exists():
            old_path.rename(new_path)

    # ---- 写 safetensors index ----
    new_index = {
        "metadata": {"total_size": total_dst_bytes},
        "weight_map": final_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2, ensure_ascii=False)

    # ---- 生成 config.json (含 quantization_config) ----
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    # 使用 vLLM 原生 fp8 配置。Ampere/SM86 上会走 FP8 存储 + BF16 计算路径，
    # 非原生 FP8 算力由 cutlass/marlin 等后端接管。
    config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_per_tensor": True,
        "act_per_tensor": False,
        "modules_to_not_convert": [
            "lm_head",
            "re:.*embed_tokens.*",
            "re:.*vision_tower.*",
            "re:.*embed_vision.*",
            "re:.*router\\..*",
            "re:.*layernorm.*",
        ],
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    ok("config.json (含 quantization_config)")

    # ---- 复制辅助文件 ----
    aux_files = [
        "tokenizer.json", "tokenizer_config.json",
        "processor_config.json", "chat_template.jinja",
        "special_tokens_map.json", "preprocessor_config.json",
        "generation_config.json",
    ]
    copied = []
    for fname in aux_files:
        src = model_dir / fname
        if src.is_file():
            shutil.copy2(str(src), str(output_dir / fname))
            copied.append(fname)
    if copied:
        ok(f"复制辅助文件: {', '.join(copied)}")

    # ---- 汇总 ----
    elapsed = time.time() - t0
    ratio = total_dst_bytes / total_src_bytes if total_src_bytes > 0 else 0
    print()
    print(f"{C.BOLD}{'═' * 50}{C.NC}")
    ok(f"FP8 量化完成!")
    print(f"{C.BOLD}{'═' * 50}{C.NC}")
    info(f"量化: {quantized_count} tensors")
    info(f"跳过: {skipped_count} tensors (敏感层保留 BF16)")
    info(f"压缩: {fmt_size(total_src_bytes)} → {fmt_size(total_dst_bytes)} ({ratio:.0%})")
    info(f"Shards: {total_shards}")
    info(f"耗时: {fmt_duration(elapsed)}")
    info(f"输出: {C.BOLD}{output_dir}{C.NC}")
    print()
    print(f"  {C.BOLD}下一步:{C.NC}")
    print(f"  修改 docker-compose.yml 中 VLLM_MODEL 指向:")
    print(f"  {output_dir}")
    print()


# ============================================================
# CLI
# ============================================================
def main():
    print()
    print(f"{C.BOLD}╔═══════════════════════════════════════════════════╗{C.NC}")
    print(f"{C.BOLD}║  🔧 FP8 Weight-Only Quantizer (standalone)       ║{C.NC}")
    print(f"{C.BOLD}║  Gemma-4-26B-A4B-it  BF16 → FP8                 ║{C.NC}")
    print(f"{C.BOLD}╚═══════════════════════════════════════════════════╝{C.NC}")
    print()

    parser = argparse.ArgumentParser(
        description="Gemma-4-26B-A4B-it FP8 离线量化 (standalone, 无 llm-compressor)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="BF16 源模型目录 (含 config.json + *.safetensors). "
             "默认: 自动检测 QUANTIZE_MODEL_PATH / DOWNLOAD_MODEL_PATH / .cache/modelscope/google/gemma-4-26B-A4B-it"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录. 默认: <model-path>-FP8"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅扫描并打印量化计划, 不执行"
    )
    args = parser.parse_args()

    # 环境检查
    info(f"Python:       {sys.version.split()[0]}")
    info(f"PyTorch:      {torch.__version__}")
    info(f"FP8 dtype:    {FP8_DTYPE} (max={FP8_MAX})")
    try:
        import safetensors
        info(f"safetensors:  {safetensors.__version__}")
    except Exception:
        pass

    # 检查 FP8 支持
    try:
        _ = torch.tensor([1.0]).to(FP8_DTYPE)
        ok("float8_e4m3fn 支持正常")
    except Exception as e:
        err(f"PyTorch 不支持 float8_e4m3fn: {e}")
        err("需要 PyTorch >= 2.1")
        sys.exit(1)
    print()

    model_path = resolve_model_path(args.model_path)
    output_dir = resolve_output_dir(args.output_dir, model_path)

    quantize_model(model_path, output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
