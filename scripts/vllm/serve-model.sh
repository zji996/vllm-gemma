#!/usr/bin/env bash
set -euo pipefail

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

append_flag() {
    local flag="${1:?flag is required}"
    local value="${2:-}"
    if [[ -n "${value}" ]]; then
        ARGS+=("${flag}" "${value}")
    fi
}

if [[ $# -gt 0 ]]; then
    exec "$@"
fi

MODEL="${VLLM_MODEL:-}"
if [[ -z "${MODEL}" ]]; then
    echo "VLLM_MODEL is required." >&2
    exit 1
fi

ARGS=("${MODEL}")

# ---- 基础参数 ----
append_flag "--served-model-name" "${VLLM_SERVED_MODEL_NAME:-}"
append_flag "--tensor-parallel-size" "${VLLM_TENSOR_PARALLEL_SIZE:-}"
append_flag "--pipeline-parallel-size" "${VLLM_PIPELINE_PARALLEL_SIZE:-}"
append_flag "--gpu-memory-utilization" "${VLLM_GPU_MEMORY_UTILIZATION:-}"
append_flag "--max-model-len" "${VLLM_MAX_MODEL_LEN:-}"
append_flag "--max-num-seqs" "${VLLM_MAX_NUM_SEQS:-}"
append_flag "--api-key" "${VLLM_API_KEY:-}"

# ---- Gemma 4 tool calling ----
if is_truthy "${SERVE_ENABLE_AUTO_TOOL_CHOICE:-true}"; then
    ARGS+=("--enable-auto-tool-choice")
fi
append_flag "--tool-call-parser" "${SERVE_TOOL_CALL_PARSER:-gemma4}"

# ---- Gemma 4 thinking / reasoning ----
# Keep the reasoning parser enabled independently from request-side thinking.
# Default requests remain non-thinking unless the prompt/template injects
# `<|think|>`, but when a request does opt into thinking we still want the
# OpenAI response to split thought text out of `message.content`.
append_flag "--reasoning-parser" "${SERVE_REASONING_PARSER:-gemma4}"
# Optional, patched-parser-only heuristic:
# when enabled, salvage explicit trailing markers like `Final Answer: ...`
# if the model omitted `<channel|>`. Default stays conservative/off.
export SERVE_GEMMA4_HEURISTIC_FINAL_ANSWER="${SERVE_GEMMA4_HEURISTIC_FINAL_ANSWER:-false}"

# ---- 多模态 ----
append_flag "--mm-processor-cache-type" "${SERVE_MM_PROCESSOR_CACHE_TYPE:-${VLLM_MM_PROCESSOR_CACHE_TYPE:-}}"
append_flag "--mm-processor-kwargs" "${SERVE_MM_PROCESSOR_KWARGS:-}"
append_flag "--limit-mm-per-prompt" "${VLLM_LIMIT_MM_PER_PROMPT:-}"

# ---- 性能 ----
if is_truthy "${SERVE_ASYNC_SCHEDULING:-true}"; then
    ARGS+=("--async-scheduling")
fi
if is_truthy "${VLLM_ENFORCE_EAGER:-false}"; then
    ARGS+=("--enforce-eager")
fi
append_flag "--moe-backend" "${SERVE_MOE_BACKEND:-${VLLM_MOE_BACKEND:-}}"
append_flag "--compilation-config" "${VLLM_COMPILATION_CONFIG:-}"

# ---- 缓存 ----
if is_truthy "${VLLM_ENABLE_PREFIX_CACHING:-true}"; then
    ARGS+=("--enable-prefix-caching")
fi

# vLLM CLI parameters above are sourced from project-specific env vars.
# Unset them before exec so vLLM does not warn about unknown VLLM_* env names.
unset \
    VLLM_MODEL \
    VLLM_SERVED_MODEL_NAME \
    VLLM_TENSOR_PARALLEL_SIZE \
    VLLM_PIPELINE_PARALLEL_SIZE \
    VLLM_GPU_MEMORY_UTILIZATION \
    VLLM_MAX_MODEL_LEN \
    VLLM_MAX_NUM_SEQS \
    VLLM_LIMIT_MM_PER_PROMPT \
    VLLM_MOE_BACKEND \
    VLLM_ENABLE_PREFIX_CACHING

exec vllm serve "${ARGS[@]}"
