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

append_flag "--served-model-name" "${VLLM_SERVED_MODEL_NAME:-qwen}"
append_flag "--tensor-parallel-size" "${VLLM_TENSOR_PARALLEL_SIZE:-}"
append_flag "--gpu-memory-utilization" "${VLLM_GPU_MEMORY_UTILIZATION:-}"
append_flag "--max-model-len" "${VLLM_MAX_MODEL_LEN:-}"
append_flag "--max-num-seqs" "${VLLM_MAX_NUM_SEQS:-}"
append_flag "--api-key" "${VLLM_API_KEY:-abc123}"
append_flag "--reasoning-parser" "${SERVE_REASONING_PARSER:-${VLLM_REASONING_PARSER:-qwen3}}"
if is_truthy "${SERVE_ENABLE_AUTO_TOOL_CHOICE:-${VLLM_ENABLE_AUTO_TOOL_CHOICE:-true}}"; then
    ARGS+=("--enable-auto-tool-choice")
fi
append_flag "--tool-call-parser" "${SERVE_TOOL_CALL_PARSER:-${VLLM_TOOL_CALL_PARSER:-qwen3_coder}}"
if is_truthy "${VLLM_ENFORCE_EAGER:-false}"; then
    ARGS+=("--enforce-eager")
fi
append_flag "--compilation-config" "${VLLM_COMPILATION_CONFIG:-}"
append_flag "--mm-encoder-tp-mode" "${SERVE_MM_ENCODER_TP_MODE:-${VLLM_MM_ENCODER_TP_MODE:-}}"
append_flag "--mm-encoder-attn-backend" "${VLLM_MM_ENCODER_ATTN_BACKEND:-}"
append_flag "--mm-processor-cache-type" "${SERVE_MM_PROCESSOR_CACHE_TYPE:-${VLLM_MM_PROCESSOR_CACHE_TYPE:-}}"
if is_truthy "${VLLM_ENABLE_PREFIX_CACHING:-true}"; then
    ARGS+=("--enable-prefix-caching")
fi
append_flag "--limit-mm-per-prompt" "${VLLM_LIMIT_MM_PER_PROMPT:-}"

exec vllm serve "${ARGS[@]}"
