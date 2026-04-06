VLLM_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT_DIR="${VLLM_ROOT_DIR:-$(cd "${VLLM_MODULE_DIR}/../.." && pwd)}"

SCRIPT_DIR="${VLLM_ROOT_DIR}"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
COMPOSE=(docker compose -f "${COMPOSE_FILE}")
LOCK_FILE="${SCRIPT_DIR}/.vllm.lock"
STOP_TIMEOUT="${STOP_TIMEOUT:-20}"
PORT="${PORT:-8000}"

declare -ag ALL_PROFILES=()
declare -Ag MODEL_NAMES=()
declare -Ag MODEL_CTX=()
declare -Ag MODEL_GPU=()
declare -Ag MODEL_GPU_COUNT=()
declare -Ag MODEL_HOST_PORT=()
declare -Ag MODEL_DESC=()
declare -Ag MODEL_ICONS=()

register_model() {
    local profile="${1:?profile is required}"
    local name="${2:?name is required}"
    local ctx="${3:?ctx is required}"
    local gpu="${4:?gpu is required}"
    local desc="${5:?desc is required}"
    local icon="${6:?icon is required}"
    local gpu_count="${7:?gpu_count is required}"

    ALL_PROFILES+=("$profile")
    MODEL_NAMES["$profile"]="$name"
    MODEL_CTX["$profile"]="$ctx"
    MODEL_GPU["$profile"]="$gpu"
    MODEL_GPU_COUNT["$profile"]="$gpu_count"
    MODEL_HOST_PORT["$profile"]="8000"
    MODEL_DESC["$profile"]="$desc"
    MODEL_ICONS["$profile"]="$icon"
}

vllm_model_exists() {
    local profile="${1:-}"
    [[ -n "$profile" && -n "${MODEL_NAMES[$profile]+x}" ]]
}

vllm_model_name() {
    local profile="${1:-}"
    printf '%s' "${MODEL_NAMES[$profile]:-}"
}

vllm_model_ctx() {
    local profile="${1:-}"
    printf '%s' "${MODEL_CTX[$profile]:-}"
}

vllm_model_gpu() {
    local profile="${1:-}"
    printf '%s' "${MODEL_GPU[$profile]:-}"
}

vllm_model_desc() {
    local profile="${1:-}"
    printf '%s' "${MODEL_DESC[$profile]:-}"
}

vllm_model_icon() {
    local profile="${1:-}"
    printf '%s' "${MODEL_ICONS[$profile]:-}"
}

vllm_model_gpu_count() {
    local profile="${1:-}"
    printf '%s' "${MODEL_GPU_COUNT[$profile]:-1}"
}

vllm_is_single_gpu() {
    local profile="${1:-}"
    [[ "$(vllm_model_gpu_count "$profile")" == "1" ]]
}

vllm_model_host_port() {
    local profile="${1:-}"
    printf '%s' "${MODEL_HOST_PORT[$profile]:-8000}"
}

vllm_profile_index() {
    local profile="${1:-}"
    local i
    for i in "${!ALL_PROFILES[@]}"; do
        if [[ "${ALL_PROFILES[$i]}" == "$profile" ]]; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

vllm_models_markdown_table() {
    local profile
    local gpu

    printf '| Profile | 模型 | 上下文 | GPU | 说明 |\n'
    printf '|---------|------|--------|-----|------|\n'
    for profile in "${ALL_PROFILES[@]}"; do
        gpu="$(vllm_model_gpu "$profile")"
        printf '| `%s` | %s | %s | %s | %s |\n' \
            "$profile" \
            "$(vllm_model_name "$profile")" \
            "$(vllm_model_ctx "$profile")" \
            "${gpu%GPU}" \
            "$(vllm_model_desc "$profile")"
    done
}

# ---- Gemma 4 Model Registry ----
register_model "gemma26b" "Gemma-4-26B-A4B-it" "64K" "2×GPU" "⭐ 默认: PP=2 / TP=1 / Gemma4 patch" "💎" 2
