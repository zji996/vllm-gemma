VLLM_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT_DIR="${VLLM_ROOT_DIR:-$(cd "${VLLM_MODULE_DIR}/../.." && pwd)}"

SCRIPT_DIR="${VLLM_ROOT_DIR}"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
COMPOSE=(docker compose -f "${COMPOSE_FILE}")
LOCK_FILE="${SCRIPT_DIR}/.vllm.lock"
STOP_TIMEOUT="${STOP_TIMEOUT:-20}"
PORT="${PORT:-${VLLM_HOST_PORT:-8000}}"
MODEL_REGISTRY_FILE="${VLLM_MODEL_REGISTRY_FILE:-${VLLM_MODULE_DIR}/models.tsv}"
DEFAULT_PROFILE=""

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
    local host_port="${8:-${VLLM_HOST_PORT:-8000}}"

    ALL_PROFILES+=("$profile")
    MODEL_NAMES["$profile"]="$name"
    MODEL_CTX["$profile"]="$ctx"
    MODEL_GPU["$profile"]="$gpu"
    MODEL_GPU_COUNT["$profile"]="$gpu_count"
    MODEL_HOST_PORT["$profile"]="$host_port"
    MODEL_DESC["$profile"]="$desc"
    MODEL_ICONS["$profile"]="$icon"

    if [[ -z "$DEFAULT_PROFILE" ]]; then
        DEFAULT_PROFILE="$profile"
    fi
}

load_model_registry() {
    local line
    local profile
    local name
    local ctx
    local gpu
    local desc
    local icon
    local gpu_count
    local host_port

    if [[ ! -f "${MODEL_REGISTRY_FILE}" ]]; then
        echo "Model registry file not found: ${MODEL_REGISTRY_FILE}" >&2
        exit 1
    fi

    while IFS=$'\t' read -r profile name ctx gpu desc icon gpu_count host_port; do
        [[ -n "${profile:-}" ]] || continue
        [[ "${profile:0:1}" == "#" ]] && continue
        register_model "$profile" "$name" "$ctx" "$gpu" "$desc" "$icon" "$gpu_count" "$host_port"
    done < "${MODEL_REGISTRY_FILE}"

    if [[ ${#ALL_PROFILES[@]} -eq 0 ]]; then
        echo "Model registry is empty: ${MODEL_REGISTRY_FILE}" >&2
        exit 1
    fi
}

vllm_default_profile() {
    printf '%s' "${DEFAULT_PROFILE}"
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
    printf '%s' "${MODEL_HOST_PORT[$profile]:-${VLLM_HOST_PORT:-8000}}"
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

load_model_registry
