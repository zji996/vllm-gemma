MODELSCOPE_CACHE_DIR="${VLLM_ROOT_DIR}/.cache/modelscope"
MODELSCOPE_CONTAINER_CACHE_DIR="/root/.cache/modelscope"
MODELSCOPE_VENV_DIR="${VLLM_ROOT_DIR}/.cache/venv"
MODELSCOPE_PYTHON="${MODELSCOPE_VENV_DIR}/bin/python3"
VLLM_MODEL_SYNC_POLICY="${VLLM_MODEL_SYNC_POLICY:-if_missing}"

modelscope_ensure_venv() {
    if [[ -f "${MODELSCOPE_PYTHON}" ]] && \
        "${MODELSCOPE_PYTHON}" -c "import modelscope" >/dev/null 2>&1; then
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        error "python3 not found. Please install Python 3.8+."
        exit 1
    fi

    if [[ ! -f "${MODELSCOPE_PYTHON}" ]]; then
        info "Creating venv at ${BOLD}.cache/venv/${NC} ..."
        python3 -m venv "${MODELSCOPE_VENV_DIR}"
    fi

    info "Installing/refreshing modelscope SDK into venv..."
    "${MODELSCOPE_VENV_DIR}/bin/pip" install --quiet --upgrade modelscope
    success "modelscope SDK ready."
}

modelscope_abs_path() {
    local path="${1:?path is required}"
    python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$path"
}

modelscope_is_local_path_ref() {
    local ref="${1:-}"
    [[ -n "$ref" ]] || return 1
    [[ -e "$ref" || -e "${VLLM_ROOT_DIR}/${ref}" ]]
}

modelscope_cache_path_for_model_id() {
    local model_id="${1:?model_id is required}"
    printf '%s/%s' "${MODELSCOPE_CACHE_DIR}" "${model_id}"
}

modelscope_profile_model_id_var() {
    local profile="${1:?profile is required}"
    case "$profile" in
        gemma26b) printf '%s' "MS_GEMMA26B_MODEL_ID" ;;
        *)
            error "No model id mapping is defined for profile '${profile}'."
            exit 1
            ;;
    esac
}

modelscope_profile_model_path_var() {
    local profile="${1:?profile is required}"
    case "$profile" in
        gemma26b) printf '%s' "MS_GEMMA26B_MODEL_PATH" ;;
        *)
            error "No model path mapping is defined for profile '${profile}'."
            exit 1
            ;;
    esac
}

modelscope_profile_default_model_id() {
    local profile="${1:?profile is required}"
    case "$profile" in
        gemma26b) printf '%s' "kuohao/gemma-4-26B-A4B-it-FP8" ;;
        *)
            error "No default model id is defined for profile '${profile}'."
            exit 1
            ;;
    esac
}

modelscope_snapshot_download() {
    local model_id="${1:?model_id is required}"
    local local_files_only="${2:-false}"
    local revision="${3:-}"
    local access_token="${MODELSCOPE_ACCESS_TOKEN:-${MODELSCOPE_API_TOKEN:-}}"

    modelscope_ensure_venv

    MODEL_ID="${model_id}" \
    CACHE_DIR="${MODELSCOPE_CACHE_DIR}" \
    LOCAL_FILES_ONLY="${local_files_only}" \
    REVISION="${revision}" \
    MODELSCOPE_ACCESS_TOKEN="${access_token}" \
    "${MODELSCOPE_PYTHON}" - <<'PY'
from modelscope import snapshot_download
import os

model_id = os.environ["MODEL_ID"]
cache_dir = os.environ["CACHE_DIR"]
local_files_only = os.environ["LOCAL_FILES_ONLY"].lower() == "true"
revision = os.environ.get("REVISION") or None
token = os.environ.get("MODELSCOPE_ACCESS_TOKEN") or None

path = snapshot_download(
    model_id=model_id,
    cache_dir=cache_dir,
    local_files_only=local_files_only,
    revision=revision,
    token=token,
)
print(path)
PY
}

modelscope_resolve_host_dir() {
    local model_ref="${1:?model_ref is required}"
    local sync_policy="${2:-${VLLM_MODEL_SYNC_POLICY}}"
    local expected_path
    local resolved_path

    if modelscope_is_local_path_ref "$model_ref"; then
        if [[ -e "$model_ref" ]]; then
            modelscope_abs_path "$model_ref"
        else
            modelscope_abs_path "${VLLM_ROOT_DIR}/${model_ref}"
        fi
        return 0
    fi

    expected_path="$(modelscope_cache_path_for_model_id "$model_ref")"
    case "$sync_policy" in
        always)
            info "Syncing model snapshot from ModelScope: ${BOLD}${model_ref}${NC}"
            resolved_path="$(modelscope_snapshot_download "$model_ref" false)"
            ;;
        if_missing)
            if [[ -f "${expected_path}/config.json" ]]; then
                resolved_path="${expected_path}"
            else
                info "Model snapshot missing locally. Downloading: ${BOLD}${model_ref}${NC}"
                resolved_path="$(modelscope_snapshot_download "$model_ref" false)"
            fi
            ;;
        never)
            if [[ ! -f "${expected_path}/config.json" ]]; then
                error "Model snapshot not found locally: ${expected_path}"
                error "Use VLLM_MODEL_SYNC_POLICY=if_missing|always, or run ./download-model.sh ${model_ref}"
                exit 1
            fi
            resolved_path="${expected_path}"
            ;;
        *)
            error "Invalid VLLM_MODEL_SYNC_POLICY: ${sync_policy} (expected: if_missing, always, never)"
            exit 1
            ;;
    esac

    if [[ ! -f "${resolved_path}/config.json" ]]; then
        error "Resolved model directory is missing config.json: ${resolved_path}"
        exit 1
    fi

    printf '%s\n' "$resolved_path"
}

modelscope_apply_local_patches() {
    local profile="${1:?profile is required}"
    local host_dir="${2:?host_dir is required}"
    local patch_script="${VLLM_ROOT_DIR}/scripts/patch-modelscope-gemma4-chat-template.py"

    case "$profile" in
        gemma26b)
            info "Applying local Gemma 4 chat template patch..."
            python3 "${patch_script}" "${host_dir}" >/dev/null
            ;;
    esac
}

modelscope_container_path_for_host_dir() {
    local host_dir="${1:?host_dir is required}"
    local abs_cache_dir abs_host_dir

    abs_cache_dir="$(modelscope_abs_path "${MODELSCOPE_CACHE_DIR}")"
    abs_host_dir="$(modelscope_abs_path "${host_dir}")"

    if [[ "${abs_host_dir}" != "${abs_cache_dir}" && "${abs_host_dir}" != "${abs_cache_dir}/"* ]]; then
        error "Model directory is outside the mounted ModelScope cache: ${abs_host_dir}"
        error "Place local variants under ${MODELSCOPE_CACHE_DIR} so the container can see them."
        exit 1
    fi

    printf '%s%s\n' "${MODELSCOPE_CONTAINER_CACHE_DIR}" "${abs_host_dir#${abs_cache_dir}}"
}

prepare_profile_model() {
    local profile="${1:?profile is required}"
    local model_id_var model_path_var model_ref default_model_id
    local host_dir container_dir

    model_id_var="$(modelscope_profile_model_id_var "$profile")"
    model_path_var="$(modelscope_profile_model_path_var "$profile")"
    default_model_id="$(modelscope_profile_default_model_id "$profile")"
    model_ref="${!model_id_var:-${default_model_id}}"

    host_dir="$(modelscope_resolve_host_dir "$model_ref" "${VLLM_MODEL_SYNC_POLICY}")"
    modelscope_apply_local_patches "$profile" "$host_dir"
    container_dir="$(modelscope_container_path_for_host_dir "$host_dir")"

    printf -v "$model_path_var" '%s' "$container_dir"
    export "$model_path_var"

    export VLLM_USE_MODELSCOPE=False
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    info "Model snapshot: ${BOLD}${host_dir}${NC}"
    info "Container path: ${BOLD}${container_dir}${NC}"
    info "Startup mode: ${BOLD}offline local path${NC} (sync policy: ${VLLM_MODEL_SYNC_POLICY})"
}
