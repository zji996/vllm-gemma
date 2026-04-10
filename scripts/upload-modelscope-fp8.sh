#!/usr/bin/env bash
set -euo pipefail

# Upload the local FP8 export to ModelScope using the official HTTP-based
# methods from the docs. Prefer `modelscope upload`; fall back to SDK
# `upload_folder` when the CLI is unavailable.
#
# Usage:
#   ./scripts/upload-modelscope-fp8.sh
#   MODELSCOPE_ACCESS_TOKEN=ms-xxx ./scripts/upload-modelscope-fp8.sh
#   UPLOAD_METHOD=sdk ./scripts/upload-modelscope-fp8.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
VENV_DIR="${ROOT_DIR}/.cache/venv"
PYTHON="${VENV_DIR}/bin/python3"
MODELSCOPE_CLI="${VENV_DIR}/bin/modelscope"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}▸${NC} $*"; }
success() { echo -e "${GREEN}✔${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC} $*"; }
error()   { echo -e "${RED}✖${NC} $*" >&2; }

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

MS_USERNAME="${MS_USERNAME:-kuohao}"
MS_REPO_NAME="${MS_REPO_NAME:-gemma-4-26B-A4B-it-FP8}"
SRC_DIR="${SRC_DIR:-${ROOT_DIR}/.cache/modelscope/${MS_USERNAME}/${MS_REPO_NAME}}"
MS_REPO_ID="${MS_REPO_ID:-${MS_USERNAME}/${MS_REPO_NAME}}"
MS_ENDPOINT="${MS_ENDPOINT:-https://www.modelscope.cn}"
UPLOAD_METHOD="${UPLOAD_METHOD:-auto}"
MAX_WORKERS="${MAX_WORKERS:-8}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload FP8 model files}"
COMMIT_DESCRIPTION="${COMMIT_DESCRIPTION:-Upload local FP8 export via ModelScope HTTP uploader}"

# Backward compatibility: accept the old env var name too.
MODELSCOPE_ACCESS_TOKEN="${MODELSCOPE_ACCESS_TOKEN:-${MODELSCOPE_API_TOKEN:-}}"

ensure_venv() {
    if [[ -f "${PYTHON}" ]] && "${PYTHON}" -c "import modelscope" 2>/dev/null; then
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        error "python3 not found. Please install Python 3.8+."
        exit 1
    fi

    if [[ ! -f "${PYTHON}" ]]; then
        info "Creating venv at ${BOLD}.cache/venv/${NC} ..."
        python3 -m venv "${VENV_DIR}"
    fi

    info "Installing/refreshing modelscope SDK into venv..."
    "${VENV_DIR}/bin/pip" install --quiet --upgrade modelscope
    success "modelscope SDK ready."
}

validate_inputs() {
    if [[ -z "${MODELSCOPE_ACCESS_TOKEN}" ]]; then
        error "MODELSCOPE_ACCESS_TOKEN is required."
        echo "You can also keep using MODELSCOPE_API_TOKEN for compatibility." >&2
        exit 1
    fi

    if [[ ! -d "${SRC_DIR}" ]]; then
        error "Source directory not found: ${SRC_DIR}"
        exit 1
    fi

    for required in \
        README.md \
        config.json \
        configuration.json \
        generation_config.json \
        model.safetensors.index.json \
        tokenizer.json \
        tokenizer_config.json \
        processor_config.json \
        chat_template.jinja
    do
        if [[ ! -f "${SRC_DIR}/${required}" ]]; then
            error "Missing required file: ${SRC_DIR}/${required}"
            exit 1
        fi
    done
}

print_plan() {
    echo ""
    echo -e "${BOLD}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║  ModelScope FP8 Upload                            ║${NC}"
    echo -e "${BOLD}╚════════════════════════════════════════════════════╝${NC}"
    echo ""
    info "Repo ID:      ${BOLD}${MS_REPO_ID}${NC}"
    info "Source dir:   ${BOLD}${SRC_DIR}${NC}"
    info "Endpoint:     ${BOLD}${MS_ENDPOINT}${NC}"
    info "Upload mode:  ${BOLD}${UPLOAD_METHOD}${NC}"
    info "Max workers:  ${BOLD}${MAX_WORKERS}${NC}"
    echo ""
}

upload_with_cli() {
    if [[ ! -x "${MODELSCOPE_CLI}" ]]; then
        return 1
    fi

    info "Uploading with ModelScope CLI (HTTP uploader)..."
    "${MODELSCOPE_CLI}" upload \
        "${MS_REPO_ID}" \
        "${SRC_DIR}" \
        --repo-type model \
        --include '*' \
        --exclude '.git/*' \
        --commit-message "${COMMIT_MESSAGE}" \
        --commit-description "${COMMIT_DESCRIPTION}" \
        --token "${MODELSCOPE_ACCESS_TOKEN}" \
        --max-workers "${MAX_WORKERS}" \
        --endpoint "${MS_ENDPOINT}"
}

upload_with_sdk() {
    info "Uploading with ModelScope Python SDK upload_folder()..."
    MS_REPO_ID="${MS_REPO_ID}" \
    SRC_DIR="${SRC_DIR}" \
    MODELSCOPE_ACCESS_TOKEN="${MODELSCOPE_ACCESS_TOKEN}" \
    COMMIT_MESSAGE="${COMMIT_MESSAGE}" \
    MAX_WORKERS="${MAX_WORKERS}" \
    MS_ENDPOINT="${MS_ENDPOINT}" \
    "${PYTHON}" - <<'PY'
import os
from modelscope.hub.api import HubApi

repo_id = os.environ["MS_REPO_ID"]
folder_path = os.environ["SRC_DIR"]
token = os.environ["MODELSCOPE_ACCESS_TOKEN"]
commit_message = os.environ["COMMIT_MESSAGE"]
max_workers = int(os.environ["MAX_WORKERS"])
endpoint = os.environ["MS_ENDPOINT"]

api = HubApi(endpoint=endpoint)
api.login(token)
api.upload_folder(
    repo_id=repo_id,
    folder_path=folder_path,
    commit_message=commit_message,
    repo_type="model",
    max_workers=max_workers,
)
PY
}

show_help() {
    cat <<EOF
Usage: ./scripts/upload-modelscope-fp8.sh

Environment:
  MODELSCOPE_ACCESS_TOKEN   ModelScope Access Token from https://modelscope.cn/my/myaccesstoken
  MODELSCOPE_API_TOKEN      Backward-compatible alias for MODELSCOPE_ACCESS_TOKEN
  MS_USERNAME               Repo owner, default: kuohao
  MS_REPO_NAME              Repo name, default: gemma-4-26B-A4B-it-FP8
  MS_REPO_ID                Optional explicit repo id, overrides owner/name
  SRC_DIR                   Local FP8 export dir
  UPLOAD_METHOD             auto | cli | sdk
  MAX_WORKERS               Upload concurrency, default: 8
  COMMIT_MESSAGE            Commit message
  COMMIT_DESCRIPTION        Commit description (CLI only)
  MS_ENDPOINT               Default: https://www.modelscope.cn

Examples:
  ./scripts/upload-modelscope-fp8.sh
  MODELSCOPE_ACCESS_TOKEN=ms-xxx ./scripts/upload-modelscope-fp8.sh
  UPLOAD_METHOD=sdk ./scripts/upload-modelscope-fp8.sh
EOF
}

case "${1:-}" in
    -h|--help|help)
        show_help
        exit 0
        ;;
esac

ensure_venv
validate_inputs
print_plan

case "${UPLOAD_METHOD}" in
    auto)
        if upload_with_cli; then
            :
        else
            warn "ModelScope CLI unavailable or failed to start, falling back to SDK upload_folder()."
            upload_with_sdk
        fi
        ;;
    cli)
        upload_with_cli
        ;;
    sdk)
        upload_with_sdk
        ;;
    *)
        error "Unknown UPLOAD_METHOD: ${UPLOAD_METHOD} (expected auto|cli|sdk)"
        exit 1
        ;;
esac

echo ""
success "Upload completed."
