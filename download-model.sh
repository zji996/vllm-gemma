#!/usr/bin/env bash
# ============================================================
# ModelScope 模型下载脚本
# 下载模型到 .cache/modelscope/, 与 vLLM 容器内缓存路径幂等
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${SCRIPT_DIR}/.cache/modelscope"
MODEL_ID="${1:-google/gemma-4-26B-A4B-it}"

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

# ---- 检查 Python 环境 ----
check_python() {
    if ! command -v python3 &>/dev/null; then
        error "python3 not found. Please install Python 3.8+."
        exit 1
    fi
}

# ---- 确保 modelscope SDK 已安装 ----
ensure_modelscope() {
    if python3 -c "import modelscope" 2>/dev/null; then
        return 0
    fi

    warn "modelscope SDK not found. Installing..."
    pip3 install --quiet --upgrade modelscope
    success "modelscope SDK installed."
}

# ---- 下载模型 ----
download_model() {
    local model_id="$1"
    local cache_dir="$2"
    local model_dir="${cache_dir}/hub/${model_id}"

    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║  📦 ModelScope Model Downloader              ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
    echo ""
    info "Model:     ${BOLD}${model_id}${NC}"
    info "Cache dir: ${BOLD}${cache_dir}${NC}"
    info "Model dir: ${BOLD}${model_dir}${NC}"
    echo ""

    # 检查模型是否已下载 (检查 config.json 存在)
    if [[ -f "${model_dir}/config.json" ]]; then
        success "Model config found at ${model_dir}/config.json"
        info "Running incremental check (skipping existing files)..."
    else
        info "Starting fresh download..."
    fi

    echo ""

    # 使用 modelscope snapshot_download，天然幂等：
    # - 已下载的文件会跳过
    # - 部分下载的文件会断点续传
    # - cache_dir 结构与 vLLM 容器内 MODELSCOPE_CACHE 完全一致
    python3 -c "
from modelscope import snapshot_download
import os

model_id = '${model_id}'
cache_dir = '${cache_dir}'

print(f'Downloading {model_id} to {cache_dir} ...')
print()

path = snapshot_download(
    model_id,
    cache_dir=cache_dir,
)

print()
print(f'✔ Model ready at: {path}')

# 验证关键文件
config_file = os.path.join(path, 'config.json')
if os.path.isfile(config_file):
    print(f'✔ config.json verified')
else:
    print(f'⚠ config.json not found, model may be incomplete')

# 统计文件
safetensor_files = [f for f in os.listdir(path) if f.endswith('.safetensors')]
total_size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
print(f'✔ {len(safetensor_files)} safetensor file(s), total {total_size / (1024**3):.1f} GB')
"

    local exit_code=$?
    echo ""

    if [[ $exit_code -eq 0 ]]; then
        success "Download complete! Model is ready for vLLM."
        echo ""
        echo -e "  ${CYAN}Next step:${NC} ${BOLD}./vllm.sh start gemma26b${NC}"
        echo ""
    else
        error "Download failed with exit code ${exit_code}."
        exit $exit_code
    fi
}

# ---- 帮助信息 ----
show_help() {
    echo ""
    echo -e "${BOLD}Usage:${NC} $0 [model_id]"
    echo ""
    echo -e "${BOLD}Arguments:${NC}"
    echo "  model_id    ModelScope model ID (default: google/gemma-4-26B-A4B-it)"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  $0                                    # 下载 Gemma-4-26B-A4B-it"
    echo "  $0 google/gemma-4-26B-A4B-it          # 同上 (显式指定)"
    echo ""
    echo -e "${BOLD}Notes:${NC}"
    echo "  - 模型下载到 .cache/modelscope/hub/<model_id>/"
    echo "  - 与 vLLM 容器内 MODELSCOPE_CACHE 路径完全一致"
    echo "  - 支持断点续传, 重复运行不会重新下载已有文件"
    echo ""
}

# ---- Main ----
case "${1:-}" in
    -h|--help|help)
        show_help
        exit 0
        ;;
esac

check_python
ensure_modelscope
download_model "$MODEL_ID" "$CACHE_DIR"
