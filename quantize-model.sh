#!/usr/bin/env bash
# ============================================================
# FP8 离线量化入口
# 将 Gemma-4-26B-A4B-it (BF16 ~48GB) 量化为 FP8 (~24GB)
#
# 用法:
#   ./quantize-model.sh                    # 默认: 自动检测模型
#   ./quantize-model.sh --dry-run          # 仅检查, 不执行
#   ./quantize-model.sh --help             # 查看选项
#
# 依赖: 仅 torch + safetensors (自动安装到 .cache/venv/)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.cache/venv"
PYTHON="${VENV_DIR}/bin/python3"
PIP="${VENV_DIR}/bin/pip"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}▸${NC} $*"; }
success() { echo -e "${GREEN}✔${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC} $*"; }

# ---- venv ----
ensure_venv() {
    if [[ ! -f "${PYTHON}" ]]; then
        info "Creating venv at ${BOLD}.cache/venv/${NC} ..."
        python3 -m venv "${VENV_DIR}"
    fi
}

# ---- 依赖 (仅 torch + safetensors) ----
install_deps() {
    if "${PYTHON}" -c "import torch; import safetensors" 2>/dev/null; then
        success "torch + safetensors ready"
        return 0
    fi

    info "Installing dependencies..."
    "${PIP}" install --upgrade torch safetensors
    success "Dependencies installed"
}

# ---- main ----
ensure_venv
install_deps

echo ""
info "Starting FP8 quantization..."
echo ""

exec "${PYTHON}" "${SCRIPT_DIR}/scripts/quantize-fp8.py" "$@"
