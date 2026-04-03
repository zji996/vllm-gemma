#!/usr/bin/env bash
# ============================================================
# vLLM Model Launcher
# 一键管理 Qwen 3.5 系列模型的启动/停止/切换
# ============================================================
# Usage:
#   ./vllm.sh              - 启动交互式 TUI
#   ./vllm.sh <command>    - 直接执行命令
# ============================================================
set -euo pipefail

VLLM_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${VLLM_ROOT_DIR}/scripts/vllm/entry.sh"

vllm_main "$@"
