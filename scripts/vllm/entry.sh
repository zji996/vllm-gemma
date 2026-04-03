VLLM_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT_DIR="${VLLM_ROOT_DIR:-$(cd "${VLLM_MODULE_DIR}/../.." && pwd)}"

source "${VLLM_MODULE_DIR}/config.sh"
source "${VLLM_MODULE_DIR}/common.sh"
source "${VLLM_MODULE_DIR}/cli.sh"
source "${VLLM_MODULE_DIR}/tui.sh"

vllm_main() {
    local cmd="${1:-}"

    if [[ -z "$cmd" ]]; then
        run_tui
        return 0
    fi

    run_cli_command "$@"
}
