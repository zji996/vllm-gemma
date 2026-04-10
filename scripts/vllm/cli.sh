# CLI mode (non-interactive, backward compatible)

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

cmd_list() {
    echo ""
    echo -e "${BOLD}📦 Available Gemma 4 Deploy Profiles${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}Note:${NC} PROFILE 是 docker compose profile 名；MODEL 是它实际启动的模型。"
    echo -e "        当前默认部署为 ${BOLD}gemma26b -> Gemma-4-26B-A4B-it${NC}，并行策略 ${BOLD}PP=2 / TP=1${NC}。"
    echo ""

    local running
    running=$(get_running_profile 2>/dev/null || echo "")

    printf "  ${BOLD}%-16s %-28s %-10s %-6s %s${NC}\n" "PROFILE" "MODEL" "CTX LEN" "GPU" "DEPLOYMENT"
    echo "  ──────────────────────────────────────────────────────────────────────────────────"

    _cli_print_row() {
        local profile="$1" model="$2" ctx="$3" gpu="$4" desc="$5"
        local status=""
        if [[ "$profile" == "$running" ]]; then
            status=" ${GREEN}● RUNNING${NC}"
        fi
        printf "  %-16s %-28s %-10s %-6s %s%b\n" "$profile" "$model" "$ctx" "$gpu" "$desc" "$status"
    }

    local profile
    for profile in "${ALL_PROFILES[@]}"; do
        local gpu_label
        gpu_label=$(vllm_model_gpu "$profile")
        gpu_label="${gpu_label%GPU}"
        _cli_print_row \
            "$profile" \
            "$(vllm_model_name "$profile")" \
            "$(vllm_model_ctx "$profile")" \
            "$gpu_label" \
            "$(vllm_model_desc "$profile")"
    done

    echo ""
    echo -e "  ${BOLD}Usage:${NC} $0 start <profile> [--gpu 0|1]"
    echo -e "  ${YELLOW}Tip:${NC}   单卡 profile 可用 --gpu 1 切换到 GPU 1 (端口 8001)"
    echo ""
}

# 解析 --gpu N 参数 (从参数列表中提取)
# 用法: _parse_gpu_arg arg1 arg2 --gpu 1 arg3
# 设置: _PARSED_GPU_ID (空串=未指定)
_parse_gpu_arg() {
    _PARSED_GPU_ID=""
    _PARSED_REMAINING_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gpu)
                if [[ -z "${2:-}" ]]; then
                    error "--gpu requires a value (0 or 1)"
                    exit 1
                fi
                _PARSED_GPU_ID="$2"
                shift 2
                ;;
            *)
                _PARSED_REMAINING_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

# 设置 GPU 相关环境变量 (由 cmd_start 调用)
# 返回 0 表示成功, 1 表示跳过 (双卡 profile 忽略 --gpu)
_apply_gpu_selection() {
    local profile="$1"
    local gpu_id="$2"

    # 验证 gpu_id
    if [[ "$gpu_id" != "0" && "$gpu_id" != "1" ]]; then
        error "Invalid GPU ID: '${gpu_id}'. Must be 0 or 1."
        exit 1
    fi

    # 双卡 profile 忽略 --gpu
    if ! vllm_is_single_gpu "$profile"; then
        warn "Profile '${profile}' uses dual GPU — ignoring --gpu ${gpu_id}"
        return 0
    fi

    local target_port
    target_port=$(gpu_to_port "$gpu_id")

    export VLLM_GPU_ID="$gpu_id"
    export VLLM_HOST_PORT="$target_port"
    PORT="$target_port"

    info "GPU: ${BOLD}cuda:${gpu_id}${NC}  Port: ${BOLD}${target_port}${NC}"
}

cmd_start() {
    local profile="${1:-}"
    shift || true

    # 解析剩余参数中的 --gpu
    _parse_gpu_arg "$@"
    local gpu_id="$_PARSED_GPU_ID"

    if [[ -z "$profile" ]]; then
        error "请指定要启动的模型 profile"
        echo ""
        cmd_list
        exit 1
    fi

    if ! validate_profile "$profile"; then
        error "Unknown profile: ${profile}"
        echo ""
        cmd_list
        exit 1
    fi

    PORT="$(vllm_model_host_port "$profile")"

    # 应用 GPU 选择
    if [[ -n "$gpu_id" ]]; then
        _apply_gpu_selection "$profile" "$gpu_id"
    fi

    # 检查目标端口是否有容器在运行
    local running
    running=$(get_running_profile 2>/dev/null || echo "")

    if [[ -n "$running" ]]; then
        if [[ "$running" == "$profile" ]]; then
            warn "Profile '${profile}' is already running!"
            echo ""
            cmd_status
            return 0
        fi

        warn "Profile '${running}' is currently running. Stopping it first..."
        cmd_stop
        echo ""
    fi

    # 如果目标端口被占用 (可能是不同 compose 项目的容器), 也尝试检测
    if is_port_in_use "$PORT" ; then
        local occupying
        occupying=$(get_vllm_service_on_port "$PORT" 2>/dev/null || echo "")
        if [[ -n "$occupying" ]]; then
            warn "Port ${PORT} occupied by '${occupying}'. Stopping..."
            stop_service_on_port "$PORT" "info"
        else
            error "Port ${PORT} is occupied by a non-vLLM process."
            if has_cmd lsof; then
                lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN || true
            fi
            exit 1
        fi
    fi

    info "Starting profile: ${BOLD}${profile}${NC} ..."
    "${COMPOSE[@]}" --profile "$profile" up -d

    local running_after
    running_after=$(get_running_profile 2>/dev/null || echo "")
    if [[ "$running_after" != "$profile" ]]; then
        error "Start verification failed: requested '${profile}', but running '${running_after:-none}'."
        cmd_status
        exit 1
    fi

    local service_port service_model_name service_api_key service_model_source
    service_port=$(get_running_port 2>/dev/null || echo "$PORT")
    service_model_name=$(get_service_env "$profile" "VLLM_SERVED_MODEL_NAME" 2>/dev/null || true)
    service_api_key=$(get_service_env "$profile" "VLLM_API_KEY" 2>/dev/null || true)
    service_model_source=$(get_service_env "$profile" "VLLM_MODEL" 2>/dev/null || true)

    service_model_name="${service_model_name:-${SERVED_MODEL_NAME:-$(vllm_model_name "$profile")}}"
    service_api_key="${service_api_key:-${API_KEY:-}}"
    service_model_source="${service_model_source:-unknown}"

    echo ""
    success "Model container started."
    echo ""
    echo -e "  ${CYAN}API Endpoint:${NC}  http://localhost:${service_port}/v1"
    echo -e "  ${CYAN}Model Name:${NC}    ${service_model_name}"
    echo -e "  ${CYAN}Model Source:${NC}  ${service_model_source}"
    echo -e "  ${CYAN}API Key:${NC}       ${service_api_key:-<not set>} ${YELLOW}(set API_KEY in .env to override)${NC}"
    echo ""
    echo -e "  ${BOLD}Quick test:${NC}"
    echo -e "  curl http://localhost:${service_port}/v1/chat/completions \\"
    echo -e "    -H 'Authorization: Bearer ${service_api_key:-<your-api-key>}' \\"
    echo -e "    -H 'Content-Type: application/json' \\"
    echo -e "    -d '{\"model\":\"${service_model_name}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'"
    echo ""
    info "Use '${BOLD}$0 logs${NC}' to watch startup progress."
}

cmd_stop() {
    local running_profiles=()
    local running_ports=()
    local profile
    local port
    mapfile -t running_profiles < <(get_running_profiles)
    mapfile -t running_ports < <(get_running_ports)

    if [[ ${#running_profiles[@]} -gt 0 ]]; then
        info "Stopping model containers: ${running_profiles[*]} ..."
        for profile in "${running_profiles[@]}"; do
            "${COMPOSE[@]}" stop --timeout "${STOP_TIMEOUT}" "$profile" 2>/dev/null || true
            "${COMPOSE[@]}" rm -f "$profile" 2>/dev/null || true
        done
    else
        info "No running managed model container found. Cleaning compose project state..."
    fi

    "${COMPOSE[@]}" down --remove-orphans --timeout "${STOP_TIMEOUT}" 2>/dev/null || true

    if [[ ${#running_ports[@]} -eq 0 ]]; then
        running_ports=("$PORT")
    fi

    for port in "${running_ports[@]}"; do
        if ! wait_for_port_free "$port"; then
            error "Stop finished but port ${port} is still occupied."
            if has_cmd lsof; then
                lsof -nP -iTCP:"${port}" -sTCP:LISTEN || true
            else
                docker ps --filter "publish=${port}" --format 'table {{.ID}}\t{{.Names}}\t{{.Ports}}' || true
            fi
            exit 1
        fi
    done

    success "Stopped. Managed ports are free."
}

cmd_restart() {
    local running
    running=$(get_running_profile 2>/dev/null || echo "")

    if [[ -z "$running" ]]; then
        error "No model is currently running. Use '$0 start <profile>' instead."
        exit 1
    fi

    info "Restarting profile: ${BOLD}${running}${NC} ..."
    "${COMPOSE[@]}" --profile "$running" restart
    success "Restarted."
}

cmd_status() {
    local running_profiles=()
    mapfile -t running_profiles < <(get_running_profiles)

    echo ""
    if [[ ${#running_profiles[@]} -eq 0 ]]; then
        echo -e "  ${YELLOW}●${NC} No model is currently running."
    else
        echo -e "  ${GREEN}●${NC} Running profile(s): ${BOLD}${running_profiles[*]}${NC}"
        echo ""
        "${COMPOSE[@]}" ps --status running
    fi

    echo ""
}

cmd_logs() {
    local running_profiles=()
    mapfile -t running_profiles < <(get_running_profiles)

    if [[ ${#running_profiles[@]} -eq 0 ]]; then
        error "No model is currently running."
        exit 1
    fi

    info "Showing logs for: ${BOLD}${running_profiles[*]}${NC} (Ctrl+C to exit)"
    "${COMPOSE[@]}" logs -f --tail 100 "${running_profiles[@]}"
}

cmd_switch() {
    local profile="${1:-}"

    if [[ -z "$profile" ]]; then
        error "请指定要切换到的模型 profile"
        cmd_list
        exit 1
    fi

    cmd_start "$profile"
}

cmd_build() {
    info "Building vLLM image..."
    "${COMPOSE[@]}" build
    success "Build complete."
}

cmd_help() {
    echo ""
    echo -e "${BOLD}vLLM Model Launcher${NC} - Gemma 4 Multi-GPU / Single-GPU"
    echo ""
    echo -e "${BOLD}Interactive Mode:${NC}"
    echo "  $0                    启动交互式 TUI"
    echo ""
    echo -e "${BOLD}Commands:${NC}"
    echo "  list                列出所有可用模型"
    echo "  start <profile>    启动指定模型 (自动停止当前运行的)"
    echo "  stop               停止当前运行的模型"
    echo "  switch <profile>   切换到另一个模型 (= start)"
    echo "  restart            重启当前运行的模型"
    echo "  status             查看当前运行状态"
    echo "  logs               查看实时日志"
    echo "  build              构建 Docker 镜像"
    echo "  bench [options]    运行性能 benchmark"

    echo "  models-md          输出 README 可用的 Markdown 模型表"
    echo "  sync-readme        自动更新 README 中的模型表"
    echo ""
    echo -e "${BOLD}GPU Selection:${NC}  ${CYAN}(单卡 profile 专用)${NC}"
    echo "  --gpu 0            使用 GPU 0, 端口 8000 (默认)"
    echo "  --gpu 1            使用 GPU 1, 端口 8001"
    echo ""
    echo -e "  ${YELLOW}规则:${NC} 双卡 profile (gemma26b) 忽略 --gpu"
    echo -e "  ${YELLOW}端口映射:${NC} GPU 0 → 8000, GPU 1 → 8001 (固定)"
    echo -e "  ${YELLOW}冲突处理:${NC} 目标端口有 vLLM 容器时自动替换"
    echo ""
    echo -e "${BOLD}Bench Options:${NC}"
    echo "  bench                           当前端口"
    echo "  bench --config-id gemma26b      指定配置标签"
    echo "  bench --base-url URL            指定服务地址"
    echo ""
    echo -e "${BOLD}Environment Variables:${NC}"
    echo "  GPU_MEMORY_UTILIZATION   GPU 显存利用率 (default: 0.93)"
    echo "  MAX_MODEL_LEN            最大上下文长度 (default: 65536)"
    echo "  MAX_NUM_SEQS             最大并发请求数 (default: 100)"
    echo "  API_KEY                  API 密钥 (default from .env/.env.example: abc123)"
    echo "  SERVED_MODEL_NAME        OpenAI 兼容 model 名称 (default from .env/.env.example: gemma)"
    echo "  STOP_TIMEOUT             停止端口释放超时秒数 (default: 20)"
    echo "  PORT                     服务端口 (default: 8000)"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  $0                              # 启动交互式 TUI"
    echo "  $0 start gemma26b              # 启动 Gemma-4-26B MoE (双卡, 默认 PP=2/TP=1)"
    echo "  $0 bench                       # 运行 benchmark"
    echo "  MAX_MODEL_LEN=8192 $0 start gemma26b  # 自定义上下文长度"
    echo ""
}

cmd_models_markdown() {
    vllm_models_markdown_table
}

cmd_sync_readme() {
    local readme_file="${VLLM_ROOT_DIR}/README.md"
    local start_marker="<!-- BEGIN:MODELS_TABLE -->"
    local end_marker="<!-- END:MODELS_TABLE -->"
    local tmp_file
    local marker_check

    if [[ ! -f "$readme_file" ]]; then
        error "README not found: ${readme_file}"
        exit 1
    fi

    marker_check=$(awk -v start="$start_marker" -v end="$end_marker" '
        $0 == start { found_start=1 }
        $0 == end { found_end=1 }
        END {
            if (found_start && found_end) {
                print "ok"
            }
        }
    ' "$readme_file")

    if [[ "$marker_check" != "ok" ]]; then
        error "README markers not found. Expected ${start_marker} and ${end_marker}."
        exit 1
    fi

    tmp_file=$(mktemp)
    awk \
        -v start="$start_marker" \
        -v end="$end_marker" \
        -v table="$(vllm_models_markdown_table)" '
        $0 == start {
            print
            print table
            in_block=1
            next
        }
        $0 == end {
            in_block=0
            print
            next
        }
        !in_block { print }
    ' "$readme_file" >"$tmp_file"

    if cmp -s "$readme_file" "$tmp_file"; then
        rm -f "$tmp_file"
        success "README model table is already up to date."
        return 0
    fi

    mv "$tmp_file" "$readme_file"
    success "README model table updated: ${readme_file}"
}

cmd_bench() {
    local bench_script="${VLLM_ROOT_DIR}/tests/performance/action_recognition_bench.py"
    local target_port
    if [[ ! -f "$bench_script" ]]; then
        error "Benchmark script not found: ${bench_script}"
        exit 1
    fi

    if ! command -v ffmpeg &>/dev/null; then
        error "ffmpeg is required for frame extraction. Install: apt install ffmpeg"
        exit 1
    fi

    target_port=$(get_running_port 2>/dev/null || echo "${PORT}")

    local default_url="http://127.0.0.1:${target_port}"
    info "Running action recognition benchmark..."
    python3 "$bench_script" run --base-url "$default_url" "$@"
}



run_cli_command() {
    local cmd="${1:-help}"

    case "$cmd" in
        start|stop|switch|restart|build)
            cli_ensure_prerequisites
            cli_with_lock
            ;;
        list|status|logs)
            cli_ensure_prerequisites
            ;;
    esac

    case "$cmd" in
        list)      cmd_list ;;
        start)     shift; cmd_start "$@" ;;
        stop)      cmd_stop ;;
        switch)    shift; cmd_start "$@" ;;
        restart)   cmd_restart ;;
        status)    cmd_status ;;
        logs)      cmd_logs ;;
        build)     cmd_build ;;
        bench)     shift; cmd_bench "$@" ;;

        models-md) cmd_models_markdown ;;
        sync-readme) cmd_sync_readme ;;
        help|--help|-h) cmd_help ;;
        *)
            error "Unknown command: ${1}"
            cmd_help
            exit 1
            ;;
    esac
}
