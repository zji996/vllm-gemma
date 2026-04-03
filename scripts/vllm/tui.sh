# TUI mode — 简洁交互式菜单

run_tui() {
    cli_ensure_prerequisites

    local running

    while true; do
        running=$(get_running_profile 2>/dev/null || echo "")

        echo ""
        echo -e "${BOLD}🤖 vLLM Model Launcher — Gemma 4${NC}"
        echo -e "────────────────────────────────────"

        if [[ -n "$running" ]]; then
            echo -e "  状态: ${GREEN}● RUNNING${NC}  ${BOLD}${running}${NC}  (port ${PORT})"
        else
            echo -e "  状态: ${YELLOW}◯ IDLE${NC}  无模型运行"
        fi

        echo ""
        echo -e "  ${BOLD}1)${NC} start    启动/切换模型"
        echo -e "  ${BOLD}2)${NC} stop     停止当前模型"
        echo -e "  ${BOLD}3)${NC} status   查看状态"
        echo -e "  ${BOLD}4)${NC} logs     查看日志"
        echo -e "  ${BOLD}5)${NC} list     列出所有模型"
        echo -e "  ${BOLD}6)${NC} build    构建镜像"
        echo -e "  ${BOLD}q)${NC} quit     退出"
        echo ""
        echo -n "请选择> "
        read -r choice

        case "$choice" in
            1|start)
                echo ""
                echo -e "${BOLD}可用模型:${NC}"
                local i profile
                for i in "${!ALL_PROFILES[@]}"; do
                    profile="${ALL_PROFILES[$i]}"
                    local status_tag=""
                    [[ "$profile" == "$running" ]] && status_tag=" ${GREEN}● RUNNING${NC}"
                    echo -e "  ${BOLD}$((i+1)))${NC} ${profile}  $(vllm_model_icon "$profile") $(vllm_model_name "$profile")  $(vllm_model_ctx "$profile")  $(vllm_model_gpu "$profile")${status_tag}"
                done
                echo ""
                echo -n "选择模型 (序号或名称)> "
                read -r model_choice

                # 支持序号或名称
                local target=""
                if [[ "$model_choice" =~ ^[0-9]+$ ]] && (( model_choice >= 1 && model_choice <= ${#ALL_PROFILES[@]} )); then
                    target="${ALL_PROFILES[$((model_choice-1))]}"
                elif validate_profile "$model_choice"; then
                    target="$model_choice"
                else
                    error "无效选择: ${model_choice}"
                    continue
                fi

                cli_with_lock
                cmd_start "$target"
                ;;
            2|stop)
                cli_with_lock
                cmd_stop
                ;;
            3|status)
                cmd_status
                ;;
            4|logs)
                cmd_logs
                ;;
            5|list)
                cmd_list
                ;;
            6|build)
                cli_with_lock
                cmd_build
                ;;
            q|quit|exit)
                echo ""
                echo -e "${GREEN}Bye!${NC}"
                break
                ;;
            "")
                continue
                ;;
            *)
                error "无效选择: ${choice}"
                ;;
        esac
    done
}
