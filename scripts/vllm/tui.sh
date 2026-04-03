# TUI mode

ESC=$'\033'
CSI="${ESC}["

C_RESET="${CSI}0m"
C_BOLD="${CSI}1m"
C_DIM="${CSI}2m"
C_ITALIC="${CSI}3m"
C_UNDERLINE="${CSI}4m"

FG_BLACK="${CSI}30m"
FG_RED="${CSI}31m"
FG_GREEN="${CSI}32m"
FG_YELLOW="${CSI}33m"
FG_BLUE="${CSI}34m"
FG_MAGENTA="${CSI}35m"
FG_CYAN="${CSI}36m"
FG_WHITE="${CSI}37m"
FG_GRAY="${CSI}90m"

BG_BLACK="${CSI}40m"
BG_RED="${CSI}41m"
BG_GREEN="${CSI}42m"
BG_YELLOW="${CSI}43m"
BG_BLUE="${CSI}44m"
BG_MAGENTA="${CSI}45m"
BG_CYAN="${CSI}46m"
BG_WHITE="${CSI}47m"

BG_SELECTED="${CSI}48;5;236m"
FG_ACCENT="${CSI}38;5;75m"
FG_ACCENT2="${CSI}38;5;114m"
FG_ACCENT3="${CSI}38;5;215m"
FG_DIM_TEXT="${CSI}38;5;245m"

BOX_TL="╭" BOX_TR="╮" BOX_BL="╰" BOX_BR="╯"
BOX_H="─" BOX_V="│"
BOX_TL2="┌" BOX_TR2="┐" BOX_BL2="└" BOX_BR2="┘"
BOX_H2="─" BOX_V2="│"
BOX_CROSS="┼" BOX_T_DOWN="┬" BOX_T_UP="┴" BOX_T_LEFT="┤" BOX_T_RIGHT="├"

_term_width() { tput cols 2>/dev/null || echo 80; }
_term_height() { tput lines 2>/dev/null || echo 24; }

_cursor_hide()    { printf '%s' "${CSI}?25l"; }
_cursor_show()    { printf '%s' "${CSI}?25h"; }
_cursor_move()    { printf '%s' "${CSI}${1};${2}H"; }
_clear_screen()   { printf '%s' "${CSI}2J${CSI}H"; }
_clear_line()     { printf '%s' "${CSI}2K"; }
_save_cursor()    { printf '%s' "${ESC}7"; }
_restore_cursor() { printf '%s' "${ESC}8"; }

_term_save() {
    _orig_stty=$(stty -g 2>/dev/null || true)
    stty -echo -icanon min 1 time 0 2>/dev/null || true
    _cursor_hide
    _clear_screen
}

_term_restore() {
    _cursor_show
    _clear_screen
    if [[ -n "${_orig_stty:-}" ]]; then
        stty "${_orig_stty}" 2>/dev/null || true
    fi
}

_cleanup() {
    _term_restore
    exit 0
}

_read_key() {
    local key
    IFS= read -rsn1 key 2>/dev/null || true
    if [[ "$key" == $'\x1b' ]]; then
        local seq
        IFS= read -rsn1 -t 0.1 seq 2>/dev/null || true
        if [[ "$seq" == "[" ]]; then
            IFS= read -rsn1 -t 0.1 seq 2>/dev/null || true
            case "$seq" in
                A) echo "UP"; return ;;
                B) echo "DOWN"; return ;;
                C) echo "RIGHT"; return ;;
                D) echo "LEFT"; return ;;
            esac
        fi
        echo "ESC"
        return
    fi

    case "$key" in
        "") echo "ENTER" ;;
        q|Q) echo "QUIT" ;;
        *) echo "$key" ;;
    esac
}

_draw_hline() {
    local row=$1 col=$2 width=$3 char="${4:-$BOX_H}"
    _cursor_move "$row" "$col"
    printf '%s' "$(printf '%*s' "$width" '' | tr ' ' "$char")"
}

_draw_box() {
    local row=$1 col=$2 w=$3 h=$4 title="${5:-}"
    local inner_w=$((w - 2))
    local r

    _cursor_move "$row" "$col"
    printf '%s%s%s%s%s' "${FG_ACCENT}${C_DIM}" "$BOX_TL" "$(printf '%*s' "$inner_w" '' | tr ' ' "$BOX_H")" "$BOX_TR" "$C_RESET"

    if [[ -n "$title" ]]; then
        local title_clean
        title_clean=$(echo -e "$title" | sed 's/\x1b\[[0-9;]*m//g')
        local tlen=${#title_clean}
        local tpos=$(( col + (w - tlen - 2) / 2 ))
        _cursor_move "$row" "$tpos"
        printf '%s %s %s' "${FG_ACCENT}${C_DIM}" "${C_BOLD}${FG_ACCENT}${title}${C_RESET}" "${FG_ACCENT}${C_DIM}"
        printf '%s' "$C_RESET"
    fi

    for (( r = row + 1; r < row + h - 1; r++ )); do
        _cursor_move "$r" "$col"
        printf '%s%s%s' "${FG_ACCENT}${C_DIM}${BOX_V}${C_RESET}" "$(printf '%*s' "$inner_w" '')" "${FG_ACCENT}${C_DIM}${BOX_V}${C_RESET}"
    done

    _cursor_move "$((row + h - 1))" "$col"
    printf '%s%s%s%s%s' "${FG_ACCENT}${C_DIM}" "$BOX_BL" "$(printf '%*s' "$inner_w" '' | tr ' ' "$BOX_H")" "$BOX_BR" "$C_RESET"
}

_print_center() {
    local row=$1 text="$2"
    local w
    w=$(_term_width)
    local text_clean
    text_clean=$(echo -e "$text" | sed 's/\x1b\[[0-9;]*m//g')
    local tlen=${#text_clean}
    local col=$(( (w - tlen) / 2 ))
    [[ $col -lt 1 ]] && col=1
    _cursor_move "$row" "$col"
    printf '%b' "$text"
}

_print_at() {
    local row=$1 col=$2
    shift 2
    _cursor_move "$row" "$col"
    printf '%b' "$*"
}

_draw_header() {
    local row=1

    _print_center "$row" "${C_BOLD}${FG_ACCENT}╔══════════════════════════════════════════╗${C_RESET}"
    row=$((row + 1))
    _print_center "$row" "${C_BOLD}${FG_ACCENT}║${C_RESET}  ${C_BOLD}🤖 vLLM Model Launcher${C_RESET}                  ${C_BOLD}${FG_ACCENT}║${C_RESET}"
    row=$((row + 1))
    _print_center "$row" "${C_BOLD}${FG_ACCENT}║${C_RESET}  ${FG_DIM_TEXT}Qwen 3.5 Multi-GPU Management${C_RESET}          ${C_BOLD}${FG_ACCENT}║${C_RESET}"
    row=$((row + 1))
    _print_center "$row" "${C_BOLD}${FG_ACCENT}╚══════════════════════════════════════════╝${C_RESET}"
}

_draw_status_bar() {
    local row=$1
    local running
    running=$(get_running_profile 2>/dev/null || echo "")

    if [[ -n "$running" ]]; then
        local model_info=""
        if vllm_model_exists "$running"; then
            model_info="$(vllm_model_icon "$running") $(vllm_model_name "$running") ($(vllm_model_gpu "$running"))"
        fi

        _print_center "$row" "${C_BOLD}${FG_GREEN}● RUNNING${C_RESET}  ${C_BOLD}${running}${C_RESET}  ${FG_DIM_TEXT}${model_info}${C_RESET}  ${FG_DIM_TEXT}Port: ${PORT}${C_RESET}"
    else
        _print_center "$row" "${FG_YELLOW}${C_BOLD}◯ IDLE${C_RESET}  ${FG_DIM_TEXT}No model is currently running${C_RESET}"
    fi
}

_draw_footer() {
    local h
    h=$(_term_height)
    _print_center "$((h - 1))" "${FG_DIM_TEXT}↑↓${C_RESET} Navigate  ${FG_DIM_TEXT}Enter${C_RESET} Select  ${FG_DIM_TEXT}q${C_RESET} Quit"
}

_tui_message() {
    local msg="$1" color="${2:-$FG_CYAN}" wait_key="${3:-true}"
    local h w
    h=$(_term_height)
    w=$(_term_width)

    local msg_clean
    msg_clean=$(echo -e "$msg" | sed 's/\x1b\[[0-9;]*m//g')
    local mlen=${#msg_clean}
    local box_w=$((mlen + 6))
    [[ $box_w -lt 40 ]] && box_w=40
    local box_h=5
    local start_row=$(( (h - box_h) / 2 ))
    local start_col=$(( (w - box_w) / 2 ))

    _draw_box "$start_row" "$start_col" "$box_w" "$box_h"
    _print_at "$((start_row + 2))" "$((start_col + 3))" "${color}${msg}${C_RESET}"

    if [[ "$wait_key" == "true" ]]; then
        _print_at "$((start_row + box_h))" "$((start_col + 2))" "${FG_DIM_TEXT}Press any key to continue...${C_RESET}"
        _read_key >/dev/null
    fi
}

_tui_error() {
    _tui_message "✖ $1" "$FG_RED" "true"
}

_tui_success() {
    _tui_message "✔ $1" "$FG_GREEN" "true"
}

_run_with_progress() {
    local msg="$1"
    shift
    local h w
    h=$(_term_height)
    w=$(_term_width)

    local box_w=56
    local box_h=5
    local start_row=$(( (h - box_h) / 2 ))
    local start_col=$(( (w - box_w) / 2 ))

    _draw_box "$start_row" "$start_col" "$box_w" "$box_h"

    local spinner_chars=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
    local spin_idx=0

    "$@" >/dev/null 2>&1 &
    local cmd_pid=$!

    while kill -0 "$cmd_pid" 2>/dev/null; do
        _print_at "$((start_row + 2))" "$((start_col + 3))" "${FG_ACCENT}${spinner_chars[$spin_idx]}${C_RESET} ${C_BOLD}${msg}...${C_RESET}    "
        spin_idx=$(( (spin_idx + 1) % ${#spinner_chars[@]} ))
        sleep 0.1
    done

    wait "$cmd_pid"
}

MAIN_MENU_ITEMS=(
    "🚀  Start Model"
    "🛑  Stop Model"
    "🔄  Restart Model"
    "📊  Status"
    "📋  View Logs"
    "📦  Model List"
    "🔧  Build Image"
    "❓  Help"
    "🚪  Quit"
)

MAIN_MENU_DESCS=(
    "Launch a model profile"
    "Stop the running model"
    "Restart the current model"
    "View the current status"
    "Show live container logs"
    "List all available models"
    "Build the Docker image"
    "Show usage information"
    "Exit the TUI"
)

_draw_main_menu() {
    local selected=$1
    local w
    w=$(_term_width)

    _clear_screen
    _draw_header
    _draw_status_bar 7

    local menu_h=$(( ${#MAIN_MENU_ITEMS[@]} + 2 ))
    local menu_w=52
    local menu_row=9
    local menu_col=$(( (w - menu_w) / 2 ))

    _draw_box "$menu_row" "$menu_col" "$menu_w" "$menu_h" "Main Menu"

    local i
    for (( i = 0; i < ${#MAIN_MENU_ITEMS[@]}; i++ )); do
        local item_row=$((menu_row + 1 + i))
        local item="${MAIN_MENU_ITEMS[$i]}"
        local desc="${MAIN_MENU_DESCS[$i]}"

        _cursor_move "$item_row" "$((menu_col + 2))"
        if [[ $i -eq $selected ]]; then
            printf '%s' "${BG_SELECTED}${C_BOLD}${FG_ACCENT}"
            printf ' ▸ %-30s' "$item"
            printf '%s' "${FG_DIM_TEXT}"
            printf '%-14s' "$desc"
            printf '%s' "$C_RESET"
        else
            printf '   '
            printf '%s' "$C_RESET"
            printf '%-30s' "$item"
            printf '%s' "${FG_DIM_TEXT}"
            printf '%-14s' "$desc"
            printf '%s' "$C_RESET"
        fi
    done

    _draw_footer
}

_screen_main_menu() {
    local selected=0
    local num_items=${#MAIN_MENU_ITEMS[@]}

    while true; do
        _draw_main_menu "$selected"

        local key
        key=$(_read_key)
        case "$key" in
            UP)
                selected=$(( (selected - 1 + num_items) % num_items ))
                ;;
            DOWN)
                selected=$(( (selected + 1) % num_items ))
                ;;
            ENTER)
                case $selected in
                    0) _screen_model_select "start" ;;
                    1) _action_stop ;;
                    2) _action_restart ;;
                    3) _screen_status ;;
                    4) _action_logs ;;
                    5) _screen_model_list ;;
                    6) _action_build ;;
                    7) _screen_help ;;
                    8) return 0 ;;
                esac
                ;;
            QUIT|ESC)
                return 0
                ;;
        esac
    done
}

_draw_model_select() {
    local selected=$1 action="$2"
    local w
    w=$(_term_width)

    _clear_screen
    _draw_header

    local running
    running=$(get_running_profile 2>/dev/null || echo "")

    local title="Select Model to ${action^}"
    local box_w=62
    local box_h=$(( ${#ALL_PROFILES[@]} * 3 + 4 ))
    local box_row=7
    local box_col=$(( (w - box_w) / 2 ))

    _draw_box "$box_row" "$box_col" "$box_w" "$box_h" "$title"

    local row=$((box_row + 2))
    local i
    for (( i = 0; i < ${#ALL_PROFILES[@]}; i++ )); do
        local profile="${ALL_PROFILES[$i]}"
        local model
        local ctx
        local gpu
        local desc
        local icon
        model="$(vllm_model_name "$profile")"
        ctx="$(vllm_model_ctx "$profile")"
        gpu="$(vllm_model_gpu "$profile")"
        desc="$(vllm_model_desc "$profile")"
        icon="$(vllm_model_icon "$profile")"

        local status_tag=""
        if [[ "$profile" == "$running" ]]; then
            status_tag="${FG_GREEN}● RUNNING${C_RESET}"
        fi

        _cursor_move "$row" "$((box_col + 2))"
        if [[ $i -eq $selected ]]; then
            printf '%s' "${BG_SELECTED}${C_BOLD}"
            printf ' ▸ %s %-15s  ' "$icon" "$profile"
            printf '%s' "${FG_ACCENT}"
            printf '%-22s' "$model"
            printf '%s%s' "$C_RESET" "${BG_SELECTED}"
            printf ' %s ' "$status_tag"
            printf '%*s' $((box_w - 47 - ${#status_tag} + ${#status_tag})) ''
            printf '%s' "$C_RESET"

            _cursor_move "$((row + 1))" "$((box_col + 2))"
            printf '%s' "${BG_SELECTED}${FG_DIM_TEXT}"
            printf '      %s  │  ctx: %s  │  %s' "$desc" "$ctx" "$gpu"
            printf '%*s' $((box_w - 50)) ''
            printf '%s' "$C_RESET"
        else
            printf '   %s %-15s  ' "$icon" "$profile"
            printf '%s' "${FG_DIM_TEXT}"
            printf '%-22s' "$model"
            printf ' %s' "$status_tag"
            printf '%s' "$C_RESET"

            _cursor_move "$((row + 1))" "$((box_col + 2))"
            printf '%s' "${FG_DIM_TEXT}"
            printf '      %s  │  ctx: %s  │  %s' "$desc" "$ctx" "$gpu"
            printf '%s' "$C_RESET"
        fi

        row=$((row + 3))
    done

    _draw_footer
    _print_center "$((_term_height - 2))" "${FG_DIM_TEXT}ESC${C_RESET} Back to Menu"
}

_screen_model_select() {
    local action="${1:-start}"
    local selected=0
    local num_items=${#ALL_PROFILES[@]}

    while true; do
        _draw_model_select "$selected" "$action"

        local key
        key=$(_read_key)
        case "$key" in
            UP)
                selected=$(( (selected - 1 + num_items) % num_items ))
                ;;
            DOWN)
                selected=$(( (selected + 1) % num_items ))
                ;;
            ENTER)
                local profile="${ALL_PROFILES[$selected]}"
                _action_start "$profile"
                return
                ;;
            QUIT|ESC)
                return
                ;;
        esac
    done
}

_screen_model_list() {
    local w
    w=$(_term_width)

    _clear_screen
    _draw_header

    local running
    running=$(get_running_profile 2>/dev/null || echo "")

    local box_w=72
    local box_h=$(( ${#ALL_PROFILES[@]} + 5 ))
    local box_row=7
    local box_col=$(( (w - box_w) / 2 ))

    _draw_box "$box_row" "$box_col" "$box_w" "$box_h" "📦 Available Models"

    local hr=$((box_row + 1))
    _print_at "$hr" "$((box_col + 2))" "${C_BOLD}${FG_ACCENT}  PROFILE          MODEL                    CTX    GPU    STATUS${C_RESET}"
    hr=$((hr + 1))
    _print_at "$hr" "$((box_col + 2))" "${FG_DIM_TEXT}  $(printf '%*s' 66 '' | tr ' ' '─')${C_RESET}"
    hr=$((hr + 1))

    local i
    for (( i = 0; i < ${#ALL_PROFILES[@]}; i++ )); do
        local profile="${ALL_PROFILES[$i]}"
        local model
        local ctx
        local gpu
        local icon
        model="$(vllm_model_name "$profile")"
        ctx="$(vllm_model_ctx "$profile")"
        gpu="$(vllm_model_gpu "$profile")"
        icon="$(vllm_model_icon "$profile")"

        local status_str="${FG_DIM_TEXT}──${C_RESET}"
        if [[ "$profile" == "$running" ]]; then
            status_str="${FG_GREEN}${C_BOLD}● RUNNING${C_RESET}"
        fi

        _cursor_move "$hr" "$((box_col + 2))"
        printf '  %s %-15s %-24s %-6s %-6s %b' "$icon" "$profile" "$model" "$ctx" "${gpu%GPU}" "$status_str"
        hr=$((hr + 1))
    done

    _draw_footer
    _print_center "$((_term_height - 2))" "${FG_DIM_TEXT}Press any key to go back${C_RESET}"
    _read_key >/dev/null
}

_screen_status() {
    local w
    w=$(_term_width)

    _clear_screen
    _draw_header

    local running
    running=$(get_running_profile 2>/dev/null || echo "")

    local box_w=60
    local box_h=12
    local box_row=7
    local box_col=$(( (w - box_w) / 2 ))

    _draw_box "$box_row" "$box_col" "$box_w" "$box_h" "📊 Current Status"

    local r=$((box_row + 2))

    if [[ -z "$running" ]]; then
        _print_at "$r" "$((box_col + 4))" "${FG_YELLOW}${C_BOLD}◯${C_RESET}  No model is currently running."
        r=$((r + 2))
        _print_at "$r" "$((box_col + 4))" "${FG_DIM_TEXT}Use ${C_BOLD}Start Model${C_RESET}${FG_DIM_TEXT} from the main menu to launch one.${C_RESET}"
    else
        _print_at "$r" "$((box_col + 4))" "${FG_GREEN}${C_BOLD}●${C_RESET}  ${C_BOLD}Model Active${C_RESET}"
        r=$((r + 2))
        _print_at "$r" "$((box_col + 6))" "${FG_DIM_TEXT}Profile:${C_RESET}   ${C_BOLD}${running}${C_RESET}"
        r=$((r + 1))
        if vllm_model_exists "$running"; then
            _print_at "$r" "$((box_col + 6))" "${FG_DIM_TEXT}Model:${C_RESET}     $(vllm_model_name "$running")"
            r=$((r + 1))
            _print_at "$r" "$((box_col + 6))" "${FG_DIM_TEXT}GPU:${C_RESET}       $(vllm_model_gpu "$running")"
            r=$((r + 1))
            _print_at "$r" "$((box_col + 6))" "${FG_DIM_TEXT}Context:${C_RESET}   $(vllm_model_ctx "$running")"
            r=$((r + 1))
        fi
        _print_at "$r" "$((box_col + 6))" "${FG_DIM_TEXT}Endpoint:${C_RESET}  http://localhost:${PORT}/v1"
        r=$((r + 1))
        _print_at "$r" "$((box_col + 6))" "${FG_DIM_TEXT}API Key:${C_RESET}   ${API_KEY:-abc123}"
    fi

    _draw_footer
    _print_center "$((_term_height - 2))" "${FG_DIM_TEXT}Press any key to go back${C_RESET}"
    _read_key >/dev/null
}

_screen_help() {
    local w
    w=$(_term_width)

    _clear_screen
    _draw_header

    local box_w=64
    local box_h=19
    local box_row=7
    local box_col=$(( (w - box_w) / 2 ))

    _draw_box "$box_row" "$box_col" "$box_w" "$box_h" "❓ Help & Usage"

    local r=$((box_row + 2))

    _print_at "$r" "$((box_col + 3))" "${C_BOLD}${FG_ACCENT}CLI Mode (non-interactive):${C_RESET}"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 5))" "${FG_DIM_TEXT}./vllm.sh start <profile>   Launch a model${C_RESET}"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 5))" "${FG_DIM_TEXT}./vllm.sh stop              Stop the model${C_RESET}"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 5))" "${FG_DIM_TEXT}./vllm.sh restart           Restart model${C_RESET}"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 5))" "${FG_DIM_TEXT}./vllm.sh status            Show status${C_RESET}"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 5))" "${FG_DIM_TEXT}./vllm.sh logs              View logs${C_RESET}"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 5))" "${FG_DIM_TEXT}./vllm.sh list              List models${C_RESET}"
    r=$((r + 2))
    _print_at "$r" "$((box_col + 3))" "${C_BOLD}${FG_ACCENT}Environment Variables:${C_RESET}"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 5))" "${FG_DIM_TEXT}GPU_MEMORY_UTILIZATION   GPU 显存利用率 (0.98)${C_RESET}"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 5))" "${FG_DIM_TEXT}MAX_MODEL_LEN            最大上下文 (262144)${C_RESET}"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 5))" "${FG_DIM_TEXT}MAX_NUM_SEQS             最大并发数 (1000)${C_RESET}"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 5))" "${FG_DIM_TEXT}API_KEY                  API 密钥 (abc123)${C_RESET}"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 5))" "${FG_DIM_TEXT}PORT                     服务端口 (8000)${C_RESET}"

    _draw_footer
    _print_center "$((_term_height - 2))" "${FG_DIM_TEXT}Press any key to go back${C_RESET}"
    _read_key >/dev/null
}

_action_start() {
    local profile="$1"

    if ! ensure_prerequisites; then return; fi
    if ! with_lock; then return; fi

    local running
    running=$(get_running_profile 2>/dev/null || echo "")

    if [[ -n "$running" ]]; then
        if [[ "$running" == "$profile" ]]; then
            _tui_message "⚠ Profile '${profile}' is already running!" "$FG_YELLOW"
            return
        fi

        _run_with_progress "Stopping ${running}" "${COMPOSE[@]}" down --remove-orphans --timeout "${STOP_TIMEOUT}"
        wait_for_port_free || true
    fi

    _run_with_progress "Starting ${profile}" "${COMPOSE[@]}" --profile "$profile" up -d

    local running_after
    running_after=$(get_running_profile 2>/dev/null || echo "")
    if [[ "$running_after" != "$profile" ]]; then
        _tui_error "Start failed: expected '${profile}', got '${running_after:-none}'"
        return
    fi

    _clear_screen
    _draw_header

    local w
    w=$(_term_width)
    local box_w=58
    local box_h=13
    local box_row=7
    local box_col=$(( (w - box_w) / 2 ))

    _draw_box "$box_row" "$box_col" "$box_w" "$box_h" "✔ Model Started"

    local r=$((box_row + 2))
    _print_at "$r" "$((box_col + 4))" "${FG_GREEN}${C_BOLD}✔${C_RESET}  ${C_BOLD}${profile}${C_RESET} launched successfully!"
    r=$((r + 2))
    _print_at "$r" "$((box_col + 6))" "${FG_DIM_TEXT}Endpoint:${C_RESET}  http://localhost:${PORT}/v1"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 6))" "${FG_DIM_TEXT}Model:${C_RESET}     qwen"
    r=$((r + 1))
    _print_at "$r" "$((box_col + 6))" "${FG_DIM_TEXT}API Key:${C_RESET}   ${API_KEY:-abc123}"
    r=$((r + 2))
    _print_at "$r" "$((box_col + 4))" "${FG_DIM_TEXT}Use ${C_BOLD}View Logs${C_RESET}${FG_DIM_TEXT} to watch startup progress.${C_RESET}"

    _print_center "$((_term_height - 2))" "${FG_DIM_TEXT}Press any key to continue${C_RESET}"
    _read_key >/dev/null
}

_action_stop() {
    if ! ensure_prerequisites; then return; fi
    if ! with_lock; then return; fi

    local running
    running=$(get_running_profile 2>/dev/null || echo "")

    if [[ -z "$running" ]]; then
        _tui_message "◯ No model is running." "$FG_YELLOW"
        return
    fi

    _clear_screen
    _draw_header
    _run_with_progress "Stopping ${running}" "${COMPOSE[@]}" down --remove-orphans --timeout "${STOP_TIMEOUT}"

    if ! wait_for_port_free; then
        _tui_error "Port ${PORT} is still occupied after stop."
        return
    fi

    _tui_success "Stopped. Port ${PORT} is free."
}

_action_restart() {
    if ! ensure_prerequisites; then return; fi
    if ! with_lock; then return; fi

    local running
    running=$(get_running_profile 2>/dev/null || echo "")

    if [[ -z "$running" ]]; then
        _tui_error "No model is running. Use Start Model instead."
        return
    fi

    _clear_screen
    _draw_header
    _run_with_progress "Restarting ${running}" "${COMPOSE[@]}" --profile "$running" restart

    _tui_success "Restarted ${running} successfully."
}

_action_build() {
    if ! ensure_prerequisites; then return; fi
    if ! with_lock; then return; fi

    _clear_screen
    _draw_header
    _run_with_progress "Building vLLM image" "${COMPOSE[@]}" build

    _tui_success "Build complete."
}

_action_logs() {
    local running
    running=$(get_running_profile 2>/dev/null || echo "")

    if [[ -z "$running" ]]; then
        _tui_error "No model is running."
        return
    fi

    _term_restore

    echo ""
    echo -e "${FG_CYAN}${C_BOLD}▸ Showing logs for: ${running}${C_RESET} (Ctrl+C to exit)"
    echo ""
    "${COMPOSE[@]}" --profile "$running" logs -f --tail 100 || true

    _term_save
}

run_tui() {
    ensure_prerequisites || exit 1

    trap _cleanup INT TERM EXIT
    _term_save
    _screen_main_menu
}
