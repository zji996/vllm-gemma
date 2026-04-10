has_cmd() { command -v "$1" >/dev/null 2>&1; }

_report_failure() {
    local handler="${1:-}"
    local message="${2:-}"
    if [[ -n "$handler" ]]; then
        "$handler" "$message"
    else
        printf '%s\n' "$message" >&2
    fi
}

_check_prerequisites() {
    local on_error="${1:-}"

    if [[ ! -f "${COMPOSE_FILE}" ]]; then
        _report_failure "$on_error" "docker-compose.yml not found: ${COMPOSE_FILE}"
        return 1
    fi
    if ! has_cmd docker; then
        _report_failure "$on_error" "docker command not found."
        return 1
    fi
    if ! docker compose version >/dev/null 2>&1; then
        _report_failure "$on_error" "docker compose plugin is not available."
        return 1
    fi
    if ! check_profile_registry_sync "$on_error"; then
        return 1
    fi
    return 0
}

_acquire_lock() {
    local on_error="${1:-}"

    if ! has_cmd flock; then
        _report_failure "$on_error" "flock command not found. Please install util-linux."
        return 1
    fi

    exec 9>"${LOCK_FILE}"
    if ! flock -n 9; then
        _report_failure "$on_error" "Another vllm.sh command is running. Please retry in a moment."
        return 1
    fi
    return 0
}

ensure_prerequisites() {
    _check_prerequisites "_tui_error"
}

with_lock() {
    _acquire_lock "_tui_error"
}

cli_ensure_prerequisites() {
    _check_prerequisites "error" || exit 1
}

cli_with_lock() {
    _acquire_lock "error" || exit 1
}

validate_profile() {
    local profile="${1:-}"
    [[ -n "$profile" ]] || return 1
    vllm_model_exists "$profile"
}

check_profile_registry_sync() {
    local on_error="${1:-}"
    local compose_profiles=()
    local compose_sorted=""
    local registry_sorted=""
    local compose_list=""
    local registry_list=""

    mapfile -t compose_profiles < <("${COMPOSE[@]}" config --profiles 2>/dev/null | sed '/^$/d')

    if [[ ${#compose_profiles[@]} -eq 0 ]]; then
        _report_failure "$on_error" "Unable to read profiles from docker-compose.yml."
        return 1
    fi

    compose_sorted=$(printf '%s\n' "${compose_profiles[@]}" | LC_ALL=C sort)
    registry_sorted=$(printf '%s\n' "${ALL_PROFILES[@]}" | LC_ALL=C sort)

    if [[ "$compose_sorted" != "$registry_sorted" ]]; then
        compose_list=$(printf '%s, ' "${compose_profiles[@]}")
        registry_list=$(printf '%s, ' "${ALL_PROFILES[@]}")
        compose_list="${compose_list%, }"
        registry_list="${registry_list%, }"
        _report_failure "$on_error" "Model registry and docker-compose profiles are out of sync. config: [${registry_list}] compose: [${compose_list}]"
        return 1
    fi
}

get_running_profiles() {
    local running_services=()
    local service
    mapfile -t running_services < <("${COMPOSE[@]}" ps --status running --services 2>/dev/null || true)
    for service in "${running_services[@]}"; do
        if validate_profile "$service"; then
            echo "$service"
        fi
    done
}

get_running_profile() {
    local matched_profiles=()
    mapfile -t matched_profiles < <(get_running_profiles)
    if [[ ${#matched_profiles[@]} -eq 0 ]]; then
        return 1
    fi
    echo "${matched_profiles[0]}"
    return 0
}

is_port_in_use() {
    local port="${1}"
    if has_cmd ss; then
        if ss -tln 2>/dev/null | grep -q ":${port} "; then return 0; fi
    elif has_cmd lsof; then
        if lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then return 0; fi
    fi
    if docker ps --format '{{.Ports}}' 2>/dev/null | grep -qE "(0\.0\.0\.0|:::):${port}->"; then
        return 0
    fi
    return 1
}

wait_for_port_free() {
    local port="${1:-$PORT}"
    local max_wait="${STOP_TIMEOUT}"
    local waited=0
    while is_port_in_use "$port"; do
        if [[ $waited -ge $max_wait ]]; then return 1; fi
        sleep 1
        waited=$((waited + 1))
    done
}

# GPU → 端口映射: GPU 0 → 8000, GPU 1 → 8001
gpu_to_port() {
    local gpu_id="${1:?gpu_id is required}"
    echo $(( 8000 + gpu_id ))
}

# 从 docker ps 的端口字段中提取宿主机端口
extract_published_port() {
    local ports="${1:-}"
    local port
    port=$(printf '%s\n' "$ports" | sed -nE 's/.*:([0-9]+)->.*/\1/p' | head -n 1)
    if [[ -n "$port" ]]; then
        echo "$port"
        return 0
    fi
    return 1
}

get_running_ports() {
    local service_ports=()
    local line
    local service
    local ports
    local port
    local -A seen=()

    mapfile -t service_ports < <("${COMPOSE[@]}" ps --status running --format '{{.Service}}\t{{.Ports}}' 2>/dev/null || true)
    for line in "${service_ports[@]}"; do
        IFS=$'\t' read -r service ports <<<"$line"
        if ! validate_profile "$service"; then
            continue
        fi
        port=$(extract_published_port "$ports" 2>/dev/null || true)
        if [[ -n "$port" && -z "${seen[$port]+x}" ]]; then
            seen["$port"]=1
            echo "$port"
        fi
    done
}

get_running_port() {
    local ports=()
    mapfile -t ports < <(get_running_ports)
    if [[ ${#ports[@]} -eq 0 ]]; then
        return 1
    fi
    echo "${ports[0]}"
}

get_service_container_id() {
    local service="${1:?service is required}"
    local container_id
    container_id=$("${COMPOSE[@]}" ps -q "$service" 2>/dev/null | head -n 1)
    if [[ -n "$container_id" ]]; then
        printf '%s\n' "$container_id"
        return 0
    fi
    return 1
}

get_container_env() {
    local container_id="${1:?container_id is required}"
    local env_name="${2:?env_name is required}"

    docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container_id" 2>/dev/null \
        | sed -n "s/^${env_name}=//p" \
        | head -n 1
}

get_service_env() {
    local service="${1:?service is required}"
    local env_name="${2:?env_name is required}"
    local container_id

    container_id=$(get_service_container_id "$service") || return 1
    get_container_env "$container_id" "$env_name"
}

get_vllm_container_on_port() {
    local port="${1:?port is required}"
    local containers=()
    local line
    local container_id
    local container_name
    local compose_service
    local candidate_service

    mapfile -t containers < <(docker ps --filter "publish=${port}" --format '{{.ID}}\t{{.Names}}\t{{.Label "com.docker.compose.service"}}\t{{.Ports}}' 2>/dev/null || true)
    for line in "${containers[@]}"; do
        IFS=$'\t' read -r container_id container_name compose_service _ <<<"$line"

        candidate_service="$compose_service"
        if [[ -z "$candidate_service" && "$container_name" =~ ^vllm-(.+)$ ]]; then
            candidate_service="${BASH_REMATCH[1]}"
        fi

        if validate_profile "$candidate_service"; then
            printf '%s\t%s\n' "$container_id" "$candidate_service"
            return 0
        fi
    done
    return 1
}

get_vllm_service_on_port() {
    local port="${1:?port is required}"
    local container_ref
    local service

    container_ref=$(get_vllm_container_on_port "$port") || return 1
    IFS=$'\t' read -r _ service <<<"$container_ref"
    printf '%s\n' "$service"
}

# 停止占用指定端口的 vllm 容器
stop_service_on_port() {
    local port="${1:?port is required}"
    local on_error="${2:-}"
    local container_ref
    local container_id
    local service

    container_ref=$(get_vllm_container_on_port "$port") || return 0
    IFS=$'\t' read -r container_id service <<<"$container_ref"

    if [[ -n "$on_error" ]]; then
        "$on_error" "Stopping '${service}' on port ${port}..."
    fi

    docker stop --time "${STOP_TIMEOUT}" "${container_id}" >/dev/null 2>&1 || true
    docker rm -f "${container_id}" >/dev/null 2>&1 || true
    wait_for_port_free "$port" || return 1
}
