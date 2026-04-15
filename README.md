# vLLM Gemma 4 Launcher

在 `2× RTX 3080 20GB` 上部署、调优和验证 `Gemma 4` 的本地集成仓库。当前主力配置是 `gemma-4-26B-A4B-it`，默认走 `PP=2 / TP=1`，并带有一组针对 Gemma 4 的本地补丁：

- launcher-managed `chat_template.jinja` patch
- Gemma 4 reasoning parser patch
- vLLM reasoning budget / repetition guard patch
- ModelScope 离线启动与本地快照同步链路

## 当前默认行为

- 默认请求是 `non-thinking`
- `reasoning_effort=none|low` 保持 `non-thinking`
- `reasoning_effort=medium|high` 才进入 `thinking`
- 服务默认开启 `--reasoning-parser gemma4`
- 服务默认提供 `--reasoning-config`，以支持 vLLM 的 `thinking_token_budget`

如果你只想先跑起来，直接看下面的 Quick Start。

## Quick Start

### 1. 构建镜像

```bash
docker compose build gemma26b
```

### 2. 启动默认模型

```bash
./vllm.sh start gemma26b
```

启动脚本会自动做三件事：

1. 根据 `MS_GEMMA26B_MODEL_ID` 解析本地或远端 ModelScope 模型
2. 对本地 `chat_template.jinja` 应用 launcher-managed patch
3. 将容器中的 `VLLM_MODEL` 指向挂载后的本地快照目录，离线启动 vLLM

### 3. 常用命令

```bash
./vllm.sh list
./vllm.sh start gemma26b
./vllm.sh restart
./vllm.sh stop
./vllm.sh status
./vllm.sh logs
./vllm.sh build
```

## 模型配置

<!-- BEGIN:MODELS_TABLE -->
| Profile | 模型 | 上下文 | GPU | 说明 |
|---------|------|--------|-----|------|
| `gemma26b` | Gemma-4-26B-A4B-it | 64K | 2× | ⭐ 默认: PP=2 / TP=1 / Gemma4 patch |
<!-- END:MODELS_TABLE -->

如后续 profile 有变化，可以运行：

```bash
./vllm.sh sync-readme
```

## 模型下载与离线启动

### 预下载 BF16 源模型

```bash
./download-model.sh
./download-model.sh google/gemma-4-26B-A4B-it
```

默认下载到：

```text
.cache/modelscope/<repo_id>/
```

容器里会以同样的相对路径挂载到：

```text
/root/.cache/modelscope/<repo_id>/
```

### 模型同步策略

启动时可以通过 `VLLM_MODEL_SYNC_POLICY` 控制是否从 ModelScope 更新快照：

- `if_missing`
  只在本地缺失时下载，默认值
- `always`
  每次 `start/restart` 前都同步远端快照
- `never`
  严格离线，只接受本地已存在的模型

示例：

```bash
VLLM_MODEL_SYNC_POLICY=always ./vllm.sh restart
VLLM_MODEL_SYNC_POLICY=never ./vllm.sh start gemma26b
```

## Thinking / Tool Use

当前仓库的推荐理解方式是：

- parser 常开，不等于默认自动 thinking
- 是否真的进入 thinking，仍取决于请求 render 后是否注入了 `<|think|>`
- 在当前本地模板补丁下，`reasoning_effort=medium|high` 会切到 thinking

当前还额外加了两层防护：

- parser 侧可选 heuristic final-answer salvage
- vLLM 侧 hard reasoning budget + 尾部重复检测

这条线的更多背景可以看：

- [thinking-mode-behavior.md](/home/zji/docker/vllm-gemma/docs/thinking-mode-behavior.md)

## 测试与 Benchmark

### 回归测试

```bash
python3 tests/reasoning/test_gemma4_reasoning_parser_patch.py
python3 tests/reasoning/test_gemma4_chat_template_patch.py
python3 tests/reasoning/test_vllm_reasoning_budget_patch.py
```

### 专项 benchmark

```bash
python3 tests/performance/gemma4_reasoning_tool_bench.py \
  --rounds 3 \
  --max-tokens 1024 \
  --output-json results/benchmarks/gemma4-reasoning-tool-1024-YYYYMMDD.json
```

现有记录：

- [benchmark-baseline.md](/home/zji/docker/vllm-gemma/docs/benchmark-baseline.md)
- [benchmark-reasoning-tool-8192.md](/home/zji/docker/vllm-gemma/docs/benchmark-reasoning-tool-8192.md)
- [gemma4-reasoning-tool-1024-20260415.json](/home/zji/docker/vllm-gemma/results/benchmarks/gemma4-reasoning-tool-1024-20260415.json)

## 把本地模型上传到 ModelScope

如果你已经把本地模型目录更新好了，可以用仓库自带脚本上传：

```bash
MODELSCOPE_ACCESS_TOKEN=ms-xxx ./scripts/upload-modelscope-fp8.sh
```

默认行为：

- 上传源目录：`.cache/modelscope/kuohao/gemma-4-26B-A4B-it-FP8`
- 目标仓库：`kuohao/gemma-4-26B-A4B-it-FP8`
- 优先使用 `modelscope upload`
- CLI 不可用时回退到 SDK `upload_folder()`

这个脚本上传的是整个模型目录，所以 `chat_template.jinja`、`tokenizer_config.json`、权重索引等都会一起进入 ModelScope 仓库。

## 更新流程

这里最容易混淆的是：模型快照更新，和容器内 vLLM 补丁更新，不是同一件事。

### 场景 A：只更新 ModelScope 上的模型文件

适用于：

- `chat_template.jinja`
- `tokenizer_config.json`
- 权重文件或索引
- 其他随模型仓库发布的文件

发布步骤：

```bash
MODELSCOPE_ACCESS_TOKEN=ms-xxx ./scripts/upload-modelscope-fp8.sh
```

另一台机器升级步骤：

```bash
git pull
VLLM_MODEL_SYNC_POLICY=always ./vllm.sh restart
```

说明：

- 这里的 `git pull` 是为了拿到 launcher 脚本和 compose 变更
- 真正的模型快照更新来自 `snapshot_download`
- 如果那台机器的代码仓库没变，只做 `VLLM_MODEL_SYNC_POLICY=always ./vllm.sh restart` 也能拿到最新模型文件

### 场景 B：更新容器内 vLLM patch / reasoning 逻辑

适用于：

- `scripts/patch-vllm-gemma4-reasoning-parser.py`
- `scripts/patch-vllm-openai-reasoning-budget.py`
- `Dockerfile.stable`
- `docker-compose.yml`
- `scripts/vllm/serve-model.sh`

这类改动是 build-time 生效的，不能只靠拉 ModelScope 模型解决。

另一台机器的正确升级步骤：

```bash
git pull
docker compose build gemma26b
VLLM_MODEL_SYNC_POLICY=always ./vllm.sh restart
```

关键点：

- `git pull` 拿到最新脚本和 Dockerfile
- `docker compose build gemma26b` 重新把本地 patch 打进镜像
- `./vllm.sh restart` 重新应用本地模型 patch，并用新镜像启动
- `VLLM_MODEL_SYNC_POLICY=always` 让模型快照也同步到最新

### 场景 C：另一套仓库只下载同一个 ModelScope 模型

如果另一套仓库只是把 `kuohao/gemma-4-26B-A4B-it-FP8` 拉下来用，但没有这里的：

- `Dockerfile.stable`
- `serve-model.sh`
- vLLM patch scripts
- compose env

那么它只能自动拿到模型侧更新，不能自动拿到这里的 parser / budget / reasoning runtime 行为。

换句话说：

- 拉模型，只能升级模型文件
- 重建镜像，才能升级容器内实现

## 仓库结构

```text
.
├── Dockerfile.stable
├── docker-compose.yml
├── vllm.sh
├── download-model.sh
├── scripts/
├── tests/
├── docs/
├── results/
└── .cache/
```

更细的项目说明见：

- [AGENTS.md](/home/zji/docker/vllm-gemma/AGENTS.md)
