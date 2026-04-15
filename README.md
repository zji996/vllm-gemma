# vLLM Gemma 4 Launcher

一个给 `Gemma 4` 准备的本地部署仓库，目标很直接：

- 在 `2× RTX 3080 20GB` 上把 `gemma-4-26B-A4B-it` 跑稳
- 默认走已经调优过的 `FP8` 路线
- 把 thinking、tool use、ModelScope 同步和离线启动这些麻烦事尽量收进脚本里

如果你是第一次 clone 这个仓库，最重要的一句话是：

> 默认主线是 `FP8` 托管，不是 `BF16`。

也就是说，平时直接启动就好：

```bash
./vllm.sh start gemma26b
```

`BF16` 那条路还在，但它更像实验台，适合做别的量化、重新导出或者对照测试。

## 这仓库现在默认怎么工作

- 默认服务模型：`kuohao/gemma-4-26B-A4B-it-FP8`
- 默认并行策略：`PP=2 / TP=1`
- 默认请求：`non-thinking`
- `reasoning_effort=none|low`：保持 `non-thinking`
- `reasoning_effort=medium|high`：进入 `thinking`
- reasoning parser 默认开启
- docker compose 默认只绑定 `127.0.0.1`
- 如果没设置 `API_KEY`，默认按本地无鉴权模式运行

## 先跑起来

### 1. 构建镜像

```bash
docker compose build gemma26b
```

### 2. 启动默认模型

```bash
./vllm.sh start gemma26b
```

### 3. 看状态

```bash
./vllm.sh status
./vllm.sh logs
```

### 4. 常用命令

```bash
./vllm.sh list
./vllm.sh start gemma26b
./vllm.sh restart
./vllm.sh stop
./vllm.sh build
```

启动脚本会自动帮你做这几件事：

1. 解析 `MS_GEMMA26B_MODEL_ID`
2. 如果需要，从 ModelScope 同步本地快照
3. 对本地 `chat_template.jinja` 打 launcher-managed patch
4. 用本地快照目录离线启动 vLLM

所以大多数时候，不需要手工拼一长串 `docker compose` 参数。

## 模型配置

<!-- BEGIN:MODELS_TABLE -->
| Profile | 模型 | 上下文 | GPU | 说明 |
|---------|------|--------|-----|------|
| `gemma26b` | Gemma-4-26B-A4B-it | 64K | 2× | ⭐ 默认: PP=2 / TP=1 / Gemma4 patch |
<!-- END:MODELS_TABLE -->

如果后面 profile 有变化，可以更新 README 里的表：

```bash
./vllm.sh sync-readme
```

## 两条模型路线

### 路线 A：默认主线，FP8 托管

这条是当前仓库的真实主线：

```text
kuohao/gemma-4-26B-A4B-it-FP8
```

平时运行就是它：

```bash
./vllm.sh start gemma26b
```

启动时会优先用 `MS_GEMMA26B_MODEL_ID` 指向这个 FP8 仓库，并按同步策略决定是否拉取最新快照。

### 路线 B：实验台，BF16 源模型

如果你想测新的量化方案，或者想自己重新导出 FP8 / 其他格式，可以先拉 BF16：

```bash
./download-model.sh
./download-model.sh google/gemma-4-26B-A4B-it
```

这条路线更适合：

- 试别的量化方法
- 重新导出模型
- 做 tokenizer / 配置 / 模板对照实验

一句话总结：

- 日常托管：走 `FP8`
- 实验和再量化：走 `BF16`

## 模型同步策略

`./vllm.sh start` 和 `./vllm.sh restart` 都会看 `VLLM_MODEL_SYNC_POLICY`：

- `if_missing`
  只在本地没有快照时下载，默认值
- `always`
  每次启动前都从 ModelScope 同步最新快照
- `never`
  严格离线，只接受本地已有模型

例如：

```bash
VLLM_MODEL_SYNC_POLICY=always ./vllm.sh restart
VLLM_MODEL_SYNC_POLICY=never ./vllm.sh start gemma26b
```

## Thinking 和 Tool Use

这里最容易绕晕的一点是：

- parser 常开，不代表默认自动 thinking
- 真正会不会 thinking，还是看请求 render 后有没有进 `<|think|>`

当前仓库里：

- `reasoning_effort=medium|high` 会切到 thinking
- parser 会尽量把 thought 从 `message.content` 里拆出去
- 还额外做了 hard reasoning budget 和尾部重复检测，防止一直在 thought 里打转

想看这条线更细的背景，可以读：

- [docs/thinking-mode-behavior.md](docs/thinking-mode-behavior.md)

## 测试和 Benchmark

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

已有记录：

- [docs/benchmark-baseline.md](docs/benchmark-baseline.md)
- [docs/benchmark-reasoning-tool-8192.md](docs/benchmark-reasoning-tool-8192.md)
- `results/benchmarks/gemma4-reasoning-tool-1024-20260415.json`

## 上传到 ModelScope

如果你已经把本地 FP8 模型目录整理好了，可以直接上传：

```bash
MODELSCOPE_ACCESS_TOKEN=ms-xxx ./scripts/upload-modelscope-fp8.sh
```

默认上传的是：

- 源目录：`.cache/modelscope/kuohao/gemma-4-26B-A4B-it-FP8`
- 目标仓库：`kuohao/gemma-4-26B-A4B-it-FP8`

这个脚本会上传整个模型目录，所以 `chat_template.jinja`、`tokenizer_config.json`、权重索引这些都会一起进入 ModelScope。

## 更新流程

这里有个很重要但也很容易混淆的点：

> 模型快照更新，和容器里的 vLLM patch 更新，不是同一件事。

### 情况 A：你只更新了模型仓库内容

比如你更新了：

- `chat_template.jinja`
- `tokenizer_config.json`
- 权重文件

发布：

```bash
MODELSCOPE_ACCESS_TOKEN=ms-xxx ./scripts/upload-modelscope-fp8.sh
```

另一台机器拿最新模型：

```bash
VLLM_MODEL_SYNC_POLICY=always ./vllm.sh restart
```

如果那台机器连仓库代码也更新一下，会更稳：

```bash
git pull
VLLM_MODEL_SYNC_POLICY=always ./vllm.sh restart
```

### 情况 B：你更新了容器内逻辑

比如你改了：

- `Dockerfile.stable`
- `docker-compose.yml`
- `scripts/vllm/serve-model.sh`
- `scripts/patch-vllm-gemma4-reasoning-parser.py`
- `scripts/patch-vllm-openai-reasoning-budget.py`

这类改动是 build-time 生效的，所以只拉模型没用，必须重建镜像：

```bash
git pull
docker compose build gemma26b
VLLM_MODEL_SYNC_POLICY=always ./vllm.sh restart
```

一句话版本：

- 拉模型，只会更新模型文件
- 重建镜像，才会更新容器内实现

### 情况 C：别的仓库只拉同一个 ModelScope 模型

如果另一套仓库只是下载 `kuohao/gemma-4-26B-A4B-it-FP8`，但没有这个仓库里的：

- `Dockerfile.stable`
- launcher 脚本
- compose 配置
- vLLM patch scripts

那它只能拿到模型侧更新，拿不到这里的 reasoning parser / budget / runtime 行为。

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

更细的项目说明在：

- [AGENTS.md](AGENTS.md)

## 最后一句

如果你刚 clone 完，只想确认“这玩意是不是能跑”：

```bash
docker compose build gemma26b
./vllm.sh start gemma26b
```

先让它跑起来，剩下的 thinking、tool use、benchmark、上传和升级链路，都已经在上面给你留好路标了。
