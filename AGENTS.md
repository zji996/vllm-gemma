# AGENTS.md — vLLM Gemma 4 项目指南

## 项目概述

这是一个围绕 vLLM 的**集成实践与推理优化**项目，用于在 RTX 3080 20GB (SM86) 上部署和调优 Google Gemma 4 系列模型。优先适配 gemma-4-26B-A4B-it (MoE: 26B总参/4B激活)。

## 技术栈

| 组件 | 版本/工具 |
|------|----------|
| 推理引擎 | vLLM v0.19.0 |
| 基础镜像 | `vllm/vllm-openai:v0.19.0` |
| GPU | 2× RTX 3080 20GB (SM86, Compute 8.6) |
| 容器编排 | Docker Compose |
| 模型 | Gemma 4 系列 (26B-A4B-it) |
| Transformers | ≥5.5.0 |

## 项目结构

```
.
├── AGENTS.md                    # ← 你正在看的文件
├── Dockerfile.stable            # 稳定版镜像 (vLLM v0.19.0 + scripts)
├── docker-compose.yml           # 模型编排
├── download-model.sh            # 模型预下载脚本 (ModelScope)
├── vllm.sh                      # CLI/TUI 入口脚本
│
├── .cache/                      # 本地缓存 (gitignore)
│   ├── modelscope/              #   ModelScope 模型缓存
│   ├── huggingface/             #   HuggingFace 模型缓存
│   ├── vllm/                    #   vLLM 引擎缓存
│   └── triton/                  #   Triton kernel 缓存
│
├── scripts/
│   ├── patch-modelscope-gemma4-chat-template.py  # 本地模型模板补丁 (reasoning_effort / thinking)
│   └── vllm/                    # vllm.sh 模块拆分
│       ├── entry.sh             #   入口分发
│       ├── cli.sh               #   命令行模式
│       ├── tui.sh               #   交互式 TUI 界面
│       ├── config.sh            #   模型注册表 (registry pattern)
│       ├── common.sh            #   公共函数
│       ├── modelscope.sh        #   本地模型同步 / 离线路径解析
│       └── serve-model.sh       #   容器内启动脚本
│
├── tests/                       # 测试与评估
├── results/                     # 实验结果 (JSONL)
├── docs/                        # 文档与经验
├── data/                        # 数据集 (gitignore)
└── .cache/                      # 运行时缓存 (gitignore)
```

## 核心概念

### 模型 Profile 系统

每个模型配置是 docker-compose 中的一个 `profile`，同一时间只能运行一个（共享端口 8000）：

| Profile | 说明 | GPU | 上下文 |
|---------|------|-----|--------|
| `gemma26b` | Gemma-4-26B-A4B-it MoE 多模态主力，默认 PP=2 / TP=1 | 2× | 64K |

### 环境锚点 (YAML anchors)

`docker-compose.yml` 使用 YAML anchors 实现配置复用：
- `x-env-modelscope` / `x-env-huggingface` — 环境变量模板
- `x-common-volumes` — 公共挂载（模型缓存、Triton cache 等）
- `x-deploy-*` — GPU 分配策略
- `x-build-config-stable` — 构建配置
- `x-base-*` — 服务基础模板

新增环境变量或挂载时，应**修改相应锚点**而非直接改单个服务，确保所有服务统一生效。

## Gemma 4 模型信息

### Gemma-4-26B-A4B-it (MoE)

| 属性 | 值 |
|------|---|
| 总参数量 | 25.2B |
| 激活参数量 | 3.8B |
| 层数 | 30 |
| 滑动窗口 | 1024 tokens |
| 上下文长度 | 64K tokens |
| 词汇表大小 | 262K |
| Expert 数量 | 8 active / 128 total + 1 shared |
| 支持模态 | Text, Image |
| Vision Encoder | ~550M 参数 |
| 架构 | `Gemma4ForConditionalGeneration` |
| License | Apache 2.0 |

### 采样参数推荐

- `temperature=1.0`
- `top_p=0.95`
- `top_k=64`

### Thinking 模式

- 通过 `<|think|>` token 在 system prompt 开头启用
- 模型输出结构: `<|channel>thought\n`[推理过程]`<channel|>`[最终回答]

## 常用操作

### Quick Start

```bash
# 1. 构建 Docker 镜像
docker compose build gemma26b

# 2. 启动模型 (默认 PP=2 / TP=1)
./vllm.sh start gemma26b
```

默认推荐链路是：

1. `vllm.sh` 先根据 `MS_GEMMA26B_MODEL_ID` 把模型同步到宿主机 `.cache/modelscope/`
2. 对本地 `chat_template.jinja` 应用 launcher-managed patch
3. 再将容器中的 `VLLM_MODEL` 指向 `/root/.cache/modelscope/...` 本地路径离线启动

这样可以避免依赖 vLLM 运行时自行联网下载模型。

### 模型下载

```bash
# 默认下载 BF16 源模型（供量化/调试使用）
./download-model.sh

# 显式指定模型 ID
./download-model.sh google/gemma-4-26B-A4B-it

# BF16 源模型缓存在 .cache/modelscope/google/gemma-4-26B-A4B-it/
# 与 vLLM 容器内 MODELSCOPE_CACHE 路径完全一致, 无需重复下载
```

`./vllm.sh start gemma26b` 的默认运行模型仍是 `kuohao/gemma-4-26B-A4B-it-FP8`，与上面的 BF16 手动下载用途不同。

### 日常使用

```bash
./vllm.sh list          # 列出所有模型 profile
./vllm.sh start gemma26b  # 启动 Gemma-4-26B (默认 PP=2 / TP=1)
./vllm.sh stop           # 停止当前模型
./vllm.sh restart        # 重建当前模型容器, 重新应用本地同步/补丁
./vllm.sh status         # 查看运行状态
./vllm.sh logs           # 查看日志
```

可用的模型同步策略：

- `VLLM_MODEL_SYNC_POLICY=if_missing` — 默认，仅在本地缺失时下载
- `VLLM_MODEL_SYNC_POLICY=always` — 每次 start/restart 前都同步远端
- `VLLM_MODEL_SYNC_POLICY=never` — 严格离线，要求本地快照已存在

当前默认部署、PP 修复说明与并发基线可参考：

- `docs/pp2-default.md`
- `docs/benchmark-baseline.md`

### 构建镜像

```bash
# 默认 stable 构建
docker compose build gemma26b

# 带 transformers 源码安装
INSTALL_TRANSFORMERS_FROM_SOURCE=true docker compose build gemma26b

# 从 vLLM v0.19.0 源码重编 (适合 2×3080 / sm_86 / Marlin 排障)
INSTALL_VLLM_FROM_SOURCE=true \
VLLM_TORCH_CUDA_ARCH_LIST=8.6 \
docker compose build gemma26b
```

## 关键约束

1. **vLLM v0.19.0 最低要求** — Gemma4ForConditionalGeneration 需要 vLLM ≥0.19.0 + transformers ≥5.5.0
2. **MoE 架构** — 26B 总参但仅 4B 激活，推理速度接近 4B 全密模型
3. **SM86 共享内存受限** — `check_shared_mem()=False`，某些 Triton kernel tile size 受限
4. **同一时间只能运行一个模型** — 所有 profile 共享端口 8000
5. **默认并行策略** — 当前默认使用 PP=2 / TP=1，以规避 TP=2 下的 Marlin shape gap 与数值风险
6. **当前运行基线** — 当前 `PP=2 / TP=1` 路径已在本机跑通，并记录了 async scheduling 开启下的文本 benchmark 基线

## 排障经验 / 踩坑记录

### 1. Gemma 4 模板更新不要只看 README，要重点看 `chat_template.jinja`

Gemma 4 上游近期的 tool use / thinking 相关变更，真正影响行为的核心文件是：

- `chat_template.jinja`
- `tokenizer_config.json`

其中：

- `chat_template.jinja` 决定 prompt 拼接、tool response 回填、`<|think|>` 注入方式
- `tokenizer_config.json` 决定输出结构的解析正则

如果只同步 README 或只看 `config.json`，很容易误判为“没变化”。

### 2. 本项目里，tool roundtrip 回归的根因在 `chat_template.jinja`，不是 `tokenizer_config.json`

已做过交叉验证：

- `旧 chat_template + 新 tokenizer`：tool roundtrip 正常
- `新 chat_template + 旧 tokenizer`：tool roundtrip 出现 `thought\n<channel|>...` 泄漏

结论：

- 这类回归优先排查 `chat_template.jinja`

### 3. 当前 vLLM OpenAI 接口下，thinking 开关实测以顶层 `chat_template_kwargs` 为准

已验证：

- 顶层 `chat_template_kwargs.enable_thinking=true`：会真实注入 `<|think|>`
- `extra_body.chat_template_kwargs.enable_thinking=true`：当前未观察到生效

因此，如需测试 thinking，不要只测 `extra_body`。

### 4. 当前默认部署是“reasoning parser 常开 + 请求级显式 thinking”

当前服务默认会始终带：

- `--reasoning-parser gemma4`

但默认请求是否真的进入 thinking，仍取决于 prompt 是否被注入 `<|think|>`。

因此排障时必须区分两层：

- 服务级 parser：`SERVE_REASONING_PARSER`
- 请求级模板开关：`chat_template_kwargs.enable_thinking`

### 5. 当前 `vLLM 0.19.0 + 本地 gemma4 reasoning parser patch` 的实际行为

已观测到：

- thinking + tool roundtrip 第二轮，`thought` 已可从 `message.content` 剥离，并放入 `message.reasoning`
- 普通 thinking 问答里，如果模型没有明确输出 thinking 结束边界，当前实现会保守返回 `content = null`
- 如显式打开 `SERVE_GEMMA4_HEURISTIC_FINAL_ANSWER=true`，parser 会额外尝试从 `Final Answer:` / `最终答案:` 这类尾部标记里回填 `message.content`
- 默认 non-thinking 请求不受影响，仍走干净的 final answer 路径

因此：

- 当前稳定托管仍建议默认走 non-thinking
- 需要显式 thinking 时，默认仍是“保守拆分”
- 若要做显式 thinking 的 A/B，可配合 `SERVE_GEMMA4_HEURISTIC_FINAL_ANSWER=true` 测试更激进的 final-answer 回填

### 6. 当前默认请求仍是 non-thinking 托管

因此：

- 默认问答: non-thinking
- 默认 tool use: non-thinking
- 默认路径下，tool roundtrip 已验证正常

### 7. 覆盖模型路径重启时，要先确认容器实际加载的是什么

历史运行中的容器可能并不一定真的使用当前 compose 默认值。

排障前建议先看：

```bash
docker inspect vllm-gemma26b --format '{{json .Config.Env}}'
```

重点确认：

- `VLLM_MODEL`
- `SERVE_TOOL_CALL_PARSER`
- `SERVE_REASONING_PARSER`
- `SERVE_GEMMA4_HEURISTIC_FINAL_ANSWER`

### 8. 容器内只看得到 compose 挂载进去的路径

本项目容器里挂载的是：

- `./.cache/modelscope:/root/.cache/modelscope`

因此：

- 宿主机 `/tmp/...` 目录在容器内默认不可见
- 若要用“临时变体目录”做 A/B 测试，必须放在项目的 `.cache/modelscope/` 下
- 并且变体目录中的软链接最好使用相对于 `.cache/modelscope` 内部的相对路径，而不是宿主机绝对路径 `/home/...`

### 9. `.cache/` 被 gitignore，模型缓存目录不是 git 发布路径

当前仓库 `.gitignore` 忽略了：

- `.cache/`

因此：

- 修改 `.cache/modelscope/...` 不会随普通 `git push` 发布
- 发布到 ModelScope 要走 `scripts/upload-modelscope-fp8.sh`

### 10. 本项目已有现成的 ModelScope 上传脚本

上传本地 FP8 导出目录时优先使用：

```bash
./scripts/upload-modelscope-fp8.sh
```

脚本会：

- 自动读取 `.env`
- 兼容 `MODELSCOPE_ACCESS_TOKEN` / `MODELSCOPE_API_TOKEN`
- 默认上传 `kuohao/gemma-4-26B-A4B-it-FP8`

### 11. 更新上游模板后，至少要做两类回归测试

建议最少覆盖：

1. 普通问答
2. tool roundtrip

tool roundtrip 推荐固定口径：

1. 第一轮让模型生成 `tool_calls`
2. 第二轮回填 `role=tool`
3. 观察最终 `message.content` 是否出现 `thought` / `<channel|>` 泄漏

### 12. thinking / tool use 专项记录已沉淀到文档

后续如需继续排查或复测，优先参考：

- `docs/thinking-mode-behavior.md`

### 13. 当前 OpenAI-compatible `reasoning_effort` 是通过本地 chat template patch 接起来的

当前仓库的 launcher-managed 本地补丁会把：

- `reasoning_effort=medium|high` → 映射为 thinking
- `reasoning_effort=none|low` → 保持 non-thinking

这里的关键点是：

- vLLM `0.19.0` 会把 `reasoning_effort` 传给 chat template
- 但 Gemma 4 当前模板是否真的切到 `<|think|>`，取决于模板本身有没有消费这个字段

因此如果后面发现“请求里传了 `reasoning_effort` 但模型没思考”，优先检查本地 `chat_template.jinja` 是否已经过 `scripts/patch-modelscope-gemma4-chat-template.py` 处理。

### 14. 当前推荐的模型更新链路是“启动脚本同步 + 容器离线加载”，不要再依赖 vLLM 运行时下载

当前推荐路径：

1. 宿主机侧通过 `vllm.sh start/restart` 调 `modelscope.snapshot_download`
2. 模型文件落在 `.cache/modelscope/<repo_id>/`
3. 容器用挂载后的 `/root/.cache/modelscope/<repo_id>/` 本地路径启动
4. 容器环境默认走离线模式

因此：

- 更新远端仓库后，如需拉到本地可用 `VLLM_MODEL_SYNC_POLICY=always ./vllm.sh restart`
- 严格离线复现可用 `VLLM_MODEL_SYNC_POLICY=never ./vllm.sh start gemma26b`
- 不建议再把“首次启动能否自动联网拉模型”当成默认依赖
