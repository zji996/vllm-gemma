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
│   └── vllm/                    # vllm.sh 模块拆分
│       ├── entry.sh             #   入口分发
│       ├── cli.sh               #   命令行模式
│       ├── tui.sh               #   交互式 TUI 界面
│       ├── config.sh            #   模型注册表 (registry pattern)
│       ├── common.sh            #   公共函数
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
# 1. 预下载模型 (约 48GB, 支持断点续传)
./download-model.sh

# 2. 构建 Docker 镜像
docker compose build gemma26b

# 3. 启动模型 (默认 PP=2 / TP=1)
./vllm.sh start gemma26b
```

### 模型下载

```bash
# 默认下载 Gemma-4-26B-A4B-it
./download-model.sh

# 显式指定模型 ID
./download-model.sh google/gemma-4-26B-A4B-it

# 模型缓存在 .cache/modelscope/hub/google/gemma-4-26B-A4B-it/
# 与 vLLM 容器内 MODELSCOPE_CACHE 路径完全一致, 无需重复下载
```

### 日常使用

```bash
./vllm.sh list          # 列出所有模型 profile
./vllm.sh start gemma26b  # 启动 Gemma-4-26B (默认 PP=2 / TP=1)
./vllm.sh stop           # 停止当前模型
./vllm.sh status         # 查看运行状态
./vllm.sh logs           # 查看日志
```

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
