# AGENTS.md — vLLM Experiments 项目指南

## 项目概述

这是一个围绕 vLLM 的**集成实践与推理优化**项目，用于在 RTX 3080 20GB (SM86) 上部署和调优 Qwen3.5 系列多模态模型。核心场景是视频行为识别（12 帧 → Charades 数据集）。

## 技术栈

| 组件 | 版本/工具 |
|------|----------|
| 推理引擎 | stable: vLLM v0.18.0 |
| 基础镜像 | stable: `vllm/vllm-openai:v0.18.0-cu130` |
| GPU | 2× RTX 3080 20GB (SM86, Compute 8.6) |
| 容器编排 | Docker Compose |
| 模型 | Qwen3.5 系列 (4B / 9B / 27B-FP8) |
| Triton | 3.6.0 |
| FlashInfer | 0.6.4 (GDN 仅 SM90，SM86 不可用) |

## 项目结构

```
.
├── AGENTS.md                    # ← 你正在看的文件
├── Dockerfile.stable            # 稳定版镜像 (官方原版 + scripts, 默认)
├── Dockerfile.experimental      # 实验版镜像 (0.17.1 + rootfs patch + scripts)
├── docker-compose.yml           # 多模型编排 (6 个 profile)
├── vllm.sh                      # CLI/TUI 入口脚本
├── plan.md                      # 优化路线图
│
├── scripts/
│   └── vllm/                    # vllm.sh 模块拆分
│       ├── entry.sh             #   入口分发
│       ├── cli.sh               #   命令行模式
│       ├── tui.sh               #   交互式 TUI 界面
│       ├── config.sh            #   模型注册表 (registry pattern)
│       └── common.sh            #   公共函数
│
├── patches/
│   └── rootfs/                  # 容器内文件覆盖 (stable + experimental 均使用)
│       └── usr/local/lib/python3.12/dist-packages/vllm/
│           ├── model_executor/models/
│           │   ├── qwen3_5.py      # GDN 优化 (buffer 复用, GEMM 融合等)
│           │   └── qwen3_next.py   # GDN 优化 (buffer 复用, _get_core_attn_out_buffer 等)
│           └── model_executor/layers/fla/ops/
│               ├── chunk.py        # FLA chunk entry (chunk_size 等)
│               └── cumsum.py       # FLA cumsum kernel
│
├── tests/                       # 测试与评估
│   ├── performance/             #   性能压测
│   ├── tool_calling/            #   Tool Calling 测试
│   └── video_understanding/     #   视频理解评估
│
├── results/                     # 实验结果 (JSONL)
├── docs/                        # 文档与经验
├── data/                        # 数据集 (gitignore, Charades_v1_480)
└── .cache/                      # 运行时缓存 (gitignore)
```

## 核心概念

### 双 Dockerfile 架构 (stable / experimental)

| Dockerfile | 镜像 tag | 用途 |
|------------|----------|------|
| `Dockerfile.stable` | `local/vllm-openai:latest` | **默认生产镜像**，保持 upstream 原版基线 |
| `Dockerfile.experimental` | `local/vllm-openai:exp` | 实验分支，承载 rootfs patch 与优化验证 |

**当前状态**：
- `stable` 基础镜像：`vllm/vllm-openai:v0.18.0-cu130`
- `experimental` 基础镜像：`vllm/vllm-openai:v0.17.1-x86_64`
- 两者都支持可选安装 transformers 源码版本
- `stable` 只保留官方镜像内容和项目 scripts，不注入任何 rootfs patch
- `experimental` 继续通过 `patches/rootfs/` 覆盖 vLLM 源码，待后续再 rebase 到 0.18.0

**工作流**：新优化先在 `Dockerfile.experimental` 验证；`stable` 作为 upstream 原版基线，不再承载实验优化。

docker-compose.yml 中通过 YAML anchor 切换：
```yaml
# 默认使用 stable (upstream 原版基线)
x-build-config: &build-config
  <<: *build-config-stable

# 切换到 experimental (测试新优化):
x-build-config: &build-config
  <<: *build-config-experimental
```

### 模型 Profile 系统

每个模型配置是 docker-compose 中的一个 `profile`，同一时间只能运行一个（共享端口 8000）：

| Profile | 说明 | GPU | 上下文 |
|---------|------|-----|--------|
| `qwen27b` | 27B-FP8 多模态主力 | 2× | 262K |
| `qwen27b-text` | 27B-FP8 纯文本 | 2× | 262K |
| `qwen9b` | 9B 超长上下文 | 2× | 262K |
| `qwen9b-nvfp4` | 9B NVFP4 量化 | 1× | 262K |
| `qwen4b` | 4B 轻量开发调试 | 1× | 128K |
| `qwen4b-cuda1` | 4B cuda:1 对比 (端口 8001) | 1× | 128K |

### 文件覆盖机制 (patches/rootfs)

当前只有 `Dockerfile.experimental` 在构建时执行两步注入：

```dockerfile
# 1. rootfs 覆盖 — 整个文件替换
cp -a /opt/patches/rootfs/. /

# 2. .patch 文件 — diff 增量补丁 (当前无 .patch 文件)
find /opt/patches -maxdepth 2 -name '*.patch' | patch -p1 -d /
```

**修改实验镜像源码的首选方式**是将完整文件放到 `patches/rootfs/` 对应路径下。

### 环境锚点 (YAML anchors)

`docker-compose.yml` 使用 YAML anchors 实现配置复用：
- `x-env-modelscope` / `x-env-huggingface` — 环境变量模板
- `x-common-volumes` — 公共挂载（模型缓存、Triton cache 等）
- `x-deploy-*` — GPU 分配策略
- `x-build-config-stable` / `x-build-config-experimental` — 构建配置
- `x-base-*` — 服务基础模板

新增环境变量或挂载时，应**修改相应锚点**而非直接改单个服务，确保所有服务统一生效。

## 常用操作

### 日常使用

```bash
./vllm.sh list          # 列出所有模型 profile
./vllm.sh start qwen4b  # 启动指定模型
./vllm.sh stop           # 停止当前模型
./vllm.sh status         # 查看运行状态
./vllm.sh logs           # 查看日志
./vllm.sh bench          # 跑 benchmark (Charades 视频行为识别)
```

### 构建镜像

```bash
# 默认 stable 构建
docker compose build qwen4b

# 带 transformers 源码安装
INSTALL_TRANSFORMERS_FROM_SOURCE=true docker compose build qwen4b
```

### 切换 stable / experimental

修改 `docker-compose.yml` 中的默认 build-config anchor：
```yaml
x-build-config: &build-config
  <<: *build-config-experimental  # 或 *build-config-stable
```
然后重新构建：`docker compose build qwen4b`

### 添加新的源码覆盖

1. 从基础镜像提取原文件：
   ```bash
   docker run --rm --entrypoint cat vllm/vllm-openai:v0.17.1-x86_64 \
     /usr/local/lib/python3.12/dist-packages/vllm/path/to/file.py \
     > /tmp/file_orig.py
   ```
2. 修改并放入 rootfs：
   ```bash
   cp /tmp/file_orig.py patches/rootfs/.../file.py
   # 编辑 patches/rootfs/.../file.py
   ```
3. 切换到 experimental 构建并验证；如需升级 experimental 基线版本，先从新版本镜像重新提取并对齐 rootfs 文件

## 当前优化状态

参见 [plan.md](plan.md) 获取完整路线图和深层分析。

### 优化状态 (已收敛)

| 编号 | 内容 | 状态 |
|------|------|------|
| Step 2 | Triton 编译缓存持久化 | ✅ 已完成 |
| Step 3 | torch.compile (MM encoder) | ✅ compile_mm_encoder=true |
| Step 5 | GDN buffer 复用 | ✅ _get_core_attn_out_buffer |
| Step 6 | ViT 注意力后端 | ✅ FLASH_ATTN |
| Task 2 | GEMM 融合 (qkvz+ba 合并) | ✅ Gen +0.7% |
| Task 3 | 消除 b/a contiguous | ✅ 已完成 |
| Task 6 | compile+cudagraph (默认路径) | ✅ Gen +91% vs eager |
| ~~Task 1~~ | FLA autotune 扩展 | ❌ 回退 |
| ~~Task 3b~~ | 消除 rearrange .contiguous() | ❌ 回退 |
| ~~Task 4~~ | FLA chunk_size=32 | ❌ 回退 |
| ~~Task 7~~ | 手写 CUDA Kernel | ⚠️ 搁置 |

## 关键约束

1. **FlashInfer GDN 不支持 SM86** — 所有线性注意力只能走 FLA Triton 路径
2. **SM86 共享内存受限** — `check_shared_mem()=False`，限制 Triton kernel tile size（BKV 最大 64）
3. **Generation 是时间大头 (73%)** — 且为内存带宽瓶颈，当前利用率 ~77%
4. **同一时间只能运行一个模型** — 所有 profile 共享端口 8000 (`qwen4b-cuda1` 例外：端口 8001)
5. **experimental 基础镜像仍锁定** — `vllm/vllm-openai:v0.17.1-x86_64`；升级 experimental 时需重新提取并对齐所有 rootfs 文件
6. **qwen4b 是优化主战场** — 先在 4B 单卡验证收益，再拓展到 9B TP=2

## Benchmark 约定

- 场景：Charades 数据集，固定 seed=20260313，3 个视频 × 12 帧
- 命令：`./vllm.sh bench --config-id <label>`
- 结果：`results/performance/action_bench_*.jsonl`
- 核心指标：TTFT、generation TPS、prefill TPS、整体 latency
