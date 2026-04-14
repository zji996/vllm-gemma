## Gemma-4-26B 默认部署方案

当前仓库默认的双卡部署方案已经收敛为：

- Profile: `gemma26b`
- 并行策略: `PP=2 / TP=1`
- 默认上下文: `64K`
- 默认显存利用率: `${GPU_MEMORY_UTILIZATION:-0.93}`
- 默认模型源: `kuohao/gemma-4-26B-A4B-it-FP8` (由启动脚本同步到本地缓存后离线加载)

该配置已经在本机 `2× RTX 3080 20GB / SM86` 上完成实际推理验证，可稳定响应 `/v1/chat/completions`。当前推荐链路是由 `./vllm.sh start gemma26b` 先把模型同步到宿主机 `.cache/modelscope/`，再把容器中的 `VLLM_MODEL` 指向本地挂载路径 `/root/.cache/modelscope/...`，使 vLLM 启动阶段不依赖运行时联网下载。

## 为什么不再默认 TP=2

在 2× RTX 3080 20GB / SM86 上，`TP=2` 会把 Gemma-4-26B 的部分中间维度切成更容易命中 Marlin shape gap 的形状，排障期曾出现：

- 请求能完成，但输出为空串或近似 `<pad>`
- Marlin FP8 / MoE 路径数值异常

`PP=2 / TP=1` 的目标是先避开这类 TP 诱发的 shape 风险，把问题缩小到更可控的 Gemma4 + PP 路径。

## 已知 PP 修复

为让 Gemma4 在 vLLM v0.19.0 的 PP 路径上稳定运行，当前镜像在 `scripts/patch-vllm-gemma4-fp8.py` 里额外做了两类修复：

- 离线 FP8 expert scale 的 loader 映射
- Gemma4 PLE / PP 中间张量在 `IntermediateTensors` 中保持静态键集

其中 `per_layer_inputs` 和 `residual` 在为空时会使用零张量占位，以符合 PP 预分配 buffer 的设计假设。

## 关于 Async Scheduling

运行期日志已确认当前服务实际处于：

- `Asynchronous scheduling is enabled.`

因此，本仓库现阶段记录到的并发 benchmark，本质上已经是“开启 async scheduling”的基线，而不是关闭 async 后的结果。

对当前这套 `Gemma4 + FP8 + PP=2` 路径，现阶段建议把 async 视为“当前可用且已验证”的运行模式；如需做稳定性或吞吐 A/B，对比对象应当是“显式关闭 async 的另一版配置”。

## 关于 Thinking / Tool Use

当前默认部署的 thinking 行为已做过专项验证，结论如下：

- 当前默认服务会始终带 `--reasoning-parser gemma4`
- 但默认 `gemma26b` 仍然是“默认 non-thinking 托管”
- 默认请求下，`/v1/chat/completions` 走 non-thinking 模板，普通问答返回干净 final answer
- 默认 non-thinking 路径下，tool roundtrip 已验证可正常返回最终结果，不会再泄漏 `thought` 标签

### 当前可用结论

- 默认请求: 不思考
- `reasoning_effort=medium|high`: 在当前本地 launcher patch 下，会映射为 thinking
- `reasoning_effort=none|low`: 在当前本地 launcher patch 下，会保持 non-thinking
- 顶层 `chat_template_kwargs.enable_thinking=true`: 会真实注入 `<|think|>`
- `extra_body.chat_template_kwargs.enable_thinking=true`: 在当前 vLLM OpenAI 接口里，实测未观察到生效

### 当前兼容性现状

当前仓库已新增本地 Gemma4 reasoning parser 补丁，行为分成两类：

- `reasoning_effort=medium|high` 的 tool roundtrip 第二轮，现在可以把 thought 从 `message.content` 里剥出来，并把 reasoning 放到 `message.reasoning`
- 普通问答里，如果模型没有明确输出 thinking 结束边界，当前实现会保守地把整段放进 `message.reasoning`，并返回 `content = null`，而不是继续把 thought 泄漏到 `content`
- 如显式打开 `SERVE_GEMMA4_HEURISTIC_FINAL_ANSWER=true`，parser 还会尝试从 `Final Answer:` / `最终答案:` 这类尾部标记里回填 `message.content`

已确认仍成立的是：

- 默认请求不会自动进入 thinking
- `extra_body.chat_template_kwargs.enable_thinking=true` 仍未观察到生效
- 对没有明确 `<channel|>` 结束边界的普通 thinking 回复，当前还没有做激进的“最终答案猜测式拆分”

因此，当前更准确的表述是：

- non-thinking 默认路径稳定
- thinking + tool roundtrip 已可做到“reasoning/content 保守拆分”
- 普通 thinking 问答已能避免 `content` 泄漏，但不保证一定同时拿到非空 final `content`

更完整的专项测试记录见：

- `docs/thinking-mode-behavior.md`
- `python3 tests/reasoning/check_gemma4_reasoning_split.py --check-heuristic-sample`

## Benchmark 基线

本轮已补充文本并发 benchmark 脚本：

- `tests/performance/text_chat_bench.py`

基线结果单独整理在：

- `docs/benchmark-baseline.md`

其中包含三类口径：

- 短 prompt 并发吞吐
- 长输出 decode 吞吐
- 高并发长 prompt prefill 吞吐

## 启动方式

```bash
docker compose build gemma26b
./vllm.sh start gemma26b
```

默认启动行为现在是：

- 启动脚本按 `MS_GEMMA26B_MODEL_ID` 解析远端仓库
- 将快照同步到宿主机 `.cache/modelscope/<repo_id>/`
- 对本地 `chat_template.jinja` 应用 launcher-managed patch
- 将容器内 `VLLM_MODEL` 固定为 `/root/.cache/modelscope/<repo_id>/`
- 使用离线模式启动 vLLM

如需控制同步策略，可临时覆盖：

```bash
VLLM_MODEL_SYNC_POLICY=always ./vllm.sh restart
VLLM_MODEL_SYNC_POLICY=never ./vllm.sh start gemma26b
```

如需覆盖默认值，可临时指定：

```bash
GPU_MEMORY_UTILIZATION=0.93 MAX_MODEL_LEN=65536 ./vllm.sh start gemma26b
```

如需切到其他远端仓库或本地变体目录，可临时覆盖：

```bash
MS_GEMMA26B_MODEL_ID=kuohao/gemma-4-26B-A4B-it-FP8 ./vllm.sh start gemma26b
MS_GEMMA26B_MODEL_ID=.cache/modelscope/variants/newchat-oldtok ./vllm.sh start gemma26b
```

注意：

- 本地变体目录应放在 `.cache/modelscope/` 下，容器才能直接看到
- `./vllm.sh restart` 现在会重建容器，以便重新应用同步结果和本地补丁

如需显式切换服务级 thinking 开关，可临时覆盖：

```bash
SERVE_ENABLE_THINKING=true ./vllm.sh start gemma26b
```

但请注意：

- 这不会让默认请求自动进入 thinking 模式
- 默认请求是否注入 `<|think|>`，仍取决于请求侧 chat template 参数
- 当前 thinking 模式的 OpenAI 响应结构仍存在兼容性问题，见上文说明

## 关于启动时的 unknown env warning

项目的 `serve-model.sh` 会先把仓库内部使用的 `VLLM_*` 环境变量翻译成 `vllm serve` 的 CLI 参数，再在 `exec` 前清掉这些仅供脚本使用的变量，以避免出现这类警告：

- `Unknown vLLM environment variable detected: VLLM_MODEL`
- `Unknown vLLM environment variable detected: VLLM_MAX_MODEL_LEN`
- `Unknown vLLM environment variable detected: VLLM_PIPELINE_PARALLEL_SIZE`

这些变量不是 vLLM 官方环境变量，而是本项目内部的 compose-to-CLI 适配层。

## 运行期备注

- 文档默认配置仍以 `64K` 为目标值记录。
- 本轮 benchmark 针对的是当时在线容器的实际运行状态；该实例的 `/v1/models` 返回 `max_model_len=49152`。
- 若后续以新镜像重建并切换到完整 `65536`，建议重新补一轮同口径 benchmark。
