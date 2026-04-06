## Gemma-4-26B 默认部署方案

当前仓库默认的双卡部署方案已经收敛为：

- Profile: `gemma26b`
- 并行策略: `PP=2 / TP=1`
- 默认上下文: `64K`
- 默认显存利用率: `${GPU_MEMORY_UTILIZATION:-0.93}`
- 模型路径: `/root/.cache/modelscope/google/gemma-4-26B-A4B-it-FP8`

这条路径已经在本机 `2× RTX 3080 20GB / SM86` 上完成实际推理验证，可稳定响应 `/v1/chat/completions`。

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

如需覆盖默认值，可临时指定：

```bash
GPU_MEMORY_UTILIZATION=0.93 MAX_MODEL_LEN=65536 ./vllm.sh start gemma26b
```

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
