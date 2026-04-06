## Gemma-4-26B PP=2 基线

本文记录当前 `Gemma-4-26B-A4B-it-FP8` 在本机 `2× RTX 3080 20GB / SM86` 上的文本推理 baseline。

测试前提：

- 服务接口：`/v1/chat/completions`
- 模型：`gemma`
- 并行策略：`PP=2 / TP=1`
- 运行期日志显示：`Asynchronous scheduling is enabled.`
- 因此，下述结果默认包含 async scheduling 的实际效果

压测脚本：

- `tests/performance/text_chat_bench.py`

## 短 Prompt，并发吞吐

口径：

- 短中文问答 prompt
- `max_tokens=96`

饱和版结果：

| 并发 | 请求数 | req/s | output tok/s | p50 | p95 |
|------|-------:|------:|-------------:|----:|----:|
| 8 | 32 | 5.65 | 537.6 | 1.42s | 1.45s |
| 16 | 32 | 9.79 | 932.4 | 1.63s | 1.65s |

更高并发结果：

| 并发 | 请求数 | req/s | output tok/s | p50 | p95 |
|------|-------:|------:|-------------:|----:|----:|
| 16 | 64 | 7.40 | 706.5 | 2.16s | 2.24s |
| 24 | 64 | 8.79 | 839.3 | 2.48s | 2.52s |
| 32 | 64 | 14.02 | 1343.1 | 2.26s | 2.30s |

结论：

- 短 prompt 下，系统在高并发时仍有明显扩展性。
- 当前观测到的高点约为 `1343 output tok/s @ concurrency=32`。

## 长输出，Decode 吞吐

口径：

- 短 prompt
- `max_tokens=256`

结果：

| 并发 | 请求数 | req/s | output tok/s | p50 | p95 |
|------|-------:|------:|-------------:|----:|----:|
| 4 | 32 | 1.67 | 166.3 | 2.29s | 2.77s |
| 8 | 32 | 5.02 | 492.7 | 1.51s | 1.89s |
| 16 | 32 | 9.22 | 893.1 | 1.71s | 1.82s |

结论：

- 长输出场景下，`16` 并发已能稳定接近 `900 output tok/s`。

## 长 Prompt，高并发 Prefill 吞吐

口径：

- `prompt_repeat=128`
- 单请求约 `4.6k prompt tokens`
- `max_tokens=64`

结果：

| 并发 | 请求数 | req/s | prompt tok/s | output tok/s | p50 | p95 |
|------|-------:|------:|-------------:|-------------:|----:|----:|
| 16 | 32 | 0.99 | 4673.8 | 63.4 | 15.80s | 26.72s |
| 24 | 32 | 0.97 | 4599.6 | 62.0 | 21.72s | 32.63s |
| 32 | 32 | 0.96 | 4557.2 | 61.4 | 31.02s | 33.28s |

结论：

- 长 prompt 下，系统瓶颈明显转向 prefill。
- 当前 prefill 吞吐大致稳定在 `4.5k ~ 4.7k prompt tok/s`。
- 并发从 `16` 提到 `32` 时，总吞吐几乎不再增长，但尾延迟会明显上升。

## 文件索引

原始结果：

- `results/benchmarks/pp2-baseline-20260406.json`
- `results/benchmarks/pp2-baseline-c1-c16-20260406.json`
- `results/benchmarks/pp2-baseline-c8-c16x32-20260406.json`
- `results/benchmarks/pp2-baseline-c16-c32x64-20260406.json`
- `results/benchmarks/pp2-baseline-longgen-c4-c16-20260406.json`
- `results/benchmarks/pp2-baseline-longprompt-c16-c32-20260406.json`

后续若切换镜像、修改 async 配置、调整 `max_model_len` 或更换并行策略，建议继续沿用同一脚本与口径复测。
