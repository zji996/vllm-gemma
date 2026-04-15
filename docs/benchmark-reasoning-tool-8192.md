# Gemma 4 Thinking / Tool Benchmark (`max_tokens=8192`)

本文记录当前 `gemma26b` 容器在本机上的专项行为 benchmark，重点观察：

- 放宽 `max_tokens` 到 `8192` 后，thinking 问答是否更容易拿到 final `content`
- thinking + function call roundtrip 在当前补丁下是否保持干净

测试时间：

- `2026-04-14`

测试环境：

- 服务：`gemma26b`
- 模型：`gemma`
- 部署：`PP=2 / TP=1`
- 当前容器环境：`SERVE_GEMMA4_HEURISTIC_FINAL_ANSWER=true`

测试脚本：

- `tests/performance/gemma4_reasoning_tool_bench.py`

原始结果：

- `results/benchmarks/gemma4-reasoning-tool-8192-20260414.json`

## 口径

- 每个场景运行 `3` 轮
- 请求级 thinking 统一使用 `reasoning_effort=medium`
- 所有场景统一使用 `max_tokens=8192`

覆盖场景：

1. `thinking_greeting_medium`
2. `thinking_math_medium`
3. `tool_roundtrip_medium`

## 结果

| 场景 | content 成功轮数 | reasoning 轮数 | finish_reason | p50 latency |
|------|-----------------:|---------------:|---------------|------------:|
| `thinking_greeting_medium` | 3 / 3 | 3 / 3 | `stop` | 1.67s |
| `thinking_math_medium` | 0 / 3 | 3 / 3 | `length` | 87.76s |
| `tool_roundtrip_medium` | 3 / 3 | 0 / 3 | `stop` | 0.95s |

## 结论

- 对极短问候类请求，`8192` 预算足以让 thinking 路径走到 final answer；当前样本里 `content` 可稳定回填。
- 对基础算术问答，`8192` 预算仍然不足以让模型稳定收束到 final answer；3 轮全部打满 `8192 completion tokens`，`content` 仍为 `null`。
- 当前问题已经不是 parser 抽取不到，而是模型在 thought 中持续展开，直到预算耗尽都没有切到 final channel。
- function call roundtrip 维持稳定：第一轮能正常发起 `tool_calls`，第二轮返回干净 final content，没有 `thought` / `<channel|>` 泄漏。

## 补充观察

- `thinking_greeting_medium` 的 `message.reasoning` 仍偏长，说明当前模板里的“简短思考”提示约束力有限，只是该场景足够短，最终还能落到 final answer。
- `thinking_math_medium` 的 reasoning 末尾出现明显重复，说明单纯放宽 token 预算并不能自然修复“只出 reasoning”问题，反而会放大长思维占用。
- `tool_roundtrip_medium` 的最终 `content` 不是裸 `579`，而是完整句子 `The result of adding 123 and 456 is 579.`；这说明 tool 路径当前更像“稳定返回最终回答”，而不是“强制只回传工具结果字符串”。
