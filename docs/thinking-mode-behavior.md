# Gemma 4 Thinking Mode Behavior on vLLM 0.19.0

本文记录 `Gemma-4-26B-A4B-it-FP8` 在当前项目默认部署上的 thinking / non-thinking 实测行为。

补充说明：

- 本文前半部分记录的是最初基于 `chat_template_kwargs.enable_thinking` 的专项测试
- 当前仓库后续又增加了一层 launcher-managed 本地 template patch
- 在这层本地补丁下，OpenAI-compatible `reasoning_effort` 也已被接到 Gemma 4 模板上

测试时间：

- `2026-04-14`

测试环境：

- vLLM: `0.19.0`
- 服务: `gemma26b`
- 模型目录: `kuohao/gemma-4-26B-A4B-it-FP8`
- 并行策略: `PP=2 / TP=1`

## 结论

当前默认部署下，`gemma26b` 是“默认不思考”的。

依据：

- 当前服务默认始终带 `--reasoning-parser gemma4`
- 默认 render 出来的 prompt 不含 `<|think|>` system turn
- 默认问答与默认 tool use 都表现为 non-thinking 路径

但有一个重要兼容性点：

- 在当前 vLLM OpenAI 接口里，实测生效的是顶层参数 `chat_template_kwargs.enable_thinking=true`
- `extra_body.chat_template_kwargs.enable_thinking=true` 在当前栈里没有观察到效果
- 在当前仓库的 launcher-managed 本地 patch 下，`reasoning_effort=medium|high` 也会切到 thinking
- 在当前仓库的 launcher-managed 本地 patch 下，`reasoning_effort=none|low` 会保持 non-thinking

补充更新：

- 当前仓库还额外加了一层本地 Gemma4 reasoning parser 补丁
- 这层补丁已经可以把 thinking + tool roundtrip 第二轮里的 `thought` 从 `message.content` 中剥离
- 对普通 thinking 问答，如果模型没有明确输出 thinking 结束边界，当前实现会保守返回 `reasoning != null` 且 `content = null`
- 如显式打开 `SERVE_GEMMA4_HEURISTIC_FINAL_ANSWER=true`，parser 会额外尝试从 `Final Answer:` / `最终答案:` 这类尾部标记中回填 `content`
- 也就是说，当前优先保证“不把 thought 继续泄漏到 content”，而不是对缺失边界的回复做激进猜测

## Prompt Render 实测

使用接口：

- `POST /v1/chat/completions/render`

### 1. 默认请求

解码后的 prompt:

```text
<bos><|turn>user
What is 17 * 19?<turn|>
<|turn>model
<|channel>thought
<channel|>
```

说明：

- 默认请求会进入 non-thinking 模板
- model turn 前会补一个空的 thought block

### 2. 顶层 `chat_template_kwargs.enable_thinking=true`

解码后的 prompt:

```text
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
What is 17 * 19?<turn|>
<|turn>model
```

说明：

- 顶层 `chat_template_kwargs.enable_thinking=true` 会真实注入 `<|think|>`
- 这是当前实测有效的 thinking 开关方式

### 3. `extra_body.chat_template_kwargs.enable_thinking=true`

实测与默认 render 结果一致。

说明：

- 在当前栈里，没有观察到它能改变 chat template

### 4. `reasoning_effort`

在当前仓库的 launcher-managed 本地 patch 下：

- `reasoning_effort=low` 的 render 结果与默认请求一致
- `reasoning_effort=medium` 会真实注入 `<|think|>`

解码后的 prompt 对比：

`reasoning_effort=low`

```text
<bos><|turn>user
What is 17 * 19?<turn|>
<|turn>model
<|channel>thought
<channel|>
```

`reasoning_effort=medium`

```text
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
What is 17 * 19?<turn|>
<|turn>model
```

## 响应层实测

使用接口：

- `POST /v1/chat/completions`

测试问题：

- `Solve 123 * 456. Return only the final number.`

### 服务模式 A: `SERVE_ENABLE_THINKING=false`

#### 默认请求

结果：

- `content = "56088"`
- `finish_reason = "stop"`

#### 顶层 `chat_template_kwargs.enable_thinking=true`

结果：

- model 会输出 thought 内容
- thought 没有被拆到独立字段
- thought 直接泄漏到了 `message.content`
- 本次实测 `finish_reason = "length"`

典型返回片段：

```text
thought
The user wants to multiply 123 by 456 ...
```

说明：

- 这表示 prompt 已经切到 thinking 模式
- 但服务端没有把 thinking 内容解析成结构化字段

#### `extra_body.chat_template_kwargs.enable_thinking=true`

结果与默认请求一致：

- `content = "56088"`

#### `reasoning_effort=low`

结果与默认请求一致：

- `content = "56088"`

#### `reasoning_effort=medium`

结果：

- prompt 会切到 thinking 模式
- thought 仍没有被拆到独立字段
- thought 仍会直接泄漏到 `message.content`

本次实测返回片段：

```text
thought
The objective is to multiply 123 by 456.
```

## 服务模式 B: `SERVE_ENABLE_THINKING=true`

### 默认请求

结果仍然是：

- `content = "56088"`
- `reasoning = null`

说明：

- 仅仅把服务端 `SERVE_ENABLE_THINKING=true` 打开，不会让默认请求自动进入 thinking 模式
- 默认请求依然需要显式注入 `<|think|>` 才会真的切换

### 顶层 `chat_template_kwargs.enable_thinking=true`

结果：

- thought 仍然直接出现在 `message.content`
- `message.reasoning = null`

说明：

- 在当前 `vLLM 0.19.0 + gemma4 parser` 组合下，`--reasoning-parser gemma4` 没有把这类返回拆到 OpenAI 响应中的独立 reasoning 字段

这也意味着：

- 即使当前改用 `reasoning_effort=medium|high` 作为 OpenAI-compatible thinking 开关
- 底层兼容性限制仍然相同
- thought 依然不会被稳定拆到独立 reasoning 字段

## Tool Use 实测

测试流程：

1. 第一轮让模型调用 `add(a=123, b=456)`
2. 第二轮回填 `role=tool` 结果 `579`
3. 观察最终 assistant 回复

### 默认 non-thinking 请求

结果：

- 第一轮正常返回 `tool_calls`
- 第二轮正常返回 `content = "579"`

### 顶层 `chat_template_kwargs.enable_thinking=true`

结果：

- 第一轮仍能正常发起 `tool_calls`
- 第二轮返回：

```text
thought
<channel|>579
```

说明：

- 当前补丁已经修复了 non-thinking 模式下的标签泄漏
- 但 thinking 模式下，tool roundtrip 仍然会把 thought/channel 标签带进 `content`

### `reasoning_effort=low`

结果：

- 第一轮正常返回 `tool_calls`
- 第二轮正常返回 `content = "579"`

### `reasoning_effort=medium`

结果：

- 第一轮仍能正常发起 `tool_calls`
- 第二轮返回：

```text
thought
<channel|>579
```

## 本次补丁结论

这次已修复的是：

- 上游新版 `chat_template.jinja` 在 non-thinking + tool roundtrip 下的 `thought\n<channel|>` 泄漏问题

本次未解决的是：

- 顶层 thinking 打开后，thought 仍未被 vLLM 结构化到独立 reasoning 字段
- thinking + tool roundtrip 仍有内容泄漏
- `reasoning_effort=medium|high` 只是把请求级语义映射到了 thinking 模板，并没有解决响应解析层问题

## 当前推荐用法

如果你的目标是稳定托管和干净的 OpenAI 兼容输出，当前建议：

- 默认保持 `SERVE_ENABLE_THINKING=false`
- 默认请求不要传 thinking 开关
- 需要稳定 tool use 时，优先走 non-thinking 路径

如果你要尝试显式 thinking，当前建议：

- 优先使用 `reasoning_effort=medium` 或 `reasoning_effort=high`
- `reasoning_effort=none` / `reasoning_effort=low` 可视为 non-thinking
- 使用顶层 `chat_template_kwargs.enable_thinking=true`
- 不要依赖 `extra_body.chat_template_kwargs.enable_thinking=true`
- 默认仍按“保守拆分”理解 thinking 返回
- 如需验证 heuristic 回填，可显式打开 `SERVE_GEMMA4_HEURISTIC_FINAL_ANSWER=true`
- 对应 smoke 可运行 `python3 tests/reasoning/check_gemma4_reasoning_split.py --check-heuristic-sample`

## 后续建议

如果后面要继续把 thinking 模式做干净，建议优先沿这两条线继续排查：

1. 检查 `vLLM 0.19.0` 的 `gemma4` reasoning parser 是否完整覆盖 Gemma 4 最新模板输出
2. 继续缩小“默认 medium 问答在低 `max_tokens` 下仍到不了 final answer”的行为差距，优先从模板提示强度与 answer marker 稳定性入手
