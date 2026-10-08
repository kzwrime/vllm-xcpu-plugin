# Qwen GDN 混批门控审计（2026-10-08）

## 来源与适用范围

- 沿用已确认标签：`v0.25.1`；不新建 release tag，不代表完整上游版本兼容。
- 本地 vLLM 已提交快照：`8638eb14cadd564f4c323bfa712abb667a8c19fc`。
- 上游来源：vllm-project/vllm `5af7c8dad798bf899813f8f3c6b9eaf08a748e17`，#51812。
- 配套 torch_xcpu：`704cf97c75a062c4dba5fde85d64b42a87a7eeea`。
- 审计标签：`v0.25.1; local=8638eb14cadd; upstream=5af7c8dad798`。

本次改变的是 `QwenGatedDeltaNetAttention._forward_core` 调用数据重排，以及
`torch_xcpu::qwen_gdn_attention_core` / `qwen_gdn_attention_qkvzba` 的融合路径。
底层 Triton recurrent kernel 函数和签名没有变化，故不刷新
`fake_triton/vllm_kernels.py` 或 `upstream_compatibility.py` 中无关条目的版本与指纹。
现有 kernel 指纹本身不能发现调用方错误使用另一请求的数据。

便于独立核查，当前 `_forward_core` 的规范化指纹（不是新加入的运行时白名单）为：

- source：`1f072747b06fbc2196758ff8d01d805b85a371960defebe62f903785b469a6f9`
- signature：`2e41f8dacc93102f73293352a1fa8eb282163fefbde3da934cf6a0b11f36cdaf`

## 审阅与修复

已逐行审阅上游 #51812、本地 Python `_forward_core`、C++
`qwen_gdn_attention_core_xcpu` 和 `run_recurrent_update`、GDN metadata 的
spec token 索引，以及 V2 `sort_batch_req_ids`。

DFlash2/spec5 的 decode query 长度为 6。某个 prefill 的最后一块恰好也是 6
时，V2 长度排序的稳定顺序允许它出现在 spec decode 之前。旧实现对 QKV
使用 `spec_token_indx`，却把原始全批的 a/b 传给 recurrent kernel。
后者按已经压紧的 spec query 索引读取 a/b，因而混用请求的门控值，同时污染
输出和 SSM 缓存。该遗漏早于 1006，不能声称是 1006 新引入。

修复要求：所有混批 spec 输入使用相同的 token 索引重排；纯 spec 路径继续
使用原输入。Python 和融合 C++ 两条路径都必须修复；仅回移 Python 不足以
覆盖默认 XCPU full core。Torch schema、缓存布局和算子参数结构均未改变。

## 验证

- Python routing 回归修复前失败，修复后通过；插件 GDN metadata 与 layer 文件
  合计 227 passed。
- C++ 24 组 mixed-versus-isolated 数值回归：两个入口、两种顺序、新/续算
  prefill、accepted=1/3/6。修复前全部 12 个 prefill-first case 失败，另 12
  个通过；修复后全部通过，输出、conv 状态和 SSM 状态逐元素一致。
- attention 与 conv handoff 两文件共 42 passed。
- Qwen3.8-27B-FP8 TP2 DFlash2，prefix ON：标准
  `long_prefill_token_threshold=128` 产生合法的 6-token prefill + 6-token spec
  批次；观察器不改变调度结果。旧版 4/4 greeting 混批后输出改变，配套修复后
  4/4 与单独执行的所有 token 和 token logprobs 完全一致。
- prefix OFF：原始对照的prefill分块不同（54/53 vs 48+6/5），不能按bitwise
  相等判定。补充相同分块的四对请求后，message、token字符串/字节和logprob
  全部相等，并与先前相同分块的混批结果跨运行完全相等。
- 真实qwen-code 0.21.0同一TUI进程连续nihao及七次hello：ON/eager温度0与
  OFF/compile温度1各八轮主会话通过，均HTTP200、SSE DONE、stop，无流错误，
  思考/回答经逐轮审阅正常。ON有实际prefix命中，OFF命中为0，DFlash计数非零。
  为隔离1006发布行为，模型实验固定平台插件5063475；未将之后3b4f03e的通信
  变更混入对照。插件单测和mypy使用当前工作树。

以上验证运行于 x86/MCPU。真实 NPU、现场的随机数字/城市循环尚未直接验证。
NPU 若替换融合 attention core，应在相同 token 轴上重排 a/b 与 QKV；这与
1006 的首次 chunk 初始化 mask 和 conv 三槽历史修复是独立要求。

完整多轮实验与运行说明保存在开发套件本轮工作目录
`docs/investigations/dflash2_20261008_regression/`；按用户要求不为该套件创建
本轮 commit，原始日志不加入暂存区。
