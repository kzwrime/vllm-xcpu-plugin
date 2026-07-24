# vLLM v0.25.1 XCPU Triton 兼容层迁移记录

> 本文是版本特定的语义审计记录。通用接入原则、测试要求和排障方法见
> [vLLM XCPU Triton 兼容性与新算子接入指南](TRITON_COMPATIBILITY_GUIDE.md)。
> 后续版本迁移应新建独立记录，不在本文追加其他版本的结论。

## Rejection sampling kernel 迁移

v0.25.1 将
`vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils._compute_block_stats_kernel`
更名为 `_compute_local_logits_stats_kernel`。本次审计确认参数签名未变，核心语义仍是：

- grid 为 `(num_logits, ceil(vocab_size / BLOCK_SIZE))`，当前
  `BLOCK_SIZE` 固定为 8192；
- bonus token（`expanded_local_pos >= num_speculative_steps`）不写任何输出；
- greedy 请求只写 target local max/argmax，非 greedy 请求写 target
  local max/sumexp；
- `HAS_DRAFT_LOGITS=True` 时额外写 draft local max/sumexp；为 `False` 时
  `draft_logits_ptr` 必须是 `None`，两个 draft stride 必须为 0，draft 输出保持
  未写状态；
- 尾块以 `-inf` mask，argmax token ID 必须包含 block 起始偏移。

对应 dispatcher operator 同步命名为
`mcpu::vllm_rejection_compute_local_logits_stats`，使上游 kernel、adapter 和
C++ dispatcher 保持一一对应。注册项只推进该 kernel 的 source hash、名称和
`source_version`，不批量修改其他 v0.24.0 审计记录。

同一版本还为 `_rejection_kernel` 和 `_resample_kernel` 增加了 block
verification 参数与分支。XCPU dispatcher 已同步实现完整数据流：

1. `_compute_cumulative_log_p_kernel` 对每个请求计算 running joint ratio；
2. full-draft 模式通过 `_compute_local_residual_mass_kernel` 计算各 vocab block
   的 residual mass，one-hot draft 则在 rejection 阶段使用闭式公式；
3. `_rejection_kernel` 使用 block-verification threshold 决定接受长度；
4. `_resample_kernel` 使用 `p_tau` 缩放 target distribution 后构造 residual。

关闭该模式时，`cumulative_log_p_ptr`、`local_residual_mass_ptr` 仍为 `None`，
相应 stride 为 0。两个前置 kernel 也分别注册为 dispatcher operator，adapter
只负责 ABI 校验与转发，不在 Python 层实现 reference 计算。

传统路径新增的 padded draft token（`-1`）保护已由 XCPU 实现覆盖：概率采样先将
token clamp 到 0 以避免越界，但仍通过原始 token 的非负性强制拒绝；greedy
路径则因不可能等于合法 target argmax 自然拒绝。
