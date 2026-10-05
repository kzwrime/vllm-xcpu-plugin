# Mamba align postprocess 增量回移审计（2026-10-05）

本记录对应 `postprocess_mamba_fused_kernel` 的 #50432 回移。
通用规则见 [兼容性指南](TRITON_COMPATIBILITY_GUIDE.md)。

## 源码身份

| 字段 | 值 |
|---|---|
| 人工审计版本标签 | `v0.25.1` |
| 本地源码仓库 | 开发套件的 `vllm` 仓库，`dev_mcpu_v0.25.1` 开发线 |
| 本地源码 commit | `377c203f180c9048bbff58d331167142f15ec07e` |
| `source_version` | `v0.25.1; local=377c203f180c; upstream=c2881ce60302` |
| 上游仓库 | `vllm-project/vllm` |
| `upstream_commit` | `c2881ce60302b5455867d2c29cdfae5fbeddecac`（#50432） |
| source hash | `4ea8ff600d9b0df472320d75c0f7b3a63f12a41991f6478173fb6ac0b45ec2c3` |
| signature hash | `34919a7a6f15f7639eb0ee59abebf561f2c48b1243eee651aa03adeafd7ec97b` |

修复实现在本地 `ea378f497e812cb2a80feefee6e5af6d74ead395` 中提交。
本次审计取当前已提交HEAD `377c203f180c`；后者只增加块回收修复及其测试，
不改变已审阅kernel/调用点。已确认对应源码无未提交修改。
来源字符串先前的 `v0.25.1+c2881ce603` 容易将上游来源误读为本地源码身份，
因此改为版本标签、显式local及upstream三个字段；不改变已审阅的两个指纹。

## 审阅范围与语义

已阅读 `vllm/v1/worker/mamba_utils.py` 中的kernel、V1
`run_fused_postprocess`、V2 `run_fused_postprocess_align` 及其调用约定。

旧V2路径让第0个state把输入accepted计数重置为1，其他state可能读到新值，
使不同层的checkpoint决策不一致。现在V2 wrapper复制完整请求槽数组作为只读
决策快照，kernel只写独立output。必须复制整个数组，因为idx_mapping可引用
前num_reqs之外的槽位。函数签名未变，但可选指针契约发生了变化：

- V1继续使用非空output；batch行与请求一一对应。
- V2 align要求idx_mapping及output非空，scheduled/draft指针为空；只读取
  输入快照，在同块分支将live output置1。adapter严格拒绝旧V2空output。
- 负mapping槽跳过；三层状态必须使用相同的原accepted值。
- MCPU保留旧operator直接调用的in-place兼容，但在state循环外读取一次
  accepted；支持独立output。原有memmove继续处理重叠状态拷贝。

配套 `torch_mcpu` 提交：`e6508c1b7cfca805e6d1fcab90fe9461797186c3`（`fix(vllm_kernels): preserve Mamba
acceptance across layer states`）。必须同步部署vLLM、插件和后端实现。

## 验证与边界

在开发套件根目录的 `vllm-xcpu-plugin` 子目录执行：

```bash
../.venv/bin/python -m pytest ../torch_mcpu/tests/test_v2_model_state_kernels.py -q
../.venv/bin/python -m pytest tests/fake_triton/test_v2_model_state.py tests/fake_triton/test_installer.py -q
```

安装后的MCPU算子5 passed，插件8 passed。新增测试使用3个state、请求槽2、
padding槽-1，验证所有层的checkpoint、live计数重置及输入快照不变。
原算子在该回归上失败；修复后通过。来源标记规范化后完整插件测试8 passed（54.70秒）；最终三字段标签再跑
installer测试4 passed（23.15秒）。

结合独立的Mamba块回收修复（vLLM #55450），真实Qwen Code 0.21.0 TUI
在Qwen3.8-27B-FP8、TP2、DFlash2/spec5的eager和compile上完成
`nihao`→`hello`。两轮思考/回答正常，各模式命中17,952缓存tokens、最高
并发2、抢占0。原始证据和实验记录位于开发套件
`docs/investigations/qwen_code_021_multiturn/`。

NPU未实测；不能把此审计视为NPU原始乱码的唯一归因或全部平台兼容性认证。
