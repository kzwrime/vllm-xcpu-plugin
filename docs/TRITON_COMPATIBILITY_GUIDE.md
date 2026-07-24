# vLLM XCPU Triton 兼容性与新算子接入指南

> **迁移记录说明：** 具体版本的语义审计与迁移结论单独维护，不写入本通用指南。
> 当前记录见 [v0.25.1 Triton 兼容层迁移记录](TRITON_MIGRATION_V0.25.1.md)。

本文面向 `vllm-xcpu-plugin` 和 `torch_mcpu` 的维护者，说明如何使用当前 Fake Triton 兼容层、如何接入新的 vLLM Triton kernel，以及升级 vLLM、启用 `torch.compile` 时必须注意的边界。

阅读顺序按日常开发优先级组织：先讲接入步骤和验证方法，再讲注意事项与排障，最后解释内部原理。

## 1. 快速判断：新 kernel 是否适合接入

接入前先确认以下条件：

- vLLM 通过标准 `kernel[grid](...)` 边界启动该 Triton kernel。
- kernel 的输入、输出、原地修改、padding、stride、dtype 和 constexpr 语义能够被明确描述并独立测试。
- XCPU 实现可以暴露为 `torch.ops.mcpu.*` dispatcher operator。
- adapter 仅依赖 launch 时可见的参数，不依赖替换 vLLM Python wrapper、类方法或隐藏的全局 Tensor 映射。
- 该路径确实应由 `torch_mcpu` 接管，而不是已经由 `torch_xcpu` 专用实现、其他插件 patch 或正式 Triton 编译器负责。

出现以下情况时，不要直接加入 registry：

- 只能通过 monkeypatch vLLM wrapper 才能取得完整参数。
- 无法确定原 kernel 的副作用或某些边界语义。
- 需要把未知 kernel 静默回退到 Python 实现。
- kernel 属于当前明确排除的路径，例如 GDN、causal-conv、chunk-delta、MoE、grouped-topk、topk/topp sampling、gumbel、layernorm 或 rotary embedding。
- kernel 的核心输入是 raw pointer 数组，但底层 operator 尚未提供严格的 pointer ABI 和生命周期保护。

当前策略是 fail-closed：未注册、源码漂移、签名漂移、非法 grid 或不支持的 launch metadata 都应立即报错。

## 2. 新 kernel 接入流程

一次完整接入通常涉及两个仓库：

- `torch_mcpu`：定义 Torch dispatcher schema，并实现 XCPU kernel。
- `vllm-xcpu-plugin`：校验 Triton launch，并转换为 `torch.ops.mcpu.*` 调用。

推荐严格按下面顺序开发。

### 2.1 固定并审计 vLLM 参考版本

先为这个 kernel 人工记录本次语义审计对应的产品版本，并阅读以下内容：

1. Triton kernel 函数体；
2. 所有 launch 位置；
3. grid 的计算方式；
4. constexpr 和 launch metadata；
5. wrapper 在 launch 前后的清零、切片、同步或状态更新；
6. 相邻 vLLM 版本的语义变化。

审计时至少记录：

- 每个参数的 dtype、shape、stride、可选性和地址空间；
- 哪些 Tensor 被原地修改；
- padding ID、无效 index、空输入和尾块行为；
- prefill、decode、speculative decode 等模式差异；
- grid 是否属于算法语义，而不只是性能参数。

不要根据 kernel 名称或旧版本实现推断语义。例如 bitmask 方向、padding 值、optional pointer 和 constexpr 取值都可能随 vLLM 升级而改变。

`source_version` 是人工审计结论，不得从 Git tag、当前 branch 名或 commit
祖先关系自动推导。XCPU 的 vLLM 开发分支通过 cherry-pick 和手工移植组成，
版本线不保证线性；Git ref 只能辅助定位源码，不能替代这项人工标记。

### 2.2 在 `torch_mcpu` 中定义 dispatcher operator

在合适的 `torch_mcpu/csrc/aten/vllm_kernels/*.cpp` 文件中定义 schema，并为 `PrivateUse1` 注册实现：

```cpp
TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_example_kernel("
      "Tensor input, "
      "Tensor(a!) output, "
      "int block_size"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_example_kernel", &vllm_example_kernel_impl);
}
```

Schema 是 `torch.compile` 能否正确理解调用的关键部分：

- 原地修改必须使用 `Tensor(a!)` 等 alias annotation。
- optional Tensor 使用 `Tensor?`；被修改的 optional Tensor使用 `Tensor(a!)?`。
- 对动态整数语义确实需要符号化时使用 `SymInt`，不要机械地把所有整数改成 `SymInt`。
- 有返回 Tensor 的 operator 需要准确描述返回值，并提供 Fake/Meta 实现，使 Dynamo/Inductor 能推导 shape、dtype 和 alias。
- schema、C++ 函数签名和 adapter 参数顺序必须一致。

实现层还需要显式检查 XCPU 支持边界，例如 dtype、device、contiguous/stride、维度和数值范围。不要让不支持的输入产生未定义行为。

### 2.3 为 `torch_mcpu` operator 编写独立语义测试

先直接测试 `torch.ops.mcpu.*`，不要用 E2E 代替算子测试。测试应覆盖：

- 参考实现或原 Triton kernel 的差分结果；
- 支持的全部 dtype；
- 最小值、边界值、尾块和非整除 shape；
- padding、空输入、负 index、optional Tensor；
- 非 contiguous 或自定义 stride（如果 ABI 声称支持）；
- 原地修改和未写区域保持不变；
- raw pointer 数组在异步执行期间的生命周期；
- 多 stream 或多 group 情况（如果适用）。

现有集中测试文件是：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24
.venv/bin/python -m pytest -q torch_mcpu/tests/test_vllm_kernel_launch.py
```

如果新增 C++ 源文件或 schema，先按 `torch_mcpu` 的构建流程重新编译并安装，再运行测试。只更新源码但继续使用旧 `.so` 会产生非常难定位的假失败。

### 2.4 在插件中编写 launch adapter

在 `vllm_xcpu_plugin/fake_triton/vllm_kernels.py` 中增加 adapter。adapter 的职责只有三类：

1. 从 `KernelLaunch` 读取已经绑定的原始 Triton 参数；
2. 校验 grid、constexpr、stride、optional 参数和 launch metadata；
3. 调用一个或少量 `torch.ops.mcpu.*` operator。

示例：

```python
def _example_kernel(launch: KernelLaunch) -> None:
    args = launch.arguments
    rows = args["input_ptr"].shape[0]

    _expect_grid(launch, (rows,))
    _expect(args["BLOCK_SIZE"] == 1024, "BLOCK_SIZE must be 1024")

    torch.ops.mcpu.vllm_example_kernel(
        args["input_ptr"],
        args["output_ptr"],
        args["BLOCK_SIZE"],
    )
```

adapter 不应：

- 执行目标 kernel 的完整 Python reference 实现；
- 修改 vLLM wrapper；
- 捕获异常后静默回退；
- 在全局表中缓存 Tensor，以便从裸地址反查对象；
- 忽略 grid 或 constexpr，只因为当前 E2E 恰好使用一个固定值。

只要 adapter 最终调用的是注册到 Torch dispatcher 的 `torch.ops.mcpu.*`，Dynamo 就可以把它作为 Torch operator 捕获，Inductor/AOTI 在编译产物中保留相应 dispatcher 调用。

### 2.5 添加版本锁定注册项

在同一文件的 `_KERNELS` 中加入：

```python
(
    (
        "vllm.some.module",
        "_example_kernel",
        "<source hash>",
        "<signature hash>",
        "v0.24.0",  # 仅这一条 kernel 的人工审计版本
        _example_kernel,
        ("num_warps",),  # 没有 metadata 时使用 ()
    ),
)
```

source hash 和 signature hash 可以从安装 Fake Triton 后的 kernel 对象读取：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24
PYTHONPATH=vllm-xcpu-plugin .venv/bin/python - <<'PY'
import importlib

from vllm_xcpu_plugin.fake_triton import install_fake_triton

install_fake_triton(replace_existing=True)
module = importlib.import_module("vllm.some.module")
kernel = module._example_kernel
print("source:", kernel.source_hash)
print("signature:", kernel.signature_hash)
print("qualname:", kernel.qualname)
PY
```

哈希只能在完成源码和语义 review 后写入。vLLM 升级导致哈希变化时，禁止仅运行脚本刷新哈希；必须重新检查函数体、签名和所有 launch 点。

`_MANIFEST_BASELINE_VERSION` 只说明清单最初建立时的背景版本，不参与注册，
也不是任何 kernel 的默认值。每条 `_KERNELS` 记录必须直接写自己的
`source_version` 字面量。某个 kernel 完成新版本审计后，只修改它自己的
版本、source hash 和 signature hash；禁止通过修改 baseline 批量放行。

绕过 Fake Triton 的替代路径登记在
`vllm_xcpu_plugin/upstream_compatibility.py`。attention、causal-conv、GDN/FLA、
Gumbel/temperature、top-k/top-p、grouped-topk 和 topk-softmax 的上游
Triton kernel 或语义入口都必须在真正安装 patch 前通过同样的指纹检查。
报错后需要更新 XCPU 实现和差分测试，再单独推进对应记录的版本与哈希。
复合算子按插件真正发生替换的边界登记。例如 ChunkGatedDeltaRule 检查
`forward_native` 和公开 FLA wrapper，不把 `solve_tril`、`wy_fast`、
`chunk_delta_h` 等仅由 native wrapper 间接调用的内部 kernel 分别登记，
避免纯内部调度或性能重构被误报为多个独立替代算子漂移。

如果 launch 使用 `num_warps`、`num_stages` 等 metadata，必须在注册项中逐个 allowlist。当前 runtime 支持的 metadata 集合见 `_TRITON_LAUNCH_METADATA`，未声明的 metadata 会被拒绝。

### 2.6 增加插件层测试

至少增加以下测试：

- 目标模块导入后，kernel 确实是 `FakeJITFunction`。
- `register_vllm_kernels()` 能完成版本锁定注册，重复注册保持幂等。
- adapter 能用代表性输入 dispatch 到 `torch.ops.mcpu.*`，并验证输出。
- `launch_counts()` 中对应 kernel 的计数增加。
- 错误 grid、constexpr、metadata、optional 参数或 stride 被明确拒绝。
- 错误源码哈希或签名哈希触发 `KernelVersionError`。
- 未注册 kernel 触发 `UnknownKernelError`。

运行插件测试：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24
PYTHONPATH=vllm-xcpu-plugin .venv/bin/python -m pytest \
  -q vllm-xcpu-plugin/tests/fake_triton
```

提交前还应运行：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24/vllm-xcpu-plugin
./scripts/check.sh
```

### 2.7 运行 eager 和 compile E2E

算子和 adapter 测试通过后，再进行端到端验收：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24/vllm_scripts

# 验证普通运行路径
./run_vllm_test.sh \
  -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh

# 验证 torch.compile / AOTI 路径
./run_vllm_test.sh \
  -e presets/serial/Qwen3-0.6B_dp1_tp1_compile.sh
```

E2E 通过后检查 worker 日志和 registry 计数，确认目标 kernel 实际启动。E2E 成功只能证明当前 workload 可用，不能替代差分测试，也不能证明所有注册 kernel 都被覆盖。

## 3. 常见注意事项

### 3.1 FakeBackend 不是 Triton 编译 backend

`installer.py` 中的 FakeBackend 只提供 TorchInductor 初始化和缓存 key 所需的最小元数据 ABI：

- `driver.active.get_current_target()`；
- `triton.compiler.compiler.make_backend()`；
- 稳定的 `backend.hash()`。

它不解析 Triton AST，不生成代码，也不执行 Triton kernel。不要为了通过新的 Inductor 调用而逐步把它扩展成不完整的 Triton compiler。

真正的 compile 兼容路径是 adapter 调用 `torch.ops.mcpu.*`。如果错误栈进入 Triton codegen、lowering 或 binary compilation，优先检查为什么调用没有转换成 Torch operator，而不是继续扩充 FakeBackend。

### 3.2 Torch schema 必须准确表达副作用

Fake Triton 接管的 vLLM kernel 多数会原地写 Tensor。如果 schema 缺少 alias annotation，Dynamo/Inductor 可能删除、重排或错误复用调用。

重点检查：

- 每个被写 Tensor 是否标记为 `Tensor(a!)`；
- 多个输出是否使用不同 alias set；
- optional 输出是否同时表达 optional 和 mutation；
- 未写区域是否必须保持不变；
- raw pointer 容器背后的真实输出是否能被 dispatcher 正确建模。

对于无法用普通 Tensor schema准确表达的 pointer ABI，需要单独评估 compile 行为和生命周期，不能假设 eager 可用就等价于 compile 可用。

### 3.3 raw pointer 数组必须保持生命周期

部分 vLLM launch 参数是保存地址的 Tensor，而不是普通 Tensor list。adapter 在 launch 边界只能看到真实参数，不能通过原 wrapper 恢复 Python Tensor 对象。

此类 operator 必须：

- 明确地址所属设备、dtype、对齐和 stride；
- 在异步 kernel 完成前保持被引用存储有效；
- 使用 `KernelPointerMemoryGuard` 或等价的底层生命周期机制；
- 覆盖 single/multi-group、多 stream、空 group 和异常 index 测试；
- 禁止维护隐藏的进程级“地址到 Tensor”映射。

### 3.4 grid 和 constexpr 是 ABI 的一部分

grid、`BLOCK_SIZE`、`NUM_TOPK`、`CP_SIZE` 等参数可能决定数据布局和写入范围。adapter 应验证它们是否与底层 operator 的假设一致。

只有确认某个值纯粹影响 Triton 性能、不会影响语义时，才可以不传给 `torch_mcpu`；即使如此，也建议验证其允许范围，防止 vLLM 升级后悄悄改变算法。

### 3.5 保留 wrapper 前后的语义

Fake Triton 只替换 kernel launch，不替换 wrapper。wrapper 中以下操作通常必须继续由原 vLLM 执行：

- launch 前清零；
- 输出 Tensor 分配；
- active/padded 行切分；
- launch 后的 view、slice 或状态更新；
- stream/event 管理。

将这些逻辑重复搬进 adapter，可能导致 eager 与 compile 行为不同，也会破坏未来与正式 Triton 产物在同一 launch 边界比较的目标。

### 3.6 vLLM 升级必须重新做语义审计

版本锁定失败是保护机制，不是需要绕过的安装问题。升级流程应是：

1. 对比目标 kernel 函数体和签名；
2. 对比所有 launch 点；
3. 检查 constexpr、metadata、padding 和 optional 参数；
4. 更新 `torch_mcpu` 实现与差分测试；
5. 更新 adapter 校验；
6. 最后只更新该 kernel 自己的 source/signature hash 和 `source_version`，
   不修改 `_MANIFEST_BASELINE_VERSION`；
7. 重跑算子、插件、eager E2E 和 compile E2E。

### 3.7 不要把 Fake Language 占位对象当作任意兼容答案

Torch 和 vLLM 可能在导入或 compile 阶段检查 Triton API 的对象类型。例如 TorchDynamo 会把 `triton.language.dtype` 放入 `isinstance` 的类型集合，因此它必须是真实 Python 类型。

新增占位符时要确认调用方需要的是：

- module；
- class/type；
- callable；
- singleton；
- 仅用于 annotation 的 symbol。

宽泛的 `__getattr__` 只能帮助 kernel 定义期导入，不能保证返回对象满足 Torch 的运行时类型契约。

### 3.8 保持非 XCPU 环境隔离

Fake Triton 由 XCPU platform plugin 在早期安装，并应保持幂等。修改安装器时必须验证：

- 非 XCPU 平台不会被替换；
- 已安装 Fake Triton 时重复调用无副作用；
- 已导入其他 Triton runtime 时不会在未授权情况下混用模块；
- 存在正式 Triton 包时，compiler/runtime 支持模块能够按设计透传。

## 4. 排障顺序

### 4.1 `UnknownKernelError`

说明 kernel 已被 Fake Triton 装饰，但没有注册。检查：

- kernel 是否属于预期接管范围；
- 模块名和函数名是否准确；
- `register_ops()` 是否执行；
- `_KERNELS` 是否包含该项。

不要为消除错误增加通用 fallback。若 kernel 不应接管，应明确走排除路径；若应接管，应完成 operator、adapter 和测试。

### 4.2 `KernelVersionError`

说明源码、签名或重复注册发生漂移。先 diff vLLM kernel 和 launch 点，再更新实现和测试。不要直接替换哈希。

### 4.3 `InvalidLaunchError`

错误通常来自参数绑定、grid、metadata 或 adapter 的显式语义校验。将日志中的实际值与目标 vLLM commit 的 launch 逻辑对照；如果新增值合法，应先扩充差分测试，再调整 adapter。

### 4.4 TorchInductor 在 Triton import/探测阶段失败

如果错误发生在模型图生成前，例如缺少 `make_backend`、driver target 或类型对象，属于导入/元数据兼容问题。只补齐调用方实际需要的最小 ABI，并增加 installer 回归测试。

如果错误要求 AST、lowering、codegen 或 binary compilation，则说明执行路径没有停留在 `torch.ops.mcpu.*` 边界，应检查 adapter 和 Dynamo 图捕获。

### 4.5 eager 通过但 compile 失败

按以下顺序检查：

1. Torch schema 的 mutation/alias/optional 标注；
2. 返回 Tensor 是否有 Fake/Meta 实现；
3. adapter 是否真的调用 `torch.ops.mcpu.*`；
4. Python 标量或全局对象是否生成了非法 Dynamo guard；
5. AOTI 运行时是否加载了包含 operator 注册的扩展；
6. C++ wrapper 的 include、link 和 rpath 是否包含 `torch_mcpu`/`torch_xcpu` 依赖；
7. 是否混用了旧构建产物或旧缓存。

必要时检查 FX graph 或生成的 AOT wrapper，确认目标调用表现为 `torch.ops.mcpu.<op>`，而不是 Triton JITFunction 或任意 Python fallback。

## 5. 工作原理

### 5.1 激活顺序

XCPU platform plugin 的入口是 `xcpu_platform_plugin()`：

1. 早期调用 `install_fake_triton(replace_existing=True)`；
2. 导入 `torch_mcpu`，加载 dispatcher operator；
3. vLLM 导入包含 `@triton.jit` 的模块时，kernel 被包装为 `FakeJITFunction`；
4. general plugin 的 `register_ops()` 调用 `register_vllm_kernels()`；
5. registry 按 module、qualname、源码哈希和签名哈希绑定 adapter。

未注册的 Triton 函数可以完成模块定义，但一旦启动就会 fail-closed。

### 5.2 eager 执行路径

```text
vLLM wrapper
  -> kernel[grid](*args, **kwargs)
  -> FakeJITFunction / KernelRegistry.dispatch
  -> 参数绑定、grid/metadata/版本校验
  -> adapter
  -> torch.ops.mcpu.<operator>
  -> PrivateUse1 implementation
  -> XCPU kernel
```

registry 只在 adapter 成功返回后增加 launch count。

### 5.3 `torch.compile` 执行路径

```text
TorchDynamo tracing
  -> 执行 Fake Triton 的 Python launch/adapter
  -> 捕获 torch.ops.mcpu.<operator>
  -> Inductor/AOTI 保留 Torch dispatcher 调用
  -> 编译产物运行时调用已注册的 PrivateUse1 implementation
```

这里的“透传”发生在 `torch.ops.mcpu.*` 边界，而不是 FakeBackend。FakeBackend 只让 TorchInductor 的 Triton 探测和缓存元数据流程能够初始化；它不参与目标 operator 的 codegen 或执行。

### 5.4 版本锁定和严格失败

`FakeJITFunction` 在装饰时计算：

- 规范化源码 AST 的 SHA-256；
- Python signature 的 SHA-256；
- module + qualname。

注册和每次 dispatch 都会检查这些信息。这样做的目的，是防止 vLLM 升级后继续运行一个“名称相同但语义已变”的 XCPU 实现。

严格失败让兼容性问题尽早暴露在 kernel launch 边界，避免错误结果被误认为性能问题或随机运行时故障。

## 6. 接入完成检查表

- [ ] 固定并记录 vLLM commit。
- [ ] 审计 kernel 函数体、签名和全部 launch 点。
- [ ] 明确输入、输出、副作用、grid、constexpr、metadata 和边界语义。
- [ ] 在 `torch_mcpu` 注册准确的 dispatcher schema 和 `PrivateUse1` 实现。
- [ ] 为返回 Tensor 的 operator 提供必要的 Fake/Meta 实现。
- [ ] `torch_mcpu` reference/differential 测试通过。
- [ ] 插件 adapter 只做校验和 `torch.ops.mcpu.*` 调用。
- [ ] `_KERNELS` 中写入 review 后的源码/签名哈希。
- [ ] 未注册、版本漂移、非法 grid/metadata 测试通过。
- [ ] Fake Triton 全套测试和代码检查通过。
- [ ] eager E2E 通过并确认目标 kernel 实际启动。
- [ ] compile E2E 通过；必要时确认 FX/AOT wrapper 中保留目标 Torch operator。
- [ ] 更新兼容矩阵、测试记录和已知限制。
