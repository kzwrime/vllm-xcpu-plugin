# vLLM MoE Workspace 与 torch.compile 行为差异技术报告

本文记录定制版 vLLM xcpu 开发套件中，MoE + `torch.compile` + workspace 组合下出现巨大
`cpp_fused_view_2` copy 的分析、验证和修复过程。

重点结论：

- 0.19 中耗时很长的 `cpp_fused_view_2` 不是清零，而是 workspace copy-in。
- 这个 copy 的直接原因是：compile 时 workspace 已经由 profiling 阶段预先分配，并被 Dynamo hoist 成 graph input。
- `fused_moe_compute_bf16` 和 `moe_finalize_bf16` 共用同一个 buffer 是正确行为；这个 buffer 是 `fused_out`，不是 prepare 的 `recv_hidden_states`。
- 0.14 没有类似问题，关键不是 `WorkspaceManager` 切片算法不同，而是 0.14 CPU-only 路径没有在 compile 前先跑一次会分配 MoE workspace 的 memory profiling。
- 修复策略是在保留 `determine_available_memory()` 的前提下，在 profiling 完成后、compile warmup 前释放 profiling 创建的 workspace 引用，让 compile 重新捕获图内 allocation。
- 端到端 bench 验证后，新的 `torch_compile_cache_opt` 中已经没有 `cpp_fused_view_2` / `cpp_fused_copy__3`。

## 1. 相关文件

当前 0.19 工作区内的关键文件：

- `vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py`
- `vllm/vllm/v1/worker/workspace.py`
- `vllm-xcpu-plugin/vllm_xcpu_plugin/worker/worker_v1.py`
- `vllm-xcpu-plugin/vllm_xcpu_plugin/platform.py`
- `vllm-xcpu-plugin/vllm_xcpu_plugin/layers/fused_moe/mpi_alltoallv_prepare_finalize_v2.py`
- `vllm_scripts/torch_compile_cache_opt/`
- `vllm_scripts/torch_compile_cache_naive/`

用于对照的 0.14 文件使用相对路径描述：

- `vllm-xcpu-dev-kit-0.14/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py`
- `vllm-xcpu-dev-kit-0.14/vllm/vllm/v1/worker/cpu_worker.py`
- `vllm-xcpu-dev-kit-0.14/vllm_scripts/torch_compile_cache/`

## 2. 初始现象

0.19 optimized AOTI C++ 中观察到一个耗时较长的 kernel：

```cpp
cpp_fused_view_2((const at::BFloat16*)(buf68.data_ptr()),
                 (at::BFloat16*)(buf77.data_ptr()));
```

该函数名包含 `view`，但生成内容实际是 vectorized copy，不是清零：

```cpp
loadu(in_ptr0 + x0).store(out_ptr0 + x0);
```

在旧的 0.19 optimized cache 中，相关数据流是：

```cpp
auto arg27_1 = std::move(inputs[27]);  // persistent workspace graph input
buf68 = view_dtype(arg27_1, bf16);
buf77 = empty_strided(..., bf16);

cpp_fused_view_2(buf68, buf77);        // copy-in

aoti_torch_mcpu_fused_moe_compute_bf16(... buf77 ...);
aoti_torch_mcpu_moe_finalize_bf16(... buf77 ...);

cpp_fused_copy__3(buf77.view(uint8), arg27_1);  // copy-back
```

在旧的 0.19 naive cache 中，copy 规模约为：

- `16785408` 个 bf16
- 对应约 `8196 * 2048`

在旧的 0.19 optimized cache 中，copy 规模约为：

- `4202496` 个 bf16
- 对应约 `2052 * 2048`

因此性能负担来自大块 workspace 的 copy-in/copy-back，而不是一次真正的 memset/zero-fill。

## 3. MoE Python 侧数据流

0.19 MoE 主流程在 `modular_kernel.py`。简化后是：

```python
def forward(...):
    prepare_result = self._prepare(...)
    fused_out = self._fused_experts(...)
    return self._finalize(output, fused_out, ...)
```

因此 C++ 中：

```cpp
aoti_torch_mcpu_fused_moe_compute_bf16(... buf77 ...);
aoti_torch_mcpu_moe_finalize_bf16(... buf77 ...);
```

这两个 op 使用同一个 `buf77` 是合理的：

- `fused_moe_compute_bf16` 写 `fused_out`
- `moe_finalize_bf16` 读 `fused_out`

这不是 compute 和 finalize 之间发生了额外拷贝。真正额外的是 compute 前从 graph input workspace copy 到 private temp，以及末尾 copy-back 到 graph input workspace。

`buf77` 也不是 alltoallv prepare 阶段的巨大 `recv_hidden_states`。prepare 的静态 shape buffer 可能很大，但当前这个 copy 的对象来自 `_allocate_buffers()` 中的 `fused_out/common_workspace`。

## 4. `_allocate_buffers()` 与 common workspace

0.19 的 `_allocate_buffers()` 中，`workspace13` 和 `fused_out` 共享一个 `common_workspace`：

```python
max_shape_size = max(prod(workspace13_shape), prod(fused_out_shape))
common_workspace, workspace2 = current_workspace_manager().get_simultaneous(
    ((max_shape_size,), workspace_dtype),
    (workspace2_shape, workspace_dtype),
)
workspace13 = _resize_cache(common_workspace, workspace13_shape)
fused_out = _resize_cache(common_workspace, fused_out_shape)
```

当前 CPUGroupGemmExperts v2 的 workspace shape 近似为：

```python
workspace13 = (0,)
workspace2 = (0,)
output = (M, K)
```

所以在该配置下，`common_workspace` 实际主要承载 `fused_out`。

0.14 的 `_allocate_buffers()` 分支略有不同：

```python
if num_chunks == 1 and prod(workspace13_shape) >= prod(fused_out_shape):
    workspace13, workspace2 = current_workspace_manager().get_simultaneous(...)
    fused_out = _resize_cache(workspace13, fused_out_shape)
else:
    workspace13, workspace2, fused_out = current_workspace_manager().get_simultaneous(
        (workspace13_shape, workspace_dtype),
        (workspace2_shape, workspace_dtype),
        (fused_out_shape, out_dtype),
    )
```

但这不是 0.14/0.19 行为分叉的根因。真正关键是：compile 捕获时，这块 workspace 是图内新分配，还是已经存在的 Python 全局 tensor。

## 5. WorkspaceManager 本身不是主因

0.14 和 0.19 的 `WorkspaceManager.get_simultaneous()` 核心行为相近，都是从
`_current_workspaces[ubatch_id]` 中切片：

```python
current_workspace[offset:offset + actual_bytes].view(dtype).reshape(shape)
```

如果 `_current_workspaces[ubatch_id]` 为空，则会分配一块新的 workspace；如果已经存在，则复用已有 tensor。

所以问题不在于 workspace manager “会不会复用”，而在于：

- compile 第一次看到它时，是 `torch.empty` 产生的新 tensor
- 还是 profile 阶段已经创建好的 persistent tensor

这两种情况会让 Dynamo/Inductor 生成完全不同的 C++。

## 6. 0.19 根因链路

0.19 xcpu plugin worker 继承了 GPUWorker 风格的 memory profiling 路径。`worker_v1.py` 中初始化 workspace：

```python
num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
init_workspace_manager(self.device, num_ubatches)
```

`gpu_worker.py` 的 memory profiling 会先跑：

```python
self.model_runner.profile_run(skip_compile=True)
```

这次 forward 是 compile 前的 eager/profile forward。它会真实执行 MoE，并调用：

```python
current_workspace_manager().get_simultaneous(...)
```

于是 `_current_workspaces[ubatch_id]` 被分配并保存在 workspace manager 中。

之后进入 compile/warmup 时，Dynamo 看到的 workspace 不再是图内 `torch.empty`，而是 Python 全局状态中已有的 tensor。这个 tensor 会被 hoist 成 compiled graph input：

```cpp
auto arg27_1 = std::move(inputs[27]);
```

由于后续 custom op 会 mutate 这个 input 的 view，Inductor/AOT functionalization 需要保护 graph input 语义，于是生成：

```text
input workspace -> private temp copy-in
private temp 上执行 fused_moe_compute / moe_finalize
private temp -> input workspace copy-back
```

这就是巨大 `cpp_fused_view_2` 和末尾 `cpp_fused_copy__3` 的来源。

## 7. 为什么只 copy 一次

0.19 旧生成代码中只出现一次大的 copy-in，是因为这块 workspace 是跨多个 MoE layer 复用的 common workspace graph input。

Inductor 对这个 input 做一次 functionalization：

```text
arg27_1 -> buf77
```

后续所有 `fused_moe_compute_bf16` / `moe_finalize_bf16` 都复用 `buf77`，所以不是每层都重新从 input copy 一次。末尾再统一 copy-back 到 `arg27_1`。

这解释了“为什么只清 0 / copy 了 1 次”的观测：它不是每层的初始化，而是针对一个 mutable graph input 的函数级 copy-in/copy-back。

## 8. 0.14 为什么没有类似问题

0.14 CPU-only worker 的 `determine_available_memory()` 更简单，近似是：

```python
def determine_available_memory(self) -> int:
    return self.cache_config.cpu_kvcache_space_bytes or 0
```

它没有在 compile 前执行一次会分配 MoE workspace 的 `profile_run(skip_compile=True)`。后续 `compile_or_warm_up_model()` 才第一次真正跑到 MoE workspace 分配。

因此 Dynamo 捕获到的是图内 allocation，0.14 AOTI C++ 形态类似：

```cpp
AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
    1, int_array_68, int_array_69,
    cached_torch_dtype_uint8,
    cached_torch_device_type_cpu,
    0,
    &buf69_handle));
RAIIAtenTensorHandle buf69(buf69_handle);

AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_mcpu_view_dtype(
    buf69,
    cached_torch_dtype_bfloat16,
    &buf71_handle));
RAIIAtenTensorHandle buf71(buf71_handle);

aoti_torch_mcpu_fused_moe_compute_bf16(... buf71 ...);
aoti_torch_mcpu_moe_finalize_bf16(... buf71 ...);
```

这里没有从 `inputs[N]` copy 到 `buf71`，因为 workspace 不是 graph input。

## 9. Eager 模式行为

在 eager 模式下，没有 Dynamo graph input functionalization 这层转换。

如果 workspace manager 中已经有 workspace，MoE 会直接拿已有 workspace 的 view 给 custom op 使用：

```python
fused_out = _resize_cache(common_workspace, fused_out_shape)
```

custom op 会直接写这块 workspace/view。因此 eager 模式通常不会出现 AOTI C++ 里的 copy-in/copy-back kernel。

这个差异也是为什么问题只在 compile 生成代码中明显暴露。

## 10. 修复策略

目标是在保留 `determine_available_memory()` 的同时，避免 profiling 创建的 workspace 持久化到 compile 捕获阶段。

实现分两部分。

### 10.1 WorkspaceManager 增加 clear

在 `vllm/vllm/v1/worker/workspace.py` 增加：

```python
def clear(self) -> None:
    """Release currently allocated workspace buffers.

    This keeps the manager configuration intact but drops tensor references
    so a later warmup/compile run observes fresh allocations.
    """
    if self._locked:
        raise AssertionError("Cannot clear workspace while it is locked.")

    if envs.VLLM_DEBUG_WORKSPACE:
        logger.info(
            "[WORKSPACE DEBUG] Clearing workspace buffers. Current sizes: %s",
            [
                self._workspace_size_bytes(ws) / _MB
                for ws in self._current_workspaces
                if ws is not None
            ],
        )

    for ubatch_id in range(len(self._current_workspaces)):
        self._current_workspaces[ubatch_id] = None
```

并增加模块级 helper：

```python
def clear_workspace() -> None:
    """Release allocated workspace tensors without resetting the manager."""
    current_workspace_manager().clear()
```

这里没有 reset workspace manager 的配置，只是丢掉当前已分配 tensor 引用。

### 10.2 xcpu worker 在 profiling 后清 workspace

在 `vllm-xcpu-plugin/vllm_xcpu_plugin/worker/worker_v1.py` 中：

```python
import gc

from vllm.config import CompilationMode, VllmConfig
from vllm.v1.worker.workspace import clear_workspace, init_workspace_manager
```

修改 `determine_available_memory()`：

```python
def determine_available_memory(self) -> int:
    available_memory = super().determine_available_memory()

    if self.vllm_config.compilation_config.mode != CompilationMode.NONE:
        # The profiling run above may allocate persistent vLLM workspaces
        # before torch.compile warmup. If left alive, Dynamo can hoist them
        # as graph inputs, causing Inductor functionalization to generate
        # large copy-in/copy-back kernels for mutable workspace views. Drop
        # the profiling workspaces so compile sees fresh graph-local
        # allocations.
        logger.info(
            "Clearing profiling workspaces before compile warmup. "
            "compilation_mode=%s",
            self.vllm_config.compilation_config.mode.name,
        )
        clear_workspace()
        gc.collect()
        torch.accelerator.empty_cache()

    kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE
    if kv_cache_space is None:
        return available_memory

    kv_cache_space_bytes = kv_cache_space * GiB_bytes
    logger.info(
        "Force reset available kv cache memory from %sGiB to "
        "VLLM_CPU_KVCACHE_SPACE: %sGiB",
        format_gib(available_memory),
        format_gib(kv_cache_space_bytes),
    )
    self.available_kv_cache_memory_bytes = kv_cache_space_bytes
    return kv_cache_space_bytes
```

关键细节：不能只判断 `mode == CompilationMode.VLLM_COMPILE`。

`vllm-xcpu-plugin/vllm_xcpu_plugin/platform.py` 会把用户侧 `VLLM_COMPILE` 改写为：

```python
if vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
    compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE
    compilation_config.backend = "inductor"
```

因此 worker 中使用：

```python
mode != CompilationMode.NONE
```

才覆盖实际 compile 路径。

## 11. Workspace 生命周期说明

用户关心的问题是：如果 workspace 分配落在 compile graph 内，谁负责释放？

结论分情况：

- 如果 workspace 是 compiled graph 内部的 `empty_strided` 临时 tensor，生命周期由 AOTI/PyTorch runtime 和底层 allocator 管理，不由 vLLM `WorkspaceManager` 释放。
- 如果 graph 把 workspace view 作为 output 返回，例如 `output_handles[1] = buf3897.release()`，则 Python wrapper/runtime 持有这个输出引用；外层不再保存它后，引用释放，再由 PyTorch allocator 回收或缓存。
- `torch.accelerator.empty_cache()` 可以尝试把 allocator cache 释放给后端 allocator，但是否立即归还 OS 取决于 mcpu/privateuse1 后端 allocator 实现。
- 修复中的 `clear_workspace()` 只释放 profiling 阶段创建并挂在 `WorkspaceManager` 上的 Python 引用，不负责释放 compiled graph 内部临时 tensor。

这也是修复策略合理的原因：compile 阶段的 workspace 不再是 vLLM workspace manager 的持久 tensor，而是 compiled graph runtime 管理的对象。

## 12. 端到端验证

语法检查：

```bash
python -m py_compile \
  vllm/vllm/v1/worker/workspace.py \
  vllm-xcpu-plugin/vllm_xcpu_plugin/worker/worker_v1.py
```

执行前移动旧 cache，避免复用：

```bash
mv vllm_scripts/torch_compile_cache_opt vllm_scripts/torch_compile_cache_opt.bk
```

运行 bench：

```bash
cd vllm_scripts
VLLM_TEST_MAX_WAIT=700 ./run_vllm_test.sh \
  -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_compile_alltoallv_v2.sh \
  --bench
```

bench 成功完成。新生成的 optimized cache 包含：

```text
vllm_scripts/torch_compile_cache_opt/ke/ckeqedhil4t7hlh4dbbyrx42gd5hcp6bg6fvrcwfpddq7uubg5q3.main.cpp
vllm_scripts/torch_compile_cache_opt/mc/cmcu4ymrtrli4rovqfhhawgb2ahgvxbnqzfdusft66d4jcmwslca.main.cpp
```

检查 copy kernel：

```bash
rg -n "cpp_fused_view_2|cpp_fused_copy__3" \
  vllm_scripts/torch_compile_cache_opt \
  -g '*.main.cpp'
```

结果无匹配，说明原先的 copy-in/copy-back kernel 已消失。

新的 C++ 形态变成图内分配 workspace：

```cpp
AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
    1,
    int_array_64,
    int_array_65,
    cached_torch_dtype_uint8,
    cached_torch_device_type_privateuse1,
    0,
    &buf69_handle));
RAIIAtenTensorHandle buf69(buf69_handle);

AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_mcpu_view_dtype(
    buf69,
    cached_torch_dtype_bfloat16,
    &buf71_handle));
RAIIAtenTensorHandle buf71(buf71_handle);
```

MoE compute/finalize 直接使用同一个图内 `buf71`：

```cpp
AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_mcpu_fused_moe_compute_bf16(
    wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(
        buf71, 2, int_array_82, int_array_83, 0L)),
    ...));

AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_mcpu_moe_finalize_bf16(
    buf89,
    wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(
        buf71, 2, int_array_90, int_array_91, 0L)),
    ...));
```

结尾仍会把 `buf71` 的 uint8 view 作为 graph output 返回：

```cpp
AtenTensorHandle tmp_buf71_144;
AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_tensor_handle(
    buf71, &tmp_buf71_144));

AtenTensorHandle buf3897_handle;
AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_mcpu_view_dtype(
    RAIIAtenTensorHandle(tmp_buf71_144),
    cached_torch_dtype_uint8,
    &buf3897_handle));
RAIIAtenTensorHandle buf3897(buf3897_handle);

output_handles[0] = buf3898.release();
output_handles[1] = buf3897.release();
```

这与 0.14 的关键行为一致：workspace 不再是 graph input，所以不再触发针对 mutable input 的 functionalization copy。

## 13. 修复后的行为边界

这次修复针对的是 compile 前 profiling 创建的 workspace 引用，不改变 hot path 中 workspace manager 的复用语义。

修复后：

- `determine_available_memory()` 仍然保留。
- profiling 阶段可以正常分配 workspace。
- profiling 结束后，workspace manager 丢掉 profiling workspace 引用。
- compile 第一次进入 MoE 时重新分配 workspace，并被捕获为图内 allocation。
- generated C++ 不再从 graph input workspace copy-in，也不再 copy-back。

需要持续关注的边界：

- 如果未来某条路径在 compile 前再次跑到 MoE 并重新填充 workspace manager，也可能再次把 workspace hoist 成 graph input。
- 如果外层 wrapper 长期持有 graph output workspace，内存释放时机会跟随该引用生命周期，而不是 workspace manager。
- 如果后续为更复杂算子配置强依赖 persistent workspace 作为跨调用状态，需要单独区分“必须跨调用持久化”的 workspace 和“只为单次 forward 复用”的 scratch workspace。

## 14. 最终结论

0.19 的巨大 `cpp_fused_view_2` 不是 prepare 的 `recv_hidden_states` 被清零，也不是 compute/finalize 之间不必要地复制同一个 tensor。

它本质来自：

```text
determine_available_memory()
-> profile_run(skip_compile=True)
-> MoE eager/profile forward 分配 persistent workspace
-> compile 时 workspace 已存在
-> Dynamo 把 workspace hoist 成 graph input
-> mutable custom op 触发 functionalization
-> 生成 workspace copy-in/copy-back
```

通过在 profiling 完成后清掉 workspace manager 中的 profiling workspace，compile 重新捕获图内 allocation，额外 copy kernel 消失。端到端 bench 已验证该修复在当前
`Qwen3-30B-A3B_dp2_tp2_ep_compile_alltoallv_v2` 配置下生效。
