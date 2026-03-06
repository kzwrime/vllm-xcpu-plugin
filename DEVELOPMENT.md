# vLLM XCPU Plugin - 开发经验总结

## 仓库架构

```
vllm-xcpu-plugin/          # 主仓库，vLLM 的 XCPU 插件
├── vllm_xcpu_plugin/      # 插件代码
├── torch_xcpu -> ../torch_ext/torch_xcpu  # 符号链接
├── torch_mpi_ext -> ../torch_ext/torch_mpi_ext  # 符号链接

torch_ext/torch_xcpu/      # 算子库（独立仓库）
├── torch_xcpu/            # Python 包
│   ├── csrc/              # PyTorch 绑定
│   ├── include/           # 头文件（内联实现）
│   └── ops.py             # Python API
├── torch_xcpu_impl/       # C++ 实现
│   ├── src/               # 源文件
│   ├── include/           # 头文件
│   └── Makefile           # 构建脚本
└── tests/                 # 算子测试
    └── test_vocab_parallel_embedding.py

torch_ext/torch_mpi_ext/  # MPI 通信库
```

## 编译流程

### torch_xcpu (算子库)
```bash
cd torch_xcpu && ./build-all.sh
# 1. 编译 torch_xcpu_impl (C++ 库)
# 2. 编译 torch_xcpu (Python 扩展)
# 3. 安装到 Python 环境
```

**关键文件**:
- `torch_xcpu_impl/Makefile`: 定义 include 路径（需包含 `torch_xcpu/include`）
- `torch_xcpu/setup.py`: 自动发现 csrc/*.cpp

### vllm-xcpu-plugin (主仓库)
```bash
pytest tests/               # 运行测试
./scripts/format.sh <file>  # 格式化代码
```

## 开发新算子流程

### 1. 添加 C++ 算子

**简单算子**（内联实现）:
```
torch_xcpu/include/my_op.hpp     # 内联实现
torch_xcpu/csrc/my_op.cpp        # PyTorch 绑定
```

**复杂算子**（分离实现）:
```
torch_xcpu_impl/src/my_op.cpp   # C++ 实现
torch_xcpu/include/my_op.hpp    # 声明
torch_xcpu/csrc/my_op.cpp       # PyTorch 绑定
```

### 2. 注册算子

**torch_xcpu/csrc/torch_bindings.cpp**:
```cpp
// 定义 schema
m.def("my_op(Tensor! out, Tensor in) -> ()");

// 注册实现
m.impl("my_op", TORCH_FN(my_op_cpu));
```

### 3. Python 绑定

**torch_xcpu/ops.py**:
```python
def my_op_check(out, in): ...
def my_op(out, in):
    my_op_check(out, in)
    torch.ops.torch_xcpu.my_op(out, in)

@torch.library.register_fake("torch_xcpu::my_op")
def _(out, in):
    my_op_check(out, in)
```

### 4. 覆盖 vLLM 算子

**vllm_xcpu_plugin/custom_ops.py**:
```python
from vllm.model_executor.layers.SomeLayer import SomeLayer

@SomeLayer.register_oot
class XcpuSomeLayer(SomeLayer):
    def __init__(self, ...):
        super().__init__(...)
        if current_platform.is_cpu():
            self._forward_method = self.forward_cpu

    def forward_cpu(self, ...):
        import torch_xcpu.ops as ops
        ops.my_op(...)
```

## 提交流程

1. **torch_xcpu 先提交**: 算子实现 → 测试 → 提交
2. **vllm-xcpu-plugin 后提交**: 覆盖实现 → 测试 → 提交
3. **Commit 格式**:
   - torch_xcpu: `[Feat]`, `[Refactor]`, `[MoE]`
   - vllm-xcpu-plugin: 同样格式
   - 包含 `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`

## 测试

**torch_xcpu 测试**:
```bash
cd torch_xcpu
pytest tests/test_vocab_parallel_embedding.py
```

**vllm-xcpu-plugin 测试**:
```bash
pytest tests/kernels/core/test_activation.py
```

## 注意事项

1. **类型分发**: 使用 `AT_DISPATCH_INTEGRAL_TYPES` 在 C++ 中统一处理 int32/int64
2. **内联实现**: 简单算子直接在 .hpp 中实现，无需分离 impl
3. **Makefile**: 修改 torch_xcpu_impl/Makefile 时确保 include 路径正确
4. **工作目录**: 注意 bash 工作目录，避免在错误的仓库执行 git 操作
5. **格式化**: 提交前运行 `./scripts/format.sh` 和 `ruff check`
