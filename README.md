# vLLM XCPU Plugin

vLLM 自定义操作插件示例，展示如何通过 plugin 机制扩展 vLLM 算子。

## 项目结构

```
vllm_xcpu_plugin/
├── __init__.py       # 插件入口，注册自定义操作
└── custom_ops.py     # 自定义算子实现 (XcpuRMSNorm)
```

## 工作原理

vLLM 通过 setuptools entry_points 发现插件：

```python
# setup.py
entry_points={
    "vllm.general_plugins": [
        "xcpu_custom_ops = vllm_xcpu_plugin:register_ops"
    ]
}
```

自定义算子通过 `CustomOp` 基类和 `register_oot` 装饰器注册：

```python
@RMSNorm.register_oot
class XcpuRMSNorm(RMSNorm):
    def forward_cpu(self, x, residual=None):
        # CPU 上的 RMSNorm 实现
```

## 开发

### 安装

```bash
pip install --no-build-isolation .
```

或者，如果您希望以可编辑模式安装（方便开发）：

```bash
pip install --no-build-isolation -e .
```

如果遇到问题，请尝试 pip uninstall vllm_xcpu_plugin，或更进一步地删除 site-packages 下的相关目录。

### 代码风格

项目使用以下工具进行代码检查和格式化：

| 脚本 | 用途 |
|------|------|
| `./scripts/format.sh` | 自动格式化代码 (ruff format + fix) |
| `./scripts/lint.sh` | 检查代码风格 (ruff check only) |
| `./scripts/mypy.py` | 类型检查 (mypy) |
| `./scripts/check.sh` | 运行所有检查 |

提交代码前请运行：

```bash
# 自动检查（不修改代码）
./scripts/check.sh

# 如果 lint 失败，可以运行自动格式化
./scripts/format.sh

# 如果 lint 仍然失败 / mypy 失败，请依据提示修复
```

### Git 钩子

项目配置了 Git 钩子，在 commit/push 前自动运行代码检查：

- **pre-commit**: 检查暂存的 Python 文件
- **pre-push**: 检查所有 Python 文件

安装钩子：

```bash
./scripts/install-hooks.sh
```

如果 lint 失败，钩子会提示运行 `./scripts/format.sh` 自动修复。

临时跳过钩子（不推荐）：

```bash
git commit --no-verify
git push --no-verify
```

### 添加新算子

1. 在 `custom_ops.py` 中创建继承自对应基类的自定义操作
2. 使用 `@{BaseClass}.register_oot` 装饰器注册
3. 在 `__init__.py` 的 `register_ops()` 中导入模块
4. 重新安装：`pip install -e .`

## 测试

### 运行测试

```bash
# 完整测试
python3 -m pytest -s -v -k "test_rms_norm and not quant" tests/kernels/core/test_layernorm.py

# 首个失败时退出
python3 -m pytest -x -s -v -k "test_rms_norm and not quant" tests/kernels/core/test_layernorm.py
```

### 测试文件修改

测试文件需加载插件并修改测试参数：

```python
# tests/kernels/core/test_layernorm.py
from vllm.platforms import current_platform

from vllm.plugins import load_general_plugins
load_general_plugins()

CUDA_DEVICES = ["cpu"]           # 修改为 CPU
HIDDEN_SIZES = [8, 192]         # 临时调试时减少用例
@pytest.mark.parametrize("strided_input", [False])  # 当前仅支持 contiguous
```
