from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_xcpu_plugin.layers.fp8_linear import (
    XcpuFp8BlockScaledMMLinearKernel,
    register_fp8_linear_kernel,
)


def test_mla_weight_stays_canonical_with_block_scales():
    """MLA splits [out, in] weights instead of consuming a packed linear."""
    layer = torch.nn.Module()
    layer._xcpu_mla_weight_only = True
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(
        torch.ones((3, 5), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0], [4.0, 8.0]]), requires_grad=False
    )
    kernel = object.__new__(XcpuFp8BlockScaledMMLinearKernel)
    kernel.weight_group_shape = (2, 3)

    kernel.process_weights_after_loading(layer)

    expected = torch.tensor(
        [[1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [4, 4, 4, 8, 8]],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)
    assert not hasattr(layer, "_xcpu_fp8_linear")


@pytest.mark.parametrize("scale_attr", ["weight_scale_inv", "weight_scale"])
@pytest.mark.parametrize(
    "row_parallel,tp_rank,skip_bias_add,fuse_bias",
    [
        (False, 0, False, True),
        (False, 1, False, True),
        (False, 0, True, False),
        (True, 0, False, True),
        (True, 1, False, False),
        (True, 0, True, False),
    ],
)
def test_process_weights_uses_initialized_torch_xcpu_operator(
    monkeypatch, scale_attr, row_parallel, tp_rank, skip_bias_add, fuse_bias
):
    import torch_xcpu

    class Layer(RowParallelLinear if row_parallel else torch.nn.Module):  # type: ignore[misc]
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.tp_rank = tp_rank
            self.skip_bias_add = skip_bias_add
            self.register_parameter(
                "weight",
                torch.nn.Parameter(
                    torch.zeros((128, 128), dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                scale_attr,
                torch.nn.Parameter(
                    torch.ones((1, 1), dtype=torch.float32),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.ones(128, dtype=torch.bfloat16),
                    requires_grad=False,
                ),
            )

    calls = []

    class FakeFp8Linear:
        def __init__(self, weight, weight_scale, bias):
            self.params = SimpleNamespace(
                packed_weight=weight.clone(),
                weight_scale=weight_scale,
                bias=bias,
                backend=SimpleNamespace(name="INTEL_AMX"),
            )

        def __call__(self, x):
            return x

    def initialize(weight, weight_scale, block_size, bias):
        calls.append((weight, weight_scale, block_size, bias))
        return FakeFp8Linear(weight, weight_scale, bias)

    monkeypatch.setattr(torch_xcpu.ops, "initialize_fp8_linear", initialize)

    kernel = object.__new__(XcpuFp8BlockScaledMMLinearKernel)
    kernel.weight_group_shape = (128, 128)
    layer = Layer()
    kernel.process_weights_after_loading(layer)

    assert len(calls) == 1
    assert calls[0][2] == (128, 128)
    if fuse_bias:
        assert calls[0][3].dtype == torch.float32
        assert layer.bias.dtype == torch.float32
        assert calls[0][3].data_ptr() == layer.bias.data_ptr()
    else:
        assert calls[0][3] is None
        assert layer.bias.dtype == torch.bfloat16
    assert (
        layer.weight.data_ptr()
        == layer._xcpu_fp8_linear.params.packed_weight.data_ptr()
    )
    assert (
        getattr(layer, scale_attr).data_ptr()
        == layer._xcpu_fp8_linear.params.weight_scale.data_ptr()
    )

    x = torch.ones((2, 128), dtype=torch.bfloat16)
    assert kernel.apply_weights(layer, x, layer.bias if fuse_bias else None) is x


@pytest.mark.parametrize(
    "input_dtype,out_dtype",
    [
        (torch.float16, torch.bfloat16),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_kernel_selection_requires_bf16_activations(
    monkeypatch, input_dtype, out_dtype
):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels import linear
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        create_fp8_quant_key,
    )

    # Exercise vLLM's actual selector and constructor; a mocked preprocessing
    # call alone cannot catch an upstream config or kernel-family change.
    monkeypatch.setattr(linear, "_POSSIBLE_FP8_BLOCK_KERNELS", {})
    register_fp8_linear_kernel()
    with set_current_vllm_config(VllmConfig()):
        kwargs = dict(
            activation_quant_key=create_fp8_quant_key(
                static=False, group_shape=GroupShape(1, 128)
            ),
            weight_quant_key=create_fp8_quant_key(
                static=True, group_shape=GroupShape(128, 128)
            ),
            weight_shape=(128, 128),
            input_dtype=input_dtype,
            out_dtype=out_dtype,
        )
        if input_dtype == out_dtype == torch.bfloat16:
            kernel = init_fp8_linear_kernel(**kwargs)
            assert isinstance(kernel, XcpuFp8BlockScaledMMLinearKernel)
            assert kernel.input_quant_key() is None
        else:
            with pytest.raises(ValueError, match="requires BF16 input and output"):
                init_fp8_linear_kernel(**kwargs)
