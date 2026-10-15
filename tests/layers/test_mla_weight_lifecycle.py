# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA must release source storage without invalidating projections."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch_mcpu  # noqa: F401
import torch_xcpu
from torch.multiprocessing.reductions import StorageWeakRef

from vllm_xcpu_plugin.attention.mla import XcpuMLAAttention
from vllm_xcpu_plugin.layers.fp8_linear import XcpuFp8BlockScaledMMLinearKernel
from vllm_xcpu_plugin.layers.quark_mxfp4 import XcpuQuarkOCPMXLinearScheme


def make_layer(weight_format, device, heads, key_dim=4, value_dim=8, rank=32):
    layer = XcpuMLAAttention.__new__(XcpuMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.num_heads = heads
    layer.qk_nope_head_dim = key_dim
    layer.v_head_dim = value_dim
    layer.kv_lora_rank = rank
    proj = layer.kv_b_proj = torch.nn.Module()
    proj.params_dtype = torch.bfloat16
    proj._xcpu_mla_weight_only = True
    layer.impl = SimpleNamespace(
        is_sparse=True,
        process_weights_after_loading=lambda dtype: None,
        kv_b_proj=proj,
    )
    rows = heads * (key_dim + value_dim)
    if weight_format == "mxfp4":
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kMxfp4Static,
        )

        proj.weight = torch.nn.Parameter(
            torch.full((rows, rank // 2), 0x21, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        proj.weight_scale = torch.nn.Parameter(
            torch.full((rows, rank // 32), 128, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        expected = torch.tensor([1, 2], dtype=torch.bfloat16).repeat(rows, rank // 2)
        processor = XcpuQuarkOCPMXLinearScheme(kMxfp4Static, None)
    else:
        expected = (torch.arange(rows * rank).reshape(rows, rank) % 16).bfloat16()
        proj.weight = torch.nn.Parameter(
            expected.to(
                device=device,
                dtype=torch.bfloat16
                if weight_format == "bf16"
                else torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        processor = None
        if weight_format != "bf16":
            proj.register_parameter(
                weight_format,
                torch.nn.Parameter(
                    torch.full((rows // 2, rank // 16), 0.5, device=device),
                    requires_grad=False,
                ),
            )
            expected = expected * 0.5
            processor = object.__new__(XcpuFp8BlockScaledMMLinearKernel)
            processor.weight_group_shape = (2, 16)
    # A CPU BF16 parameter can share expected's storage; keep the oracle separate.
    expected = expected.clone().view(heads, key_dim + value_dim, rank)
    sources = [StorageWeakRef(p.untyped_storage()) for p in proj.parameters()]
    if processor is not None:
        processor.process_weights_after_loading(proj)
    sources.append(StorageWeakRef(proj.weight.untyped_storage()))
    return layer, expected, sources


@pytest.mark.parametrize("device", ["cpu", "mcpu"])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize(
    "weight_format", ["bf16", "weight_scale_inv", "weight_scale", "mxfp4"]
)
def test_postload_releases_source_storage(weight_format, heads, device):
    layer, expected, sources = make_layer(weight_format, device, heads)
    layer.process_weights_after_loading(torch.bfloat16)

    torch.testing.assert_close(layer.W_UK_T.cpu(), expected[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(
        layer.W_UV.cpu(), expected[:, 4:].transpose(1, 2), rtol=0, atol=0
    )
    assert all(source.expired() for source in sources)
    assert layer.impl.kv_b_proj is layer.kv_b_proj
    assert all(p.numel() == 0 for p in layer.kv_b_proj.parameters())
    for weight in (layer.W_UK_T, layer.W_UV):
        assert weight.is_contiguous()
        assert not weight.requires_grad
        assert (
            weight.untyped_storage().nbytes() == weight.numel() * weight.element_size()
        )


def test_failed_projection_copy_keeps_source(monkeypatch):
    layer, _, _ = make_layer("weight_scale_inv", "cpu", 2)
    weight, scale = layer.kv_b_proj.weight, layer.kv_b_proj.weight_scale_inv
    clone = torch.Tensor.clone
    copies = 0

    def fail_second_copy(self, *args, **kwargs):
        nonlocal copies
        copies += 1
        if copies == 2:
            raise RuntimeError("projection allocation failed")
        return clone(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", fail_second_copy)
    with pytest.raises(RuntimeError, match="projection allocation failed"):
        layer.process_weights_after_loading(torch.bfloat16)
    assert layer.kv_b_proj.weight is weight
    assert layer.kv_b_proj.weight_scale_inv is scale
    assert not hasattr(layer, "W_UK_T")


def test_platform_routes_only_sparse_mla():
    from vllm_xcpu_plugin.platform import McpuPlatform

    def select(use_sparse, selected=None):
        config = SimpleNamespace(use_mla=True, use_sparse=use_sparse)
        return McpuPlatform.get_attn_backend_cls(selected, config)

    assert select(True).endswith("attention.sparse_mla.XcpuMLASparseBackend")
    with pytest.raises(NotImplementedError, match="sparse MLA"):
        select(False)
    with pytest.raises(ValueError, match="selected backend"):
        select(True, "TRITON_MLA")


@pytest.mark.parametrize("weight_format", ["weight_scale_inv", "mxfp4"])
def test_released_weights_support_eager_and_inductor(
    monkeypatch, tmp_path, weight_format
):
    package = Path(torch_xcpu.__file__).parent
    library = next(package.glob("_C*.so"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv(
        "AOTI_EXTRA_CFLAGS", f"-include {package / 'include/aoti_torch_xcpu.h'}"
    )
    monkeypatch.setenv("AOTI_EXTRA_LDFLAGS", f"-Wl,-rpath,{package} {library}")
    monkeypatch.setenv("TORCHINDUCTOR_DIRECT_DISPATCH_PREFIXES", "torch_xcpu")
    torch_xcpu.initialize_runtime()
    layer, _, sources = make_layer(weight_format, "mcpu", 8, 192, 256, 512)
    layer.process_weights_after_loading(torch.bfloat16)
    assert all(source.expired() for source in sources)

    def project(q, latent):
        key = q.new_empty((q.shape[0], 8, 512))
        value = latent.new_empty((latent.shape[0], 8, 256))
        torch_xcpu.ops.einsum_mhk_hkn_to_mhn(q, layer.W_UK_T, key)
        torch_xcpu.ops.einsum_mhk_hkn_to_mhn(latent, layer.W_UV, value)
        return key, value

    torch._dynamo.reset()
    with torch.inference_mode(), torch._inductor.config.patch(cpp_wrapper=True):
        compiled = torch.compile(project, fullgraph=True, dynamic=True)
        for tokens in (2, 5, 1):
            q = torch.randn(tokens, 8, 192, dtype=torch.bfloat16).to("mcpu")
            latent = torch.randn(tokens, 8, 512, dtype=torch.bfloat16).to("mcpu")
            expected = project(q, latent)
            actual = compiled(q, latent)
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(
                    result.cpu(), reference.cpu(), rtol=0, atol=0
                )
    torch._dynamo.reset()
