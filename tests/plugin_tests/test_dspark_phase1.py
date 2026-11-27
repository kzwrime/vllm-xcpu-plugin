# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch_mcpu  # noqa: F401
import torch_xcpu
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.utils import dispatch_xcpu_bf16_linear

from vllm_xcpu_plugin.layers.fp8_linear import XcpuFp8BlockScaledMMLinearKernel
from vllm_xcpu_plugin.layers.mxfp8_linear import XcpuMxfp8LinearKernel
from vllm_xcpu_plugin.models.deepseek_v41 import model as dsv41_model_mod
from vllm_xcpu_plugin.models.deepseek_v41.model import (
    XcpuDSparkDeepseekV41ForCausalLM,
    XcpuDSparkDeepseekV41Model,
)
from vllm_xcpu_plugin.models.dspark.speculator import XcpuDSparkSpeculator
from vllm_xcpu_plugin.rms_norm_dispatch import register_rms_norm_kernel


@contextmanager
def _dummy_run_scope(enabled: bool):
    prev = torch_xcpu.ops.is_dummy_run()
    torch_xcpu.ops.set_dummy_run(enabled)
    try:
        yield
    finally:
        torch_xcpu.ops.set_dummy_run(prev)


def _make_dummy_mxfp8_proj(
    *, q_lora_rank: int, head_dim: int, hidden_size: int, seed: int
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = q_lora_rank + head_dim
    w_fp32 = torch.randn((n, hidden_size), generator=gen) * 0.25
    w_fp8 = w_fp32.to(torch.float8_e4m3fn).to("mcpu")
    exp_blocks = torch.randint(
        125, 130, (n // 32, hidden_size // 32), generator=gen, dtype=torch.uint8
    )
    row_repeated_u8 = exp_blocks.repeat_interleave(32, dim=0).contiguous()
    layer = torch.nn.Module()
    layer.bias = None
    layer.register_parameter("weight", torch.nn.Parameter(w_fp8, requires_grad=False))
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(
            row_repeated_u8.view(torch.float8_e8m0fnu).to("mcpu"),
            requires_grad=False,
        ),
    )
    layer._xcpu_keep_raw_wkv_offset = q_lora_rank
    layer._xcpu_keep_raw_weight = True
    return layer, w_fp8.detach().clone(), row_repeated_u8.clone()


def _attach_fake_rmsnorm(weight: torch.Tensor, eps: float = 1e-6):
    register_rms_norm_kernel()

    def norm(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        torch.ops._C.rms_norm(out, x, weight, eps)
        return out

    return norm


def _build_fake_dspark_v41(
    *,
    num_layers: int = 3,
    q_lora_rank: int = 1280,
    head_dim: int = 512,
    hidden_size: int = 5120,
    projs: list[torch.nn.Module],
) -> tuple[XcpuDSparkDeepseekV41ForCausalLM, XcpuDSparkDeepseekV41Model]:
    inner = object.__new__(XcpuDSparkDeepseekV41Model)
    torch.nn.Module.__init__(inner)
    inner.config = SimpleNamespace(head_dim=head_dim, hidden_size=hidden_size)
    inner.hidden_size = hidden_size
    inner.num_dspark_layers = num_layers
    inner._context_wkv_linear = None

    layers = torch.nn.ModuleList()
    for i, proj in enumerate(projs):
        norm_w = (
            torch
            .linspace(0.8, 1.2, head_dim, dtype=torch.bfloat16)
            .to("mcpu")
            .contiguous()
        )
        attn = torch.nn.Module()
        attn.q_lora_rank = q_lora_rank
        attn.head_dim = head_dim
        attn.fused_wqa_wkv = proj
        attn.kv_norm = _attach_fake_rmsnorm(norm_w)
        layer = torch.nn.Module()
        layer.attn = attn
        layers.append(layer)
    inner.layers = layers

    outer = object.__new__(XcpuDSparkDeepseekV41ForCausalLM)
    torch.nn.Module.__init__(outer)
    outer.model = inner
    return outer, inner


def test_xcpu_dspark_speculator_greedy_argmax_dispatch(monkeypatch):
    spec = object.__new__(XcpuDSparkSpeculator)
    spec.draft_logits = None
    spec.acceptance_estimator = None
    spec.draft_token_confidence_probs = None
    spec.temperature = None
    spec._step_cols = [torch.tensor([0], dtype=torch.int32, device="mcpu")]
    spec.model = SimpleNamespace(map_draft_to_target=lambda ids: ids + 100)

    calls = []
    orig_argmax_out = torch_xcpu.ops.argmax_out

    def spy_argmax_out(self, dim, keepdim, out):
        calls.append((tuple(self.shape), self.dtype))
        return orig_argmax_out(self, dim, keepdim, out)

    monkeypatch.setattr(torch_xcpu.ops, "argmax_out", spy_argmax_out)

    idx_map = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
    pos = torch.tensor([10, 11], dtype=torch.int64, device="mcpu")

    # 1. Contiguous 2D BF16 even vocab on mcpu -> uses torch_xcpu.ops.argmax_out
    torch.manual_seed(0)
    logits = torch.randn((2, 128), dtype=torch.bfloat16).to("mcpu")
    got = spec._sample_logits(logits, idx_map, pos, 0)
    expected = logits.cpu().float().argmax(dim=-1) + 100
    assert calls == [((2, 128), torch.bfloat16)]
    assert torch.equal(got.cpu(), expected)

    # 2. Non-contiguous 2D BF16 slice -> falls back to logits.argmax(dim=-1)
    calls.clear()
    buf = torch.randn((2, 3, 128), dtype=torch.bfloat16).to("mcpu")
    non_contig = buf[:, 0]
    assert not non_contig.is_contiguous()
    got_nc = spec._sample_logits(non_contig, idx_map, pos, 0)
    assert calls == []
    assert torch.equal(got_nc.cpu(), non_contig.cpu().float().argmax(dim=-1) + 100)

    # 3. Contiguous FP32 -> falls back cleanly
    calls.clear()
    fp32_logits = logits.float()
    got_fp32 = spec._sample_logits(fp32_logits, idx_map, pos, 0)
    assert calls == []
    assert torch.equal(got_fp32.cpu(), fp32_logits.cpu().argmax(dim=-1) + 100)


def test_xcpu_dspark_speculator_probabilistic_uses_patched_gumbel(monkeypatch):
    import vllm.v1.worker.gpu.sample.gumbel as gumbel_module

    spec = object.__new__(XcpuDSparkSpeculator)
    spec.draft_logits = torch.empty((4, 2, 16), dtype=torch.bfloat16, device="mcpu")
    spec.acceptance_estimator = None
    spec.draft_token_confidence_probs = None
    spec.draft_watermarker = None
    spec._d2t_scatter_index = None
    spec._draft_scatter_buf = None
    spec.use_fp64_gumbel = False
    spec.temperature = torch.ones(4, dtype=torch.float32, device="mcpu")
    spec.seeds = torch.zeros(4, dtype=torch.int64, device="mcpu")
    spec._step_cols = [torch.tensor([1], dtype=torch.int32, device="mcpu")]

    calls = []
    sentinel = torch.tensor([5, 6], dtype=torch.int64, device="mcpu")

    def fake_gumbel(
        logits,
        idx_map,
        temperature,
        seeds,
        pos,
        *,
        apply_temperature,
        is_drafting,
        logits_cache,
        logits_cache_col,
        use_fp64,
    ):
        calls.append((pos.cpu().tolist(), apply_temperature, is_drafting, use_fp64))
        return sentinel

    monkeypatch.setattr(gumbel_module, "gumbel_sample", fake_gumbel)

    logits = torch.zeros((2, 16), dtype=torch.bfloat16, device="mcpu")
    idx_map = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
    sample_pos = torch.tensor([10, 20], dtype=torch.int64, device="mcpu")

    got = spec._sample_logits(logits, idx_map, sample_pos, 0)
    assert got is sentinel
    assert calls == [([9, 19], True, True, False)]


def test_xcpu_dspark_speculator_idle_dummy_skips_sampling():
    spec = object.__new__(XcpuDSparkSpeculator)
    run_calls = []
    sample_calls = []

    def fake_run_model(
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode,
    ):
        run_calls.append(cudagraph_runtime_mode)
        return torch.zeros((num_tokens_padded, 16), dtype=torch.bfloat16, device="mcpu")

    def fake_sample_sequential(num_reqs, head_hidden):
        sample_calls.append((num_reqs, tuple(head_hidden.shape)))

    spec._run_model = fake_run_model
    spec._sample_sequential = fake_sample_sequential

    with _dummy_run_scope(True):
        spec._generate_draft(1, 5, None, None, None, CUDAGraphMode.NONE)
        assert run_calls == [CUDAGraphMode.NONE]
        assert sample_calls == []

        # Graph capture / non-NONE modes keep _sample_sequential inside the graph.
        spec._generate_draft(1, 5, None, None, None, CUDAGraphMode.FULL)
        assert run_calls == [CUDAGraphMode.NONE, CUDAGraphMode.FULL]
        assert sample_calls == [(1, (5, 16))]

    with _dummy_run_scope(False):
        spec._generate_draft(1, 5, None, None, None, CUDAGraphMode.NONE)
        assert sample_calls == [(1, (5, 16)), (1, (5, 16))]


def test_mxfp8_context_wkv_stacked_matches_unstacked_and_frees_raw(monkeypatch):
    projs_and_refs = [
        _make_dummy_mxfp8_proj(
            q_lora_rank=1280, head_dim=512, hidden_size=5120, seed=10 + i
        )
        for i in range(3)
    ]
    projs = [p for p, _, _ in projs_and_refs]
    outer, inner = _build_fake_dspark_v41(projs=projs)
    outer._finalize_moe = lambda: None

    # Simulate the two process_weights_after_loading calls in vLLM weight loading:
    # 1. Pre-quant call inside DSparkDeepseekV4ForCausalLM.load_weights()
    outer._loading_checkpoint_weights = True
    outer.process_weights_after_loading()
    outer._loading_checkpoint_weights = False
    assert inner._context_wkv_linear is None

    # 2. Submodule MXFP8 weight packing
    kernel = object.__new__(XcpuMxfp8LinearKernel)
    individual_linears = []
    for proj in projs:
        kernel.process_weights_after_loading(proj)
        assert getattr(proj, "_xcpu_raw_wkv_fp8", None) is not None
        individual_linears.append(proj.xcpu_fp8_linear)

    # 3. Post-quant process_weights_after_loading called by model_loader
    outer.process_weights_after_loading()
    assert inner._context_wkv_linear is not None
    assert inner._context_wkv_linear.params.out_features == 1536
    assert inner._context_wkv_linear.params.in_features == 5120
    for proj in projs:
        assert proj._xcpu_raw_wkv_fp8 is None
        assert proj._xcpu_raw_weight is None

    # Wire individual forward calls to forbid calling fused_wqa_wkv on stacked path.
    for proj in projs:

        def _forbid(*_args, **_kwargs):
            raise AssertionError("fused_wqa_wkv should not run when stacked")

        proj.forward = _forbid

    recorded_stacked: list[torch.Tensor] = []

    def _capture_insert(attn, kv, positions, slot_mapping):
        recorded_stacked.append(kv.detach().cpu().clone())

    monkeypatch.setattr(dsv41_model_mod, "_insert_context_kv", _capture_insert)

    for num_tokens in (1, 7, 17):
        recorded_stacked.clear()
        torch.manual_seed(100 + num_tokens)
        main_x = (torch.randn((num_tokens, 5120), dtype=torch.bfloat16) * 0.5).to(
            "mcpu"
        )
        positions = torch.arange(num_tokens, dtype=torch.int64, device="mcpu")
        slots = [positions.clone() for _ in range(3)]

        with _dummy_run_scope(False):
            inner.precompute_and_store_context_kv(main_x, positions, slots)

        assert len(recorded_stacked) == 3
        for layer, linear, got in zip(
            inner.layers, individual_linears, recorded_stacked, strict=True
        ):
            ref_qr_kv = linear(main_x)
            ref_kv = layer.attn.kv_norm(ref_qr_kv[..., layer.attn.q_lora_rank :])
            assert torch.equal(got, ref_kv.cpu())


def test_fp8_block_and_bf16_context_wkv_stacking(monkeypatch):
    # 1. FP8 block (128x128) pre-pack stash + stacked initialization
    fp8_projs = []
    for i in range(3):
        layer = torch.nn.Module()
        layer.bias = None
        w = torch.ones((128 + 128, 256), dtype=torch.float8_e4m3fn).to("mcpu")
        s = torch.full((2, 2), float(i + 1), dtype=torch.float32, device="mcpu")
        layer.register_parameter("weight", torch.nn.Parameter(w, requires_grad=False))
        layer.register_parameter(
            "weight_scale_inv", torch.nn.Parameter(s, requires_grad=False)
        )
        layer._xcpu_keep_raw_wkv_offset = 128
        fp8_projs.append(layer)

    _, inner_fp8 = _build_fake_dspark_v41(
        num_layers=3, q_lora_rank=128, head_dim=128, hidden_size=256, projs=fp8_projs
    )
    fp8_kernel = object.__new__(XcpuFp8BlockScaledMMLinearKernel)
    fp8_kernel.weight_group_shape = (128, 128)
    for proj in fp8_projs:
        fp8_kernel.process_weights_after_loading(proj)
        assert getattr(proj, "_xcpu_raw_wkv_fp8", None) is not None

    inner_fp8.build_context_wkv_linear()
    assert inner_fp8._context_wkv_linear is not None
    assert inner_fp8._context_wkv_linear.params.out_features == 3 * 128
    for proj in fp8_projs:
        assert proj._xcpu_raw_wkv_fp8 is None
        assert proj._xcpu_raw_weight is None

    # 2. Unquantized / emulation BF16 fallback via dispatch_xcpu_bf16_linear
    bf16_projs = []
    indiv_bf16 = []
    for i in range(3):
        torch.manual_seed(20 + i)
        layer = torch.nn.Module()
        layer.bias = None
        w = (torch.randn((256, 256), dtype=torch.bfloat16) * 0.25).to("mcpu")
        layer.register_parameter("weight", torch.nn.Parameter(w, requires_grad=False))
        layer._xcpu_keep_raw_wkv_offset = 128
        layer._xcpu_keep_raw_weight = True
        assert dispatch_xcpu_bf16_linear(layer, remove_weight=True)
        indiv_bf16.append(layer._xcpu_bf16_linear)
        bf16_projs.append(layer)

    _, inner_bf16 = _build_fake_dspark_v41(
        num_layers=3, q_lora_rank=128, head_dim=128, hidden_size=256, projs=bf16_projs
    )
    inner_bf16.build_context_wkv_linear()
    assert inner_bf16._context_wkv_linear is not None
    assert inner_bf16._context_wkv_linear.params.out_features == 3 * 128
    for proj in bf16_projs:
        assert proj._xcpu_raw_weight is None

    recorded: list[torch.Tensor] = []
    monkeypatch.setattr(
        dsv41_model_mod,
        "_insert_context_kv",
        lambda attn, kv, positions, slots: recorded.append(kv.detach().cpu().clone()),
    )
    main_x = torch.randn((5, 256), dtype=torch.bfloat16).to("mcpu")
    pos = torch.arange(5, dtype=torch.int64, device="mcpu")
    with _dummy_run_scope(False):
        inner_bf16.precompute_and_store_context_kv(main_x, pos, [pos, pos, pos])
    assert len(recorded) == 3
    for layer, lin, got in zip(inner_bf16.layers, indiv_bf16, recorded, strict=True):
        ref = layer.attn.kv_norm(lin(main_x)[..., 128:])
        assert torch.allclose(got.float(), ref.cpu().float(), atol=1e-2, rtol=1e-2)


def test_dspark_deepseek_v41_idle_dummy_skips_projections():
    inner = object.__new__(XcpuDSparkDeepseekV41Model)
    torch.nn.Module.__init__(inner)
    inner.config = SimpleNamespace(head_dim=512, hidden_size=64)
    inner.hidden_size = 64
    inner.num_dspark_layers = 1

    proj_calls = []

    def fail_context_linear(_x):
        proj_calls.append("context_wkv")
        return torch.zeros((2, 512), dtype=torch.bfloat16, device="mcpu")

    inner._context_wkv_linear = fail_context_linear
    inner.main_proj = lambda _x: (_ for _ in ()).throw(
        AssertionError("main_proj must not run during dummy_run")
    )
    attn = torch.nn.Module()
    attn.kv_norm = lambda kv: kv
    layer = torch.nn.Module()
    layer.attn = attn
    inner.layers = torch.nn.ModuleList([layer])

    aux = torch.ones((2, 192), dtype=torch.bfloat16, device="mcpu")
    pos = torch.arange(2, dtype=torch.int64, device="mcpu")

    with _dummy_run_scope(True):
        combined = inner.combine_hidden_states(aux)
        assert tuple(combined.shape) == (2, 64)
        assert combined.dtype == torch.bfloat16
        assert torch.count_nonzero(combined.cpu()) == 0

        inner.precompute_and_store_context_kv(combined, pos, None)
        assert proj_calls == []

    # Profiling (is_dummy_run() == False, context_slot_mappings is None) still
    # runs _context_wkv_linear to reserve workspaces.
    with _dummy_run_scope(False):
        inner.precompute_and_store_context_kv(aux[:, :64], pos, None)
        assert proj_calls == ["context_wkv"]


def test_int8_context_wkv_stacked_matches_unstacked_and_frees_raw(monkeypatch):
    from vllm_xcpu_plugin.layers.int8_ue8m0 import (
        XcpuInt8Ue8m0Config,
        XcpuInt8Ue8m0LinearMethod,
    )

    q_lora_rank, head_dim, hidden_size = 128, 128, 256
    int8_projs = []
    indiv_linears = []
    method = XcpuInt8Ue8m0LinearMethod(XcpuInt8Ue8m0Config(weight_block_size=[16, 16]))
    for i in range(3):
        g = torch.Generator(device="cpu").manual_seed(50 + i)
        layer = torch.nn.Module()
        layer.bias = None
        w = torch.randint(
            -128,
            128,
            (q_lora_rank + head_dim, hidden_size),
            generator=g,
            dtype=torch.int8,
        ).to("mcpu")
        s = (
            torch
            .randint(
                120,
                126,
                ((q_lora_rank + head_dim) // 16, hidden_size // 16),
                generator=g,
            )
            .to(torch.uint8)
            .to("mcpu")
        )
        layer.register_parameter("weight", torch.nn.Parameter(w, requires_grad=False))
        layer.register_parameter(
            "weight_scale_inv", torch.nn.Parameter(s, requires_grad=False)
        )
        layer._xcpu_keep_raw_wkv_offset = q_lora_rank
        method.process_weights_after_loading(layer)
        assert getattr(layer, "_xcpu_raw_wkv_int8", None) is not None
        indiv_linears.append(layer._xcpu_int8_linear)
        int8_projs.append(layer)

    _, inner_int8 = _build_fake_dspark_v41(
        num_layers=3,
        q_lora_rank=q_lora_rank,
        head_dim=head_dim,
        hidden_size=hidden_size,
        projs=int8_projs,
    )
    inner_int8.build_context_wkv_linear()
    assert inner_int8._context_wkv_linear is not None
    assert inner_int8._context_wkv_linear.params.out_features == 3 * head_dim
    assert inner_int8._context_wkv_linear.params.in_features == hidden_size
    for proj in int8_projs:
        assert proj._xcpu_raw_wkv_int8 is None
        assert proj._xcpu_raw_weight is None

    recorded: list[torch.Tensor] = []
    monkeypatch.setattr(
        dsv41_model_mod,
        "_insert_context_kv",
        lambda attn, kv, positions, slots: recorded.append(kv.detach().cpu().clone()),
    )
    for num_tokens in (1, 7, 17):
        recorded.clear()
        torch.manual_seed(200 + num_tokens)
        main_x = (
            torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16) * 0.5
        ).to("mcpu")
        pos = torch.arange(num_tokens, dtype=torch.int64, device="mcpu")
        with _dummy_run_scope(False):
            inner_int8.precompute_and_store_context_kv(main_x, pos, [pos, pos, pos])
        assert len(recorded) == 3
        for layer, lin, got in zip(
            inner_int8.layers, indiv_linears, recorded, strict=True
        ):
            ref = layer.attn.kv_norm(lin(main_x)[..., q_lora_rank:])
            assert torch.equal(got, ref.cpu())
