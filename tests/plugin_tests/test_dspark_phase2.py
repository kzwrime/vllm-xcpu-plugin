# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_mcpu  # noqa: F401
from torch_xcpu import ops as xcpu_ops
from torch_xcpu.ops_defs.bf16_linear import (
    Bf16LinearBackend,
    initialize_bf16_linear,
)
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState

from vllm_xcpu_plugin import sampler_patch
from vllm_xcpu_plugin.models.dspark.speculator import XcpuDSparkSpeculator
from vllm_xcpu_plugin.models.rejection_sampler import patch as rej_patch_mod
from vllm_xcpu_plugin.models.rejection_sampler.patch import (
    _xcpu_rejection_sampler_verify,
)


def _make_fake_speculator(
    *,
    num_reqs: int = 3,
    n_spec: int = 4,
    vocab_size: int = 128,
    draft_vocab: int = 64,
    rank: int = 32,
    probabilistic: bool = False,
    use_d2t: bool = False,
    use_confidence_head: bool = False,
    pack_backend: Bf16LinearBackend | None = Bf16LinearBackend.INTEL_AMX,
) -> tuple[XcpuDSparkSpeculator, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    torch.manual_seed(31)
    hidden_size = 32
    base_logits_3d = (
        torch.randn((num_reqs, n_spec, draft_vocab), dtype=torch.bfloat16) * 0.5
    ).to("mcpu")
    w1 = (torch.randn((vocab_size, rank), dtype=torch.bfloat16) * 0.25).to("mcpu")
    w2 = (torch.randn((draft_vocab, rank), dtype=torch.bfloat16) * 0.25).to("mcpu")

    markov_w1 = torch.nn.Embedding(vocab_size, rank, dtype=torch.bfloat16).to("mcpu")
    with torch.no_grad():
        markov_w1.weight.copy_(w1)
    markov_w2 = torch.nn.Module()
    markov_w2.bias = None
    markov_w2.register_parameter("weight", torch.nn.Parameter(w2, requires_grad=False))
    if pack_backend is not None:
        markov_w2._xcpu_bf16_linear = initialize_bf16_linear(w2, backend=pack_backend)

    d2t_offsets = (
        torch.full((draft_vocab,), 3, dtype=torch.int64, device="mcpu")
        if use_d2t
        else None
    )

    spec = object.__new__(XcpuDSparkSpeculator)
    spec.device = torch.device("mcpu")
    spec.dtype = torch.bfloat16
    spec.vocab_size = vocab_size
    spec.num_speculative_steps = n_spec
    spec.num_query_per_req = n_spec
    spec.max_num_reqs = max(num_reqs + 2, 8)
    spec.max_num_tokens = spec.max_num_reqs * n_spec
    spec._draft_topk = None
    spec.use_acceptance_estimator = False
    spec.draft_watermarker = None
    spec.use_fp64_gumbel = True
    spec.use_confidence_head = use_confidence_head
    spec._d2t_scatter_index = None
    spec._draft_scatter_buf = None
    spec._d2t_target_ids = (
        (
            torch.arange(draft_vocab, dtype=torch.int64, device="mcpu") + d2t_offsets
        ).contiguous()
        if d2t_offsets is not None
        else None
    )
    spec._sp_gather_buf = None
    spec._confidence_markov_embeds = None
    spec._step_cols = torch.arange(n_spec, dtype=torch.int32, device="mcpu")
    spec._anchor_idx = (
        torch.arange(spec.max_num_reqs, dtype=torch.int64, device="mcpu") * n_spec
    )
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.randint(
            0, vocab_size, (spec.max_num_tokens,), dtype=torch.int32, device="mcpu"
        )
    )
    spec.sample_indices = torch.arange(
        spec.max_num_tokens, dtype=torch.int64, device="mcpu"
    )
    spec.sample_idx_mapping = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32, device="mcpu"), n_spec
    )
    spec.sample_pos = torch.arange(
        10, 10 + spec.max_num_tokens, dtype=torch.int64, device="mcpu"
    )
    spec.temperature = torch.tensor(
        [0.0 if i % 2 == 0 else 0.7 for i in range(spec.max_num_reqs)],
        dtype=torch.float32,
        device="mcpu",
    )
    spec.seeds = torch.arange(
        100, 100 + spec.max_num_reqs, dtype=torch.int64, device="mcpu"
    )
    spec.draft_tokens = torch.full(
        (spec.max_num_reqs, n_spec), -1, dtype=torch.int64, device="mcpu"
    )
    spec.draft_logits = (
        torch.zeros(
            (spec.max_num_reqs, n_spec, draft_vocab),
            dtype=torch.bfloat16,
            device="mcpu",
        )
        if probabilistic
        else None
    )
    spec.draft_token_confidence_probs = torch.zeros(
        (spec.max_num_reqs, n_spec), dtype=torch.float32, device="mcpu"
    )

    inner_model = SimpleNamespace(
        markov_head=SimpleNamespace(
            markov_w1=markov_w1,
            markov_w2=markov_w2,
            embed=lambda ids: markov_w1(ids),
            bias=lambda emb, _lp: (emb.float() @ w2.float().T).bfloat16(),
        )
    )
    model = SimpleNamespace(
        model=inner_model,
        logits_processor=SimpleNamespace(scale=1.0, soft_cap=None),
        draft_id_to_target_id=d2t_offsets,
        compute_draft_logits=lambda _h: base_logits_3d.reshape(
            num_reqs * n_spec, draft_vocab
        ).clone(),
        markov_embed=lambda ids: inner_model.markov_head.embed(ids),
        markov_bias=lambda emb: inner_model.markov_head.bias(emb, None),
        compute_confidence=lambda h, m: torch.sigmoid(
            (h[:, :rank].float() * m.float()).sum(dim=-1)
        ),
        map_draft_to_target=lambda ids: (
            ids if d2t_offsets is None else ids + d2t_offsets[ids]
        ),
    )
    spec.model = model
    head_hidden = torch.randn(
        (num_reqs * n_spec, hidden_size), dtype=torch.bfloat16, device="mcpu"
    )
    return spec, head_hidden, w1, w2


@pytest.mark.parametrize(
    "pack_backend",
    [Bf16LinearBackend.INTEL_AMX, Bf16LinearBackend.PORTABLE, None],
)
@pytest.mark.parametrize("probabilistic", [False, True])
def test_speculator_sample_sequential_fused_and_fallback(
    monkeypatch, pack_backend: Bf16LinearBackend | None, probabilistic: bool
) -> None:
    import vllm.v1.worker.gpu.sample.gumbel as gumbel_module

    monkeypatch.setattr(gumbel_module, "gumbel_sample", gumbel_module.gumbel_sample)
    monkeypatch.setattr(sampler_patch, "_GUMBEL_SAMPLE_PATCHED", False)
    monkeypatch.setenv("VLLM_USE_XCPU_GUMBEL_SAMPLE", "1")
    sampler_patch.maybe_patch_vllm_gumbel_sample()

    use_d2t = not probabilistic
    spec_fused, head_hidden, _, _ = _make_fake_speculator(
        probabilistic=probabilistic,
        use_d2t=use_d2t,
        use_confidence_head=True,
        pack_backend=pack_backend,
    )
    spec_ref, head_hidden_ref, _, _ = _make_fake_speculator(
        probabilistic=probabilistic,
        use_d2t=use_d2t,
        use_confidence_head=True,
        pack_backend=None,
    )
    spec_ref.use_acceptance_estimator = True
    spec_ref._maybe_predict_acceptance = lambda *_a, **_k: None

    calls: list[str] = []
    orig_op = xcpu_ops.dspark_markov_sample

    def _spy(*args, **kwargs):
        calls.append("fused")
        return orig_op(*args, **kwargs)

    monkeypatch.setattr(xcpu_ops, "dspark_markov_sample", _spy)

    spec_fused._sample_sequential(3, head_hidden)
    assert calls == ["fused"]

    # Fallback path triggers in-place .add_() and must NOT invoke fused op.
    spec_ref._sample_sequential(3, head_hidden_ref)
    assert calls == ["fused"]

    if pack_backend is None:
        torch.testing.assert_close(
            spec_fused.draft_tokens[:3].cpu(),
            spec_ref.draft_tokens[:3].cpu(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            spec_fused.draft_token_confidence_probs[:3].cpu(),
            spec_ref.draft_token_confidence_probs[:3].cpu(),
            rtol=1e-4,
            atol=1e-4,
        )
        if probabilistic:
            assert (
                spec_fused.draft_logits is not None
                and spec_ref.draft_logits is not None
            )
            torch.testing.assert_close(
                spec_fused.draft_logits[:3].cpu(),
                spec_ref.draft_logits[:3].cpu(),
                rtol=0,
                atol=0,
            )


def _build_fake_sampler(vocab_size: int = 256, max_reqs: int = 8):
    ss = SamplingStates(max_reqs, vocab_size)
    # Use host CPU arrays via SamplingStates backing in unit tests.
    req_states = SimpleNamespace(
        max_num_reqs=max_reqs,
        vocab_size=vocab_size,
        device=torch.device("mcpu"),
        prompt_len=SimpleNamespace(np=np.zeros(max_reqs, dtype=np.int32)),
    )
    vcfg = SimpleNamespace(reasoning_config=None)
    logit_bias = LogitBiasState(vcfg, req_states)
    penalties = PenaltiesState(vcfg, req_states)
    bad_words = BadWordsState(vcfg, req_states)
    thinking = ThinkingBudgetState(req_states, None)

    ss.temperature.gpu = ss.temperature.np
    temp_t = torch.tensor(ss.temperature.np, dtype=torch.float32, device="mcpu")
    seeds_t = torch.zeros((max_reqs,), dtype=torch.int64, device="mcpu")
    ss.temperature.gpu = temp_t
    ss.seeds.gpu = seeds_t

    apply_calls: list[int] = []

    def _apply_sampling_params(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        seq_lens_upper_bound_np,
        skip_top_k_top_p: bool = False,
    ):
        apply_calls.append(1)
        out = logits.float().clone()
        xcpu_ops.vllm_temperature_kernel(out, expanded_idx_mapping, ss.temperature.gpu)
        return out

    sampler = SimpleNamespace(
        return_sampling_mask=False,
        logprobs_mode="raw_logprobs",
        sampling_states=ss,
        thinking_budget_state=thinking,
        logits_processors=[logit_bias, penalties, bad_words],
        needs_logits_processing=np.zeros(max_reqs, dtype=bool),
        use_fp64_gumbel=True,
        apply_sampling_params=_apply_sampling_params,
    )
    return sampler, apply_calls


@pytest.mark.parametrize(
    "feature", ["temp_only", "top_k", "penalties", "logit_bias", "bad_words"]
)
def test_rejection_sampler_verify_routing_and_parity(monkeypatch, feature: str) -> None:
    vocab_size = 256
    num_reqs = 2
    n_spec = 3
    sampler, apply_calls = _build_fake_sampler(vocab_size=vocab_size)
    ss = sampler.sampling_states
    ss.temperature.np[:num_reqs] = [0.0, 0.7]
    ss.temperature.gpu = torch.tensor(
        ss.temperature.np, dtype=torch.float32, device="mcpu"
    )
    sampler.needs_logits_processing[:num_reqs] = True

    if feature == "top_k":
        ss.top_k.np[1] = 32
    elif feature == "penalties":
        sampler.logits_processors[1].use_penalty[1] = True
    elif feature == "logit_bias":
        sampler.logits_processors[0].use_logit_bias[1] = True
    elif feature == "bad_words":
        sampler.logits_processors[2].num_bad_words.np[1] = 1

    rs = SimpleNamespace(
        sampler=sampler,
        num_speculative_steps=n_spec,
        use_block_verification=False,
        synthetic_conditional_rates=None,
        watermark_key=None,
        _watermarking_kwargs=lambda *_a, **_k: {},
    )

    torch.manual_seed(9)
    logits = (
        (torch.randn((num_reqs * (n_spec + 1), vocab_size)) * 1.5).bfloat16().to("mcpu")
    )
    draft_logits = (torch.randn((8, n_spec, vocab_size)) * 1.5).bfloat16().to("mcpu")
    draft_sampled = torch.randint(
        0, vocab_size, (logits.shape[0],), dtype=torch.int32, device="mcpu"
    )
    pos = torch.arange(logits.shape[0], dtype=torch.int64, device="mcpu")
    cu_num_logits = torch.tensor([0, 4, 8], dtype=torch.int32, device="mcpu")
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
    idx_mapping_np = np.array([0, 1], dtype=np.int64)
    expanded_idx_mapping = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device="mcpu"
    )
    expanded_local_pos = torch.tensor(
        [0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device="mcpu"
    )
    seq_lens_ub = np.array([10, 10], dtype=np.int32)

    fused_calls: list[bool] = []
    orig_fused = xcpu_ops.vllm_rejection_sample_fused

    def _spy_fused(*args, **kwargs):
        fused_calls.append(bool(args[-1]))
        return orig_fused(*args, **kwargs)

    monkeypatch.setattr(xcpu_ops, "vllm_rejection_sample_fused", _spy_fused)

    proc, sampled, num_sampled = _xcpu_rejection_sampler_verify(
        rs,
        logits,
        draft_logits,
        draft_sampled,
        pos,
        cu_num_logits,
        idx_mapping,
        idx_mapping_np,
        expanded_idx_mapping,
        expanded_local_pos,
        seq_lens_ub,
        draft_sampled,
    )

    if feature == "temp_only":
        assert apply_calls == []
        assert fused_calls == [True]
        assert proc is logits

        # Must match temperature + fused kernel without apply_target_temperature.
        proc_logits_ref = logits.float().clone()
        xcpu_ops.vllm_temperature_kernel(
            proc_logits_ref, expanded_idx_mapping, ss.temperature.gpu
        )
        ref_sampled = torch.empty_like(sampled)
        ref_num = torch.empty_like(num_sampled)
        orig_fused(
            ref_sampled,
            ref_num,
            proc_logits_ref,
            draft_logits,
            cu_num_logits,
            expanded_idx_mapping,
            draft_sampled,
            ss.temperature.gpu,
            ss.seeds.gpu,
            pos,
            vocab_size,
            True,
            False,
        )
        torch.testing.assert_close(num_sampled.cpu(), ref_num.cpu(), rtol=0, atol=0)
        for r in range(num_reqs):
            n = int(ref_num[r].item())
            torch.testing.assert_close(
                sampled[r, :n].cpu(), ref_sampled[r, :n].cpu(), rtol=0, atol=0
            )
    else:
        assert apply_calls == [1]
        assert fused_calls == [False]
        assert proc.dtype == torch.float32


def test_rejection_sampler_verify_watermark_fallback(monkeypatch) -> None:
    vocab_size = 64
    sampler, apply_calls = _build_fake_sampler(vocab_size=vocab_size)
    rs = SimpleNamespace(
        sampler=sampler,
        num_speculative_steps=2,
        use_block_verification=True,
        synthetic_conditional_rates=None,
        watermark_key=None,
        _watermarking_kwargs=lambda *_a, **_k: {},
    )
    fallback_called = []
    monkeypatch.setattr(
        rej_patch_mod,
        "rejection_sample",
        lambda *_a, **_k: (
            fallback_called.append(True)
            or (
                torch.zeros((1, 3), dtype=torch.int64, device="mcpu"),
                torch.ones((1,), dtype=torch.int32, device="mcpu"),
            )
        ),
    )
    logits = torch.randn((3, vocab_size), dtype=torch.float32, device="mcpu")
    ds = torch.zeros((3,), dtype=torch.int32, device="mcpu")
    pos = torch.arange(3, dtype=torch.int64, device="mcpu")
    cu = torch.tensor([0, 3], dtype=torch.int32, device="mcpu")
    idx_m = torch.tensor([0], dtype=torch.int32, device="mcpu")
    idx_np = np.array([0], dtype=np.int64)
    exp_m = torch.zeros((3,), dtype=torch.int32, device="mcpu")
    exp_lp = torch.tensor([0, 1, 2], dtype=torch.int32, device="mcpu")
    _xcpu_rejection_sampler_verify(
        rs, logits, None, ds, pos, cu, idx_m, idx_np, exp_m, exp_lp, np.array([5]), ds
    )
    assert apply_calls == [1]
    assert fallback_called == [True]
