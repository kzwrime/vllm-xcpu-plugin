import pytest
import torch
import torch_mcpu  # noqa: F401

from vllm_xcpu_plugin import sampler_patch


@pytest.mark.parametrize("use_fp64", [False, True])
def test_patched_gumbel_caches_separate_source(monkeypatch, use_fp64):
    import vllm.v1.worker.gpu.sample.gumbel as gumbel_module
    import vllm.v1.worker.gpu.sample.sampler as sampler_module

    # Restore module bindings on test exit, including the direct import.
    monkeypatch.setattr(gumbel_module, "gumbel_sample", gumbel_module.gumbel_sample)
    monkeypatch.setattr(sampler_module, "gumbel_sample", sampler_module.gumbel_sample)
    monkeypatch.setattr(sampler_patch, "_GUMBEL_SAMPLE_PATCHED", False)
    monkeypatch.setenv("VLLM_USE_XCPU_GUMBEL_SAMPLE", "1")
    sampler_patch.maybe_patch_vllm_gumbel_sample()
    assert sampler_module.gumbel_sample is gumbel_module.gumbel_sample
    logits = torch.tensor(
        [[0.0, 8.0, -1.0, -2.0], [9.0, 0.0, 1.0, -2.0]], device="mcpu"
    )
    source = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=torch.bfloat16,
        device="mcpu",
    )
    cache = torch.full((2, 2, 6), -99.0, dtype=torch.bfloat16, device="mcpu")
    mapping = torch.tensor([1, 0], dtype=torch.int32, device="mcpu")
    temp = torch.zeros(2, device="mcpu")
    seed = torch.zeros(2, dtype=torch.int64, device="mcpu")
    pos = torch.arange(2, dtype=torch.int64, device="mcpu")
    col = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
    actual = sampler_module.gumbel_sample(
        logits,
        mapping,
        temp,
        seed,
        pos,
        True,
        True,
        cache,
        col,
        use_fp64,
        logits_cache_source=source,
    )
    torch.mcpu.synchronize()
    assert actual.cpu().tolist() == [1, 0]
    expected = torch.full((2, 2, 6), -99.0, dtype=torch.bfloat16, device="cpu")
    expected[1, 0, :4] = source.cpu()[0]
    expected[0, 1, :4] = source.cpu()[1]
    assert torch.equal(cache.cpu(), expected)


def test_patched_gumbel_forwards_non_xcpu_source(monkeypatch):
    import vllm.v1.worker.gpu.sample.gumbel as gumbel_module
    import vllm.v1.worker.gpu.sample.sampler as sampler_module

    calls = []
    sentinel = object()

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(gumbel_module, "gumbel_sample", original)
    monkeypatch.setattr(sampler_module, "gumbel_sample", sampler_module.gumbel_sample)
    monkeypatch.setattr(sampler_patch, "_GUMBEL_SAMPLE_PATCHED", False)
    monkeypatch.setenv("VLLM_USE_XCPU_GUMBEL_SAMPLE", "1")
    sampler_patch.maybe_patch_vllm_gumbel_sample()
    logits = torch.zeros(1, 2, device="cpu")
    source = torch.ones_like(logits)
    args = (logits, None, None, None, None, True, False, None, None, True)
    assert gumbel_module.gumbel_sample(*args, logits_cache_source=source) is sentinel
    assert calls[0][0] == args
    assert calls[0][1]["logits_cache_source"] is source
