from __future__ import annotations

from dataclasses import is_dataclass
from types import SimpleNamespace

import pytest
import torch
import torch_xcpu

from vllm_xcpu_plugin.gdn_patch import _xcpu_causal_conv1d_fn


@pytest.fixture(autouse=True)
def install_gdn_metadata_patch():
    from vllm_xcpu_plugin.gdn_metadata_patch import maybe_patch_gdn_metadata

    maybe_patch_gdn_metadata()


def test_gdn_metadata_patch_is_idempotent_and_preserves_compatibility_source():
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    from vllm_xcpu_plugin import gdn_metadata_patch
    from vllm_xcpu_plugin.upstream_compatibility import verify_upstream_compatibility

    patched = GDNAttentionMetadataBuilder.build
    gdn_metadata_patch.maybe_patch_gdn_metadata()
    assert GDNAttentionMetadataBuilder.build is patched
    assert patched.__wrapped__ is gdn_metadata_patch._ORIGINAL_GDN_METADATA_BUILD
    assert len(verify_upstream_compatibility(("gdn_metadata",))) == 14


@pytest.mark.parametrize(
    "target", ["build", "_build_chunk_metadata", "_stage_spec_decode", "_stage_decode"]
)
def test_gdn_metadata_patch_rejects_source_drift_before_replacing_builder(
    monkeypatch, target
):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    from vllm_xcpu_plugin import gdn_metadata_patch
    from vllm_xcpu_plugin.fake_triton.runtime import KernelVersionError

    def changed_build(self, *args, **kwargs):
        raise AssertionError("changed upstream builder must not run")

    original_build = GDNAttentionMetadataBuilder.build
    monkeypatch.setattr(GDNAttentionMetadataBuilder, target, changed_build)
    monkeypatch.setattr(gdn_metadata_patch, "_ORIGINAL_GDN_METADATA_BUILD", None)
    with pytest.raises(
        KernelVersionError, match=f"GDNAttentionMetadataBuilder.{target}"
    ):
        gdn_metadata_patch.maybe_patch_gdn_metadata()
    assert GDNAttentionMetadataBuilder.build is (
        changed_build if target == "build" else original_build
    )
    assert gdn_metadata_patch._ORIGINAL_GDN_METADATA_BUILD is None


@pytest.mark.parametrize("device,backend", [("cpu", "triton"), ("mcpu", "cutedsl")])
def test_gdn_metadata_patch_fallback_forwards_original_arguments(
    monkeypatch, device, backend
):
    from vllm_xcpu_plugin import gdn_metadata_patch

    recorded = []
    result = object()

    def original(*args):
        recorded.append(args)
        return result

    monkeypatch.setattr(gdn_metadata_patch, "_ORIGINAL_GDN_METADATA_BUILD", original)
    builder = SimpleNamespace(gdn_prefill_backend=backend)
    common = SimpleNamespace(
        query_start_loc=torch.empty(1, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.zeros(1, dtype=torch.int32),
    )
    accepted, draft = object(), object()
    assert (
        gdn_metadata_patch._xcpu_gdn_metadata_build(
            builder, 7, common, accepted, draft, True
        )
        is result
    )
    assert recorded == [(builder, 7, common, accepted, draft, True)]


@pytest.mark.parametrize(
    "draft_counts,query_lens",
    [
        ([5, 5], [6, 6]),
        ([5, -1, -1], [6, 0, 0]),
        ([-1, 5, -1, 5], [7, 6, 1, 6]),
        ([-1, -1], [1, 1]),
        ([0, 0, -1], [1, 1, 0]),
        ([5, 0, -1], [6, 1, 9]),
        ([-1, -1, -1], [1, 65, 0]),
        ([-1, -1], [129, 65]),
        ([-1, 5, -1, -1], [0, 6, 65, 0]),
    ],
)
@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("use_full_cuda_graph", [False, True])
@pytest.mark.parametrize("first_chunk", [False, True])
@pytest.mark.parametrize("checkpoint_blocks", [0, 1])
def test_gdn_metadata_device_indices_match_cpu(
    draft_counts,
    query_lens,
    mamba_cache_mode,
    strided,
    use_full_cuda_graph,
    first_chunk,
    checkpoint_blocks,
    token_padding=0,
    use_spec_decode=True,
    prealigned=False,
    capture=False,
):
    """Exercise pure, padded, interleaved mixed, and non-spec metadata builds."""
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    batch = len(query_lens)
    effective_drafts = draft_counts if draft_counts is not None else [-1] * batch
    query_cpu = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0).int()
    table_cpu = torch.arange(batch * 32, dtype=torch.int32).view(batch, 32)

    def to_device(tensor, device):
        if strided:
            # Exercise views with non-unit strides and nonzero storage offsets.
            storage = torch.empty(
                (*tensor.shape[:-1], 2 * tensor.shape[-1] + 1),
                dtype=tensor.dtype,
                device=device,
            )
            view = storage[..., 1::2]
            view.copy_(tensor.to(device))
            return view
        return tensor.to(device)

    def build(device):
        # Isolate build() from model loading and backend selection.
        builder = object.__new__(GDNAttentionMetadataBuilder)
        builder.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(
                mamba_cache_mode=mamba_cache_mode, prefix_match_unit=None
            ),
            speculative_config=None,
        )
        builder.kv_cache_spec = MambaSpec(
            block_size=16,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=5,
            num_prefill_checkpoint_blocks=checkpoint_blocks,
        )
        builder.use_spec_decode = use_spec_decode
        builder.num_spec = 5
        builder.use_full_cuda_graph = use_full_cuda_graph
        builder.decode_cudagraph_max_bs = batch * 6
        for name, shape, dtype in (
            ("spec_state_indices_tensor", (batch * 6, 6), torch.int32),
            ("non_spec_state_indices_tensor", (batch * 6,), torch.int32),
            ("spec_sequence_masks", (batch * 6,), torch.bool),
            ("spec_token_indx", (batch * 36,), torch.int32),
            ("non_spec_token_indx", (batch * 36,), torch.int32),
            ("spec_query_start_loc", (batch * 6 + 1,), torch.int32),
            ("non_spec_query_start_loc", (batch * 6 + 1,), torch.int32),
            ("num_accepted_tokens", (batch * 6,), torch.int32),
        ):
            setattr(builder, name, torch.empty(shape, dtype=dtype, device=device))
        if prealigned:
            builder.mamba_aligned_state_indices = to_device(
                torch.arange((batch + 1) * 6, dtype=torch.int32).view(batch + 1, 6)
                + 5000,
                device,
            )
        builder.gdn_prefill_backend = "triton"
        from vllm.model_executor.layers.mamba.checkpoint import (
            MambaPrefillCheckpointBuilder,
        )

        builder.checkpoint_builder = MambaPrefillCheckpointBuilder(
            builder.vllm_config, builder.kv_cache_spec
        )
        common = CommonAttentionMetadata(
            query_start_loc=to_device(query_cpu, device),
            query_start_loc_cpu=to_device(query_cpu, "cpu"),
            seq_lens=to_device(
                torch.tensor(query_lens, dtype=torch.int32)
                + torch.arange(batch, dtype=torch.int32) * 17,
                device,
            ),
            seq_lens_cpu_upper_bound=to_device(
                torch.tensor(query_lens, dtype=torch.int32)
                + torch.arange(batch, dtype=torch.int32) * 17,
                "cpu",
            ),
            is_prefilling=(
                torch.tensor([
                    q > 0 and d < 0 for q, d in zip(query_lens, effective_drafts)
                ])
                if first_chunk
                else None
            ),
            num_reqs=batch,
            num_actual_tokens=sum(query_lens) + token_padding,
            max_query_len=max(query_lens),
            max_seq_len=max(query_lens) + batch * 17,
            block_table_tensor=to_device(table_cpu, device),
            slot_mapping=torch.empty(sum(query_lens), dtype=torch.int64, device=device),
        )
        if capture:
            return builder.build_for_cudagraph_capture(common)
        return builder.build(
            0,
            common,
            num_accepted_tokens=to_device(
                torch.arange(1, batch + 1, dtype=torch.int32), device
            ),
            num_decode_draft_tokens_cpu=to_device(
                torch.tensor(draft_counts, dtype=torch.int32), "cpu"
            )
            if draft_counts is not None
            else None,
        )

    reference, actual = build("cpu"), build("mcpu")
    if first_chunk and query_lens[0] == 1 and all(d < 0 for d in effective_drafts):
        # An independent assertion prevents matching two broken classifiers.
        assert actual.num_prefills > 0
        assert actual.has_initial_state is not None
        assert not actual.has_initial_state.cpu()[0].item()

    if (
        use_spec_decode
        and not capture
        and draft_counts is not None
        and any(d == 0 for d in draft_counts)
        and all(d <= 0 for d in draft_counts)
    ):
        assert actual.num_spec_decodes == sum(d >= 0 for d in draft_counts)
        assert actual.num_decodes == 0
        torch.testing.assert_close(
            actual.num_accepted_tokens[: actual.num_spec_decodes].cpu(),
            torch.tensor(
                [i + 1 for i, d in enumerate(draft_counts) if d >= 0],
                dtype=torch.int32,
            ),
        )

    def assert_equal(expected, result, name):
        if isinstance(expected, torch.Tensor):
            assert result.device.type == expected.device.type or (
                result.device.type == "mcpu" and expected.device.type == "cpu"
            ), name
            torch.testing.assert_close(result.cpu(), expected.cpu(), msg=name)
        elif is_dataclass(expected):
            for key, value in vars(expected).items():
                assert_equal(value, getattr(result, key), f"{name}.{key}")
        elif isinstance(expected, dict):
            assert result.keys() == expected.keys(), name
            for key in expected:
                assert_equal(expected[key], result[key], f"{name}.{key}")
        else:
            assert result == expected, name

    for name, expected in vars(reference).items():
        assert_equal(expected, getattr(actual, name), name)

    # Every slot must remain a tensor input even if absent in the traced batch.
    # Aliased metadata values must not become one Python object in Dynamo.
    assert actual.xcpu_core_counts.device.type == "cpu"
    assert actual.xcpu_core_counts.tolist() == [
        reference.num_prefills,
        reference.num_decodes,
        reference.num_spec_decodes,
        reference.num_actual_tokens,
        reference.num_decode_tokens,
    ]
    assert len({id(t) for t in actual.xcpu_core_tensors}) == 11
    for name, tensor in zip(
        (
            "has_initial_state",
            "spec_query_start_loc",
            "non_spec_query_start_loc",
            "spec_state_indices_tensor",
            "non_spec_state_indices_tensor",
            "spec_token_indx",
            "non_spec_token_indx",
            "num_accepted_tokens",
            "prefill_query_start_loc",
            "prefill_state_indices",
            "prefill_has_initial_state",
        ),
        actual.xcpu_core_tensors,
    ):
        source = getattr(actual, name)
        if source is None:
            assert tensor.numel() == 0
        else:
            assert tensor is not source
            assert tensor.data_ptr() == source.data_ptr()
            assert_equal(source, tensor, name)


@pytest.mark.parametrize(
    "mode", ["padded", "no_spec", "no_drafts", "prealigned", "capture"]
)
@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize(
    "drafts,lens",
    [
        ([-1, -1, -1], [1, 65, 0]),
        ([5, -1, -1], [6, 0, 0]),
        ([-1, 2, -1, 0], [65, 3, 1, 1]),
        ([-1, -1, -1], [1, 1, 0]),
    ],
)
def test_gdn_metadata_upgrade_branches(
    mode, mamba_cache_mode, full_graph, drafts, lens
):
    if mode == "capture":
        lens = [1, 1, 0] if drafts[0] < 0 else [6, 6, 0]
    test_gdn_metadata_device_indices_match_cpu(
        draft_counts=None if mode == "no_drafts" else drafts,
        query_lens=lens,
        mamba_cache_mode=mamba_cache_mode,
        strided=True,
        use_full_cuda_graph=full_graph,
        first_chunk=False,
        checkpoint_blocks=1,
        token_padding=4 if mode == "padded" else 0,
        use_spec_decode=mode != "no_spec",
        prealigned=mode == "prealigned",
        capture=mode == "capture",
    )


def test_gdn_metadata_cpu_outputs_are_ready_while_device_stream_is_pending():
    """CPU counts/conv data are synchronous; device gather observes prior writes."""
    stream = torch.Stream(device="mcpu")
    query_cpu = torch.tensor([0, 6, 71, 72], dtype=torch.int32)
    with stream:
        query = query_cpu.to("mcpu")
        seq_lens = torch.tensor([16, 65, 9], dtype=torch.int32).to("mcpu")
        table = torch.empty((3, 6), dtype=torch.int32, device="mcpu")
        accepted = torch.tensor([3, 1, 1], dtype=torch.int32).to("mcpu")
        # Leave the table unavailable until after build() returns.
        blocker = torch.empty(1, dtype=torch.int64, device="mcpu")
        torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 300)
        table.fill_(42)
        tensors = torch.ops.torch_xcpu.allocate_gdn_metadata_outputs(
            query, query_cpu, torch.tensor([5, -1, -1], dtype=torch.int64), 5, 65
        )
        result = torch.ops.torch_xcpu.build_gdn_metadata_out(
            query,
            query_cpu,
            seq_lens,
            table,
            accepted,
            torch.tensor([5, -1, -1], dtype=torch.int64),
            5,
            72,
            65,
            0,
            64,
            tensors,
        )
        assert result is None
        counts = tensors[21].tolist()[:6]
        assert not stream.query()
        assert counts == [2, 66, 0, 0, 1, 6]
        # nums, mlist, offsetlist are CPU tensors, immediately readable.
        torch.testing.assert_close(tensors[14], torch.tensor([9, 1], dtype=torch.int32))
        torch.testing.assert_close(tensors[15], torch.tensor([0] * 9 + [1]))
        torch.testing.assert_close(
            tensors[16], torch.tensor(list(range(9)) + [0], dtype=torch.int32)
        )
        # Retain the first metadata while a later build is queued. Neither
        # device outputs nor asynchronous CPU staging may overwrite it.
        table.fill_(84)
        later_draft = torch.tensor([-1, 5, -1], dtype=torch.int32)
        later = torch.ops.torch_xcpu.allocate_gdn_metadata_outputs(
            query, query_cpu, later_draft, 5, 65
        )
        torch.ops.torch_xcpu.build_gdn_metadata_out(
            query,
            query_cpu,
            seq_lens,
            table,
            accepted,
            later_draft,
            5,
            72,
            65,
            0,
            64,
            later,
        )
    stream.synchronize()
    torch.testing.assert_close(
        tensors[3].cpu(), torch.full((1, 6), 42, dtype=torch.int32)
    )
    torch.testing.assert_close(
        tensors[4].cpu(), torch.full((2,), 42, dtype=torch.int32)
    )
    torch.testing.assert_close(tensors[0].cpu(), torch.tensor([False, True]))
    torch.testing.assert_close(tensors[8].cpu(), torch.tensor([3], dtype=torch.int32))
    torch.testing.assert_close(
        later[3].cpu(), torch.full((1, 6), 84, dtype=torch.int32)
    )
    torch.testing.assert_close(later[8].cpu(), torch.tensor([1], dtype=torch.int32))


def test_causal_conv1d_fn_forwards_cache_all_contract(monkeypatch):
    recorded = {}

    def fake_causal_conv1d_fn(**kwargs):
        recorded.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(
        torch_xcpu.ops,
        "causal_conv1d_fn",
        fake_causal_conv1d_fn,
    )

    values = {
        name: object()
        for name in (
            "x",
            "weight",
            "bias",
            "conv_states",
            "query_start_loc",
            "cache_indices",
            "has_initial_state",
            "block_idx_first_scheduled_token",
            "block_idx_last_scheduled_token",
            "initial_state_idx",
            "num_computed_tokens",
            "metadata",
        )
    }
    result = _xcpu_causal_conv1d_fn(
        values["x"],
        values["weight"],
        values["bias"],
        values["conv_states"],
        values["query_start_loc"],
        cache_indices=values["cache_indices"],
        has_initial_state=values["has_initial_state"],
        block_idx_first_scheduled_token=values["block_idx_first_scheduled_token"],
        block_idx_last_scheduled_token=values["block_idx_last_scheduled_token"],
        initial_state_idx=values["initial_state_idx"],
        num_computed_tokens=values["num_computed_tokens"],
        block_size_to_align=8,
        metadata=values["metadata"],
        validate_data=True,
    )

    assert result is values["x"]
    for name, value in values.items():
        assert recorded[name] is value
    assert recorded["block_size_to_align"] == 8
    assert recorded["validate_data"] is True


def test_gdn_metadata_cross_stream_inputs_survive_allocator_reuse():
    """Queued raw-pointer reads keep foreign-stream storage alive after return."""
    import gc

    # Collect before the timed window. Tensor owners below are released by
    # refcount when deleted; GC inside the window can outlast the queued sleep.
    gc.collect()
    query_cpu = torch.tensor([0, 6], dtype=torch.int32)
    query = query_cpu.to("mcpu")
    seq_lens = torch.tensor([16], dtype=torch.int32).to("mcpu")
    table = torch.arange(6, dtype=torch.int32, device="mcpu").view(1, 6)
    accepted = torch.ones(1, dtype=torch.int32, device="mcpu")
    torch.mcpu.synchronize()
    tensors = torch.ops.torch_xcpu.allocate_gdn_metadata_outputs(
        query, query_cpu, torch.tensor([5], dtype=torch.int32), 5, 6
    )
    stream = torch.Stream(device="mcpu")
    with stream:
        blocker = torch.empty(1, dtype=torch.int64, device="mcpu")
        torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 300)
        torch.ops.torch_xcpu.build_gdn_metadata_out(
            query,
            query_cpu,
            seq_lens,
            table,
            accepted,
            torch.tensor([5], dtype=torch.int32),
            5,
            6,
            6,
            0,
            64,
            tensors,
        )
    old_ptrs = {
        table.data_ptr(),
        query.data_ptr(),
        seq_lens.data_ptr(),
        accepted.data_ptr(),
    }
    old_ptrs.update(
        t.data_ptr()
        for t in tensors
        if t is not None and t.device.type == "mcpu" and t.numel() > 0
    )
    spec_state_indices_tensor = tensors[3]
    del table, query, seq_lens, accepted, tensors
    replacements = [
        torch.full((1, 6), 777, dtype=torch.int32, device="mcpu") for _ in range(8)
    ]
    # Once work completes, reuse is legal even if the host was descheduled.
    if not stream.query():
        assert all(t.data_ptr() not in old_ptrs for t in replacements)
    stream.synchronize()
    torch.testing.assert_close(
        spec_state_indices_tensor.cpu(), torch.arange(6, dtype=torch.int32).view(1, 6)
    )


def test_gdn_metadata_out_schema_has_no_returned_aliases():
    """Mutation is declared on caller-owned outputs rather than returned views."""
    query_cpu = torch.tensor([0, 1], dtype=torch.int32)
    query = query_cpu.to("mcpu")
    tensors = torch.ops.torch_xcpu.allocate_gdn_metadata_outputs(
        query, query_cpu, None, 0, 1
    )
    args = (
        query,
        query_cpu,
        torch.tensor([9], dtype=torch.int32, device="mcpu"),
        torch.tensor([[1]], dtype=torch.int32, device="mcpu"),
        None,
        None,
        0,
        1,
        1,
        0,
        64,
        tensors,
    )
    torch.library.opcheck(
        torch.ops.torch_xcpu.build_gdn_metadata_out.default,
        args,
        test_utils=("test_schema",),
    )
