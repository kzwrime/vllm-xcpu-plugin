# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
import torch_mcpu  # noqa: F401
from torch_xcpu import ops as xcpu_ops

from vllm_xcpu_plugin.models.deepseek_v41.model import (
    SpShardedAuxHiddenStates,
    _XcpuDeepseekV41SourceLayers,
)
from vllm_xcpu_plugin.models.dspark import speculator as dspark_spec_mod
from vllm_xcpu_plugin.models.dspark.speculator import XcpuDSparkSpeculator


def test_combine_target_hidden_states_sp_sharded_uses_single_gather(monkeypatch):
    spec = object.__new__(XcpuDSparkSpeculator)
    spec.max_num_tokens = 16
    spec._sp_gather_buf = None
    calls: list[tuple[str, tuple[int, ...]]] = []

    class _FakeModel:
        def combine_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
            calls.append(("combine", tuple(x.shape)))
            return (x[:, :4] + x[:, 4:8] + x[:, 8:12]).contiguous()

    spec.model = _FakeModel()

    shard = torch.arange(3 * 12, dtype=torch.bfloat16).reshape(3, 12).to("mcpu")
    sharded_aux = SpShardedAuxHiddenStates(packed_shard=shard, full_num_tokens=5)

    class _FakeComm:
        comm_ptr_wrapper = object()

    fake_tp = SimpleNamespace(world_size=2, device_communicator=_FakeComm())
    monkeypatch.setattr(dspark_spec_mod, "get_tp_group", lambda: fake_tp)
    monkeypatch.setattr(
        dspark_spec_mod,
        "sp_all_gather",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("unexpected sp_all_gather")
        ),
    )

    def _fake_all_gather_out(dst: torch.Tensor, src: torch.Tensor, _comm, dim: int = 0):
        assert dim == 0
        calls.append(("all_gather_out", tuple(src.shape)))
        dst[: src.shape[0]].copy_(src)
        dst[src.shape[0] : 2 * src.shape[0]].copy_(src + 1)

    monkeypatch.setattr(xcpu_ops, "all_gather_into_tensor_out_v2", _fake_all_gather_out)

    out = spec._combine_target_hidden_states(
        torch.empty((5, 4), dtype=torch.bfloat16, device="mcpu"),
        sharded_aux,
    )
    assert calls == [("combine", (3, 12)), ("all_gather_out", (3, 4))]
    assert tuple(out.shape) == (5, 4)
    assert (
        out.untyped_storage().data_ptr()
        == spec._sp_gather_buf.untyped_storage().data_ptr()
    )

    # TP=1 skips collective completely and slices local projection.
    calls.clear()
    monkeypatch.setattr(
        dspark_spec_mod,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1, device_communicator=None),
    )
    out_tp1 = spec._combine_target_hidden_states(
        torch.empty((5, 4), dtype=torch.bfloat16, device="mcpu"),
        SpShardedAuxHiddenStates(packed_shard=shard, full_num_tokens=2),
    )
    assert calls == [("combine", (3, 12))]
    assert tuple(out_tp1.shape) == (2, 4)


def test_source_layers_defer_sp_gather_skips_collective(monkeypatch):
    aux_layers = [0, 1]
    called = []

    class _FakeLayer(torch.nn.Module):
        def __init__(self, idx: int) -> None:
            super().__init__()
            self.idx = idx

        def forward(
            self,
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            engram_hashes,
            engram_mask,
            capture_previous_aux=False,
            mega_gate_metadata=None,
        ):
            aux = (
                torch.full(
                    (hidden_states.shape[0], 4),
                    float(self.idx + 1),
                    dtype=torch.bfloat16,
                    device="mcpu",
                )
                if capture_previous_aux
                else None
            )
            return (
                hidden_states + 1,
                hidden_states,
                hidden_states,
                hidden_states,
                hidden_states,
                aux,
            )

    fake_model = SimpleNamespace(
        _defer_dspark_aux_sp_gather=True,
        source_layer_ids=range(0, 3),
        aux_hidden_state_layers={1, 2},
        layers=[_FakeLayer(i) for i in range(3)],
        _mega_gate_metadata=lambda _ids: None,
        source_aux_layer_ids=lambda: [1, 2],
    )
    mod = object.__new__(_XcpuDeepseekV41SourceLayers)
    torch.nn.Module.__init__(mod)
    mod.model = fake_model

    import vllm.models.deepseek_v41.nvidia.model as nv_model_mod

    monkeypatch.setattr(
        nv_model_mod,
        "sp_all_gather",
        lambda *_a, **_k: called.append("sp_all_gather"),
    )
    h = torch.zeros((2, 4), dtype=torch.bfloat16, device="mcpu")
    pos = torch.arange(2, dtype=torch.int64, device="mcpu")
    outs = _XcpuDeepseekV41SourceLayers.forward(
        mod, h, pos, pos, None, None, None, None, None, None
    )
    assert called == []
    assert len(outs) == 7
    assert outs[5].shape == (2, 4) and outs[6].shape == (2, 4)
    assert aux_layers == [0, 1]
