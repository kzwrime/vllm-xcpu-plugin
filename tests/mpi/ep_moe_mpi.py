"""Full EP operation vs separate operators and an independent dense reference.

Run with mpirun -np 4 .venv/bin/python <this file>. Set XCPU_EP_COMPILE=1
and optionally TORCHINDUCTOR_CPP_WRAPPER=1 to exercise fullgraph compilation.
XCPU_EP_MAX_ROWS selects comma-separated capacities (default: 17,128,129).
Use 8 ranks to cover EP > topk, and XCPU_EP_MAX_ROWS=1024 for large capacity.
"""

# MPI must be initialized before importing the backend.
# ruff: noqa: E402

import os
import tempfile
from dataclasses import replace

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

MPI.Init()
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", tempfile.mkdtemp(prefix="ep_moe_"))

import torch
import torch.nn.functional as F
import torch_mcpu  # noqa: F401
from torch_xcpu import ops
from torch_xcpu.ops_defs.moe_prepare import moe_prepare_dispatch_buffer_bytes

from vllm_xcpu_plugin.layers.fused_moe.ep_experts import XcpuEPExperts


def direct_aoti_call(execution):
    """Exercise the generated C ABI even on PyTorch without direct dispatch."""
    import ctypes

    import torch_xcpu._C

    library = ctypes.CDLL(torch_xcpu._C.__file__)
    schema = execution._op._schema
    call = getattr(library, f"aoti_torch_mcpu_{schema.name.split('::')[1]}")
    call.argtypes = [
        ctypes.c_int64 if str(arg.type) == "int" else ctypes.c_void_p
        for arg in schema.arguments
    ] + [ctypes.c_void_p]
    call.restype = ctypes.c_int32
    pointer = ctypes.pythonapi.PyCapsule_GetPointer
    pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    pointer.restype = ctypes.c_void_p

    def invoke(*args):
        handles, converted = [], []
        try:
            for index, value in enumerate(args):
                if isinstance(value, torch.Tensor):
                    handle = torch._C._aoti.unsafe_alloc_void_ptr_from_tensor(value)
                    handles.append(handle)
                    ptr = ctypes.c_void_p(pointer(handle, None))
                    converted.append(ctypes.pointer(ptr) if 6 <= index <= 9 else ptr)
                else:
                    converted.append(value)
            assert call(*converted, None) == 0
        finally:
            for handle in handles:
                torch._C._aoti.alloc_tensor_by_stealing_from_void_ptr(handle)

    return invoke


def separate_stages(execution, x, weights, ids, baseline_compute):
    cfg, params = execution, execution.compute.params
    capacity, hidden = cfg.ep_size * cfg.max_num_tokens, params.hidden
    device, dtype = x.device, x.dtype
    recv = torch.empty(capacity, hidden, device=device, dtype=dtype)
    ri = torch.empty(capacity, cfg.topk, device=device, dtype=torch.int32)
    rw = torch.empty(capacity, cfg.topk, device=device, dtype=torch.float32)
    valid = torch.empty(1, device=device, dtype=torch.int32)
    counts = torch.empty(cfg.ep_size, device=device, dtype=torch.int32)
    offsets = torch.empty_like(counts)
    sends = torch.empty_like(counts)
    indices = torch.empty_like(ids)
    record_bytes = 8 + cfg.topk * 8 + hidden * x.element_size()
    send = torch.empty(
        moe_prepare_dispatch_buffer_bytes(
            cfg.ep_size,
            cfg.max_num_tokens,
            cfg.topk,
            record_bytes,
            send_empty_header=cfg.version == 6,
        ),
        device=device,
        dtype=torch.uint8,
    )
    getattr(ops, f"moe_prepare_fused_v{cfg.version}")(
        indices,
        recv,
        ri,
        rw,
        valid,
        counts,
        offsets,
        sends,
        send,
        x,
        ids,
        weights,
        execution.global_num_experts,
        params.experts,
        execution.comm_metadata,
        execution.comm_ptr_wrapper,
    )
    expert_output = torch.empty_like(recv)
    routes = capacity * cfg.topk
    ops.fused_moe_compute(
        output=expert_output,
        hidden_states=recv,
        backend=baseline_compute,
        topk_weights=rw,
        topk_ids=ri,
        activation="silu",
        global_num_experts=execution.global_num_experts,
        expert_map=execution.expert_map,
        expert_num_tokens=torch.empty(0, device=device, dtype=torch.int32),
        num_input_rows_valid=valid,
        topk_reduce=True,
        permuted_hidden_states=torch.empty(routes, hidden, device=device, dtype=dtype),
        sorted_by_expert=torch.empty(routes, device=device, dtype=torch.int32),
        sorted_by_expert_back=torch.empty(routes, device=device, dtype=torch.int32),
        expert_offsets=torch.empty(
            params.experts + 1, device=device, dtype=torch.int32
        ),
        intermediate_output=torch.empty(
            routes, 2 * params.intermediate, device=device, dtype=dtype
        ),
        activated=torch.empty(routes, params.intermediate, device=device, dtype=dtype),
        workspace_unpermute_and_reduce=torch.empty_like(recv, dtype=torch.float32),
    )
    output = torch.empty_like(x)
    tail = (
        execution.comm_metadata,
        execution.comm_ptr_wrapper,
        torch.empty_like(output, dtype=torch.float32),
        cfg.max_num_tokens,
    )
    if cfg.version == 5:
        ops.moe_finalize_v5(
            output, expert_output, indices, counts, offsets, sends, *tail
        )
    else:
        ops.moe_finalize_v6(output, expert_output, indices, counts, offsets, *tail)
    return output


def reference(x, weights, ids, w1, w2):
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w1.size(0)):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        gate, up = (x[rows].float() @ w1[expert].float().T).chunk(2, dim=-1)
        values = (F.silu(gate) * up) @ w2[expert].float().T
        result.index_add_(0, rows, values * weights[rows, slots, None])
    return result.to(x.dtype)


def run(
    comm,
    version,
    topk,
    dtype,
    weight_format="plain",
    backend=None,
    shape=(128, 128, 17),
    local_experts=4,
):
    torch._dynamo.reset()
    rank, peers = comm.Get_rank(), comm.Get_size()
    hidden, intermediate, bound = shape
    experts = peers * local_experts
    torch.manual_seed(101)
    w1 = torch.randn(experts, 2 * intermediate, hidden, dtype=dtype) * 0.025
    w2 = torch.randn(experts, hidden, intermediate, dtype=dtype) * 0.025
    local = slice(rank * local_experts, (rank + 1) * local_experts)
    scales = ()
    block = None
    if weight_format == "fp8":
        q1, q2 = (
            (w1 / 0.025).to(torch.float8_e4m3fn),
            (w2 / 0.025).to(torch.float8_e4m3fn),
        )
        s1 = torch.full((experts, 2, 1), 0.025)
        s2 = torch.full((experts, 1, 1), 0.025)
        w1, w2 = (q1.float() * 0.025).bfloat16(), (q2.float() * 0.025).bfloat16()
        scales = (s1[local].to("mcpu"), s2[local].to("mcpu"))
        block = (128, 128)
    elif weight_format == "mxfp4":
        q1 = torch.randint(
            256, (experts, 2 * intermediate, hidden // 2), dtype=torch.uint8
        )
        q2 = torch.randint(256, (experts, hidden, intermediate // 2), dtype=torch.uint8)
        s1 = torch.full(
            (experts, 2 * intermediate, hidden // 32), 120, dtype=torch.uint8
        )
        s2 = torch.full((experts, hidden, intermediate // 32), 120, dtype=torch.uint8)
        lut = torch.tensor([
            0,
            0.5,
            1,
            1.5,
            2,
            3,
            4,
            6,
            0,
            -0.5,
            -1,
            -1.5,
            -2,
            -3,
            -4,
            -6,
        ])

        def dequant(q):
            nibbles = torch.stack((q & 15, q >> 4), dim=-1).flatten(-2)
            return (lut[nibbles.long()] / 128).bfloat16()

        w1, w2 = dequant(q1), dequant(q2)
        scales = (s1[local].to("mcpu"), s2[local].to("mcpu"))
        block = (1, 32)
    else:
        q1, q2 = w1, w2
    compute = ops.initialize_fused_moe(
        q1[local].to("mcpu"),
        q2[local].to("mcpu"),
        *scales,
        scale_block_size=block,
        backend=backend
        if backend is not None
        else (
            ops.MoeGroupedGemmBackend.PORTABLE
            if dtype == torch.bfloat16
            else ops.MoeGroupedGemmBackend.ACC
        ),
        m_capacity=peers * bound * topk,
        allocate_scratch=False,
    )

    # Independent baseline owns its own backend scratch; the full EP entry must
    # work with empty GEMM scratch handles because its byte workspace owns it.
    def baseline_gemm(gemm):
        p = gemm.params
        block_n, block_k = p.scale_block_size or (0, 0)
        size = gemm._prepare_op(
            p.experts,
            peers * bound * topk,
            p.out_features,
            p.in_features,
            block_n,
            block_k,
            p.bias is not None,
        )
        assert p.scratch.numel() == 0
        return type(gemm)(
            replace(p, scratch=torch.empty(size, dtype=torch.int8, device="mcpu"))
        )

    baseline_compute = type(compute)(
        replace(
            compute.params,
            gemm1=baseline_gemm(compute.params.gemm1),
            gemm2=baseline_gemm(compute.params.gemm2),
        )
    )
    expert_map = torch.arange(experts, dtype=torch.int32) - rank * local_experts
    expert_map[(expert_map < 0) | (expert_map >= local_experts)] = -1
    execution = XcpuEPExperts(
        compute,
        torch.tensor([peers, rank, 0, 1, 0, 1], dtype=torch.int64),
        torch.tensor([comm.py2f()], dtype=torch.int64),
        expert_map.to("mcpu"),
        version=version,
        ep_size=peers,
        ep_rank=rank,
        max_num_tokens=bound,
        topk=topk,
    )
    forward = execution.forward
    if os.getenv("XCPU_EP_AOTI", "0") == "1":
        assert os.getenv("XCPU_EP_COMPILE", "0") != "1"
        execution._op = direct_aoti_call(execution)
    guards = []
    if os.getenv("XCPU_EP_COMPILE", "0") != "1":
        native = execution._op

        def guarded(*args):
            storage = torch.full(
                (execution.workspace_bytes + 128,),
                0xA5,
                device="mcpu",
                dtype=torch.uint8,
            )
            native(*args[:-1], storage[64:-64])
            guards.append(storage)

        execution._op = guarded
    if os.getenv("XCPU_EP_COMPILE", "0") == "1":
        forward = torch.compile(forward, fullgraph=True, dynamic=True)

    # Rank 0 can have no tokens while still owning heavily requested experts.
    pending = []
    for case in range(6):
        rows = (
            0
            if (case == 1 and rank == 0) or case == 4
            else (bound if case == 2 else bound - rank)
        )
        torch.manual_seed(301 + rank + case * 10)
        x = torch.randn(rows, hidden, dtype=dtype)
        ids = torch.full((rows, topk), -1, dtype=torch.int32)
        selected = min(experts, topk)
        ids[:, :selected] = torch.rand(rows, experts).argsort(dim=1)[:, :selected].int()
        if case == 2:
            # Every sender fills the complete capacity and targets all local
            # experts on rank 0: exactly C*min(topk, local_experts) compute rows.
            ids.fill_(-1)
            selected = min(topk, local_experts)
            ids[:, :selected] = torch.arange(selected, dtype=torch.int32)
        elif rows:
            ids[0] = -1
        weights = torch.softmax(torch.randn(rows, topk), dim=-1)
        xd, wd, ind = x.to("mcpu"), weights.to("mcpu"), ids.to("mcpu")
        dummy = case == 5 and rank == 0
        ops.set_dummy_run(dummy)
        baseline = separate_stages(execution, xd, wd, ind, baseline_compute)
        graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        new = forward(xd, wd, ind)
        if case == 3:
            assert (
                torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs_before
            )
        ops.set_dummy_run(False)
        # Keep multiple asynchronous invocations in flight and verify their
        # outputs only afterwards; workspace reuse must not corrupt old outputs.
        expected = torch.zeros_like(x) if dummy else reference(x, weights, ids, w1, w2)
        pending.append((new, baseline, expected))
    for new, baseline, expected in pending:
        torch.testing.assert_close(new.cpu(), baseline.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(new.cpu(), expected, rtol=0.03, atol=0.003)
    for storage in guards:
        assert torch.all(storage[:64].cpu() == 0xA5)
        assert torch.all(storage[-64:].cpu() == 0xA5)
    torch.accelerator.synchronize()
    getattr(ops, f"moe_prepare_fused_v{version}_cleanup")()
    if rank == 0:
        print(
            f"PASS version={version} topk={topk} dtype={dtype} "
            f"format={weight_format} backend={compute.resolved_backend} "
            f"shape={shape} workspace_bytes={execution.workspace_bytes}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        for version in (5, 6):
            # Exercise both sides of the fixed-slot/dense capacity boundary.
            for bound in map(
                int, os.getenv("XCPU_EP_MAX_ROWS", "17,128,129").split(",")
            ):
                shape = (128, 128, bound)
                if os.getenv("XCPU_EP_QUANT", "0") == "1":
                    for fmt in ("fp8", "mxfp4"):
                        for backend in (
                            ops.MoeGroupedGemmBackend.ACC,
                            ops.MoeGroupedGemmBackend.INTEL_AMX,
                        ):
                            run(
                                MPI.COMM_WORLD,
                                version,
                                8,
                                torch.bfloat16,
                                fmt,
                                backend,
                                shape,
                            )
                else:
                    for topk in (6, 8):
                        for dtype in (torch.bfloat16, torch.float32):
                            run(MPI.COMM_WORLD, version, topk, dtype, shape=shape)
        MPI.Finalize()
    except BaseException:  # noqa: BLE001 -- abort peers on any rank's failure
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
