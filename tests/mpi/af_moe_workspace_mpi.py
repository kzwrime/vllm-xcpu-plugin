"""Real A/F transactions, compact workspace, SP negotiation and dense reference.

mpirun -np 4 python <file>; XCPU_AF_COMPILE=1 compiles both roles, fullgraph.
XCPU_AF_QUANT=1 tests ACC/AMX FP8 and MXFP4. Each rank uses a fresh cache.
XCPU_AF_MAX_ROWS selects comma-separated capacities (default: 17,128,129).
Use 10 ranks and XCPU_AF_TOPK=6 to cover expert ranks > topk.
"""

# ruff: noqa: E402
import os
import tempfile
from types import SimpleNamespace

from mpi4py import MPI

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", tempfile.mkdtemp(prefix="af_workspace_")
)
import torch
import torch.nn.functional as nnf
import torch_mcpu  # noqa: F401
from torch_xcpu import ops

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiCluster, MpiWorld


def direct_aoti(op):
    """Use the generated C ABI; all optional tensor positions come from schema."""
    import ctypes

    import torch_xcpu._C

    library = ctypes.CDLL(torch_xcpu._C.__file__)
    schema = op._schema
    call = getattr(library, "aoti_torch_mcpu_" + schema.name.split("::")[1])
    call.argtypes = [
        ctypes.c_int64 if str(a.type) == "int" else ctypes.c_void_p
        for a in schema.arguments
    ] + [ctypes.c_void_p]
    call.restype = ctypes.c_int32
    capsule = ctypes.pythonapi.PyCapsule_GetPointer
    capsule.argtypes = [ctypes.py_object, ctypes.c_char_p]
    capsule.restype = ctypes.c_void_p

    def invoke(*args):
        handles, converted = [], []
        try:
            for value, spec in zip(args, schema.arguments):
                if isinstance(value, torch.Tensor):
                    handle = torch._C._aoti.unsafe_alloc_void_ptr_from_tensor(value)
                    handles.append(handle)
                    pointer = ctypes.c_void_p(capsule(handle, None))
                    converted.append(
                        ctypes.pointer(pointer)
                        if "Optional" in str(spec.type)
                        else pointer
                    )
                else:
                    converted.append(value)
            assert call(*converted, None) == 0
        finally:
            for handle in handles:
                torch._C._aoti.alloc_tensor_by_stealing_from_void_ptr(handle)

    return invoke


def run(
    version,
    num_a,
    sp,
    fmt="plain",
    backend=ops.MoeGroupedGemmBackend.PORTABLE,
    max_rows=17,
):
    import importlib

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    num_f, h, i, topk, local_e = (
        size - num_a,
        128,
        128,
        int(os.getenv("XCPU_AF_TOPK", "8")),
        4,
    )
    experts = num_f * local_e
    is_a = rank < num_a
    role = ClusterType.ATTN if is_a else ClusterType.MOE
    role_rank = rank if is_a else rank - num_a
    cluster_comm = comm.Split(int(role), role_rank)
    world = MpiWorld(
        cluster_instance_id=int(role),
        cluster_type=role,
        cluster_comm=cluster_comm,
        global_world_comm=comm,
        clusters={
            0: MpiCluster(0, ClusterType.ATTN, tuple(range(num_a))),
            1: MpiCluster(1, ClusterType.MOE, tuple(range(num_a, size))),
        },
    )
    session_cls = getattr(
        importlib.import_module(f"vllm_xcpu_plugin.af_ep.common.session_v{version}"),
        f"AfV{version}Session",
    )
    client_cls = getattr(
        importlib.import_module(f"vllm_xcpu_plugin.af_ep.attn.client_v{version}"),
        f"ExpertsClientV{version}",
    )
    service_cls = getattr(
        importlib.import_module(f"vllm_xcpu_plugin.af_ep.moe.service_v{version}"),
        f"ExpertServiceV{version}",
    )
    session = session_cls(world, max_rows_per_attention_rank=max_rows)
    torch.manual_seed(101)
    w1 = torch.randn(experts, 2 * i, h, dtype=torch.bfloat16) * 0.025
    w2 = torch.randn(experts, h, i, dtype=torch.bfloat16) * 0.025
    q1, q2 = w1, w2
    scales = ()
    block = None
    if fmt == "fp8":
        q1, q2 = (
            (w1 / 0.025).to(torch.float8_e4m3fn),
            (w2 / 0.025).to(torch.float8_e4m3fn),
        )
        s1, s2 = torch.full((experts, 2, 1), 0.025), torch.full((experts, 1, 1), 0.025)
        w1, w2 = (q1.float() * 0.025).bfloat16(), (q2.float() * 0.025).bfloat16()
        block = (128, 128)
    elif fmt == "mxfp4":
        q1 = torch.randint(256, (experts, 2 * i, h // 2), dtype=torch.uint8)
        q2 = torch.randint(256, (experts, h, i // 2), dtype=torch.uint8)
        s1 = torch.full((experts, 2 * i, h // 32), 120, dtype=torch.uint8)
        s2 = torch.full((experts, h, i // 32), 120, dtype=torch.uint8)
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
            return (
                lut[torch.stack((q & 15, q >> 4), dim=-1).flatten(-2).long()] / 128
            ).bfloat16()

        w1, w2 = dequant(q1), dequant(q2)
        block = (1, 32)
    compile_model = os.getenv("XCPU_AF_COMPILE") == "1"
    torch._dynamo.reset()
    if is_a:
        client = client_cls(session)
        client.register_layer_capacity(sp)
        client.initialize(h, topk, torch.bfloat16)
        # Allocate before tracing; only native workspace mutation enters the graph.
        client._ensure_workspace(
            torch.empty(0, h, device="mcpu", dtype=torch.bfloat16), topk
        )
        guard = torch.full(
            (client._workspace.numel() + 128,), 0xA5, dtype=torch.uint8, device="mcpu"
        )
        client._workspace = guard[64:-64]
        if os.getenv("XCPU_AF_AOTI") == "1":
            client._op = direct_aoti(client._op)
        execute = client.execute_layer
        if compile_model:
            execute = torch.compile(
                execute,
                backend="inductor",
                fullgraph=True,
                dynamic=True,
                options={
                    "epilogue_fusion": False,
                    "pattern_matcher": False,
                    "combo_kernels": False,
                },
            )
    else:
        local = slice(role_rank * local_e, (role_rank + 1) * local_e)
        if fmt != "plain":
            scales = (s1[local].to("mcpu"), s2[local].to("mcpu"))
        compute = ops.initialize_fused_moe(
            q1[local].to("mcpu"),
            q2[local].to("mcpu"),
            *scales,
            scale_block_size=block,
            backend=backend,
            m_capacity=1,
            allocate_scratch=False,
        )
        expert_map = torch.full((experts,), -1, dtype=torch.int32)
        expert_map[local] = torch.arange(local_e, dtype=torch.int32)
        model = SimpleNamespace(
            layer_indices=(1, 3, 12),
            hidden_size=h,
            intermediate_size=i,
            num_experts=experts,
            top_k=topk,
            hidden_act="silu",
            ep_size=num_f,
            ep_rank=role_rank,
            device=torch.device("mcpu"),
            dtype=torch.bfloat16,
            routed_experts={
                str(n): SimpleNamespace(expert_map=expert_map.to("mcpu"))
                for n in (1, 3, 12)
            },
            fused_moe_for_layer=lambda _: compute,
        )
        service = service_cls(model, session, compile_model=compile_model)
        service.initialize()
        if os.getenv("XCPU_AF_AOTI") == "1":
            service._op = direct_aoti(service._op)
        guard = torch.full(
            (service._workspace.numel() + 128,), 0xA5, dtype=torch.uint8, device="mcpu"
        )
        service._workspace = guard[64:-64]
    capacity = (max_rows + sp - 1) // sp
    assert session.max_rows_per_attention_rank == capacity
    pending = []
    for case in range(4):
        if is_a:
            rows = (
                capacity
                if case in (0, 1)
                else (0 if case == 2 or role_rank == 0 else 3)
            )
            torch.manual_seed(300 + case * 17 + role_rank)
            x = torch.randn(rows, h, dtype=torch.bfloat16)
            ids = torch.full((rows, topk), -1, dtype=torch.int32)
            chosen = min(experts, topk)
            ids[:, :chosen] = torch.rand(rows, experts).argsort(dim=1)[:, :chosen].int()
            if case == 1:  # every A targets all local experts on F0, reaching R exactly
                ids.fill_(-1)
                ids[:, :local_e] = torch.arange(local_e, dtype=torch.int32)
            weights = torch.softmax(torch.randn(rows, topk), dim=-1)
            ref = torch.zeros(rows, h)
            for e in range(experts):
                r, k = torch.where(ids == e)
                gate, up = (x[r].float() @ w1[e].float().T).chunk(2, dim=-1)
                ref.index_add_(
                    0,
                    r,
                    ((nnf.silu(gate) * up) @ w2[e].float().T) * weights[r, k, None],
                )
            for layer in (1, 3, 12):
                result = execute(
                    layer_idx=layer,
                    hidden_states=x.to("mcpu"),
                    topk_ids=ids.to("mcpu"),
                    topk_weights=weights.to("mcpu"),
                    num_experts=experts,
                    num_local_experts=local_e,
                )
                pending.append((result, ref.bfloat16()))
        else:
            service.execute_model_pass()
    for actual, expected in pending:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.003)
    assert guard[:64].cpu().eq(0xA5).all() and guard[-64:].cpu().eq(0xA5).all()
    torch.accelerator.synchronize()
    getattr(ops, f"moe_af_v{version}_cleanup")()
    cluster_comm.Free()
    if rank == 0:
        print(
            f"PASS V{version} A{num_a}/F{num_f} SP{sp} M{capacity} "
            f"{fmt} {backend} compile={compile_model}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        for version in (7, 8):
            for max_rows in map(
                int, os.getenv("XCPU_AF_MAX_ROWS", "17,128,129").split(",")
            ):
                if os.getenv("XCPU_AF_QUANT") == "1":
                    for fmt in ("fp8", "mxfp4"):
                        for backend in (
                            ops.MoeGroupedGemmBackend.ACC,
                            ops.MoeGroupedGemmBackend.INTEL_AMX,
                        ):
                            run(version, 2, 2, fmt, backend, max_rows=max_rows)
                elif os.getenv("XCPU_AF_COMPILE") == "1":
                    run(version, 2, 2, max_rows=max_rows)
                else:
                    for a, sp in ((2, 1), (2, 2), (1, 1), (3, 2)):
                        run(version, a, sp, max_rows=max_rows)
    except BaseException:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
