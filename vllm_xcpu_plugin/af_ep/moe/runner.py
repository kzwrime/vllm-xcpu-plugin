"""F-rank process lifecycle: bootstrap, load routed experts, serve, and stop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertServiceOptions:
    model: str
    revision: str | None
    max_num_batched_tokens: int
    max_model_passes: int = 0
    load_format: str = "auto"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_model_passes < 0:
            raise ValueError("max_model_passes must be non-negative")


def run_expert_service(options: ExpertServiceOptions) -> None:
    """Run the complete synchronous F-rank lifecycle."""
    mpi, mpi_world = _join_mpi_world()

    try:
        service = _load_expert_service(options, mpi_world)
    except Exception:
        import traceback

        traceback.print_exc()
        # A may already be waiting in initialization; stop this startup world.
        mpi.COMM_WORLD.Abort(1)
        raise

    _serve_model_passes(service, options.max_model_passes)
    import torch_xcpu

    torch_xcpu.ops.moe_af_v7_cleanup()
    mpi_world.global_world_comm.Barrier()
    mpi.Finalize()


def _load_expert_service(options: ExpertServiceOptions, mpi_world):
    import torch
    import torch_mcpu  # noqa: F401
    import torch_xcpu
    from vllm.config import (
        LoadConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.model_loader import get_model_loader

    from vllm_xcpu_plugin.distributed.mpi_world import ClusterType

    from ..common.session_v7 import AfV7Session
    from .model import RoutedExpertsModel
    from .service_v7 import ExpertServiceV7

    assert mpi_world.cluster_type == ClusterType.MOE
    session = AfV7Session(
        mpi_world,
        max_rows_per_attention_rank=options.max_num_batched_tokens,
    )

    torch.set_default_device("mcpu")
    torch.accelerator.set_device_index(torch.device("mcpu:0"))
    torch_xcpu.initialize_runtime()

    model_config = ModelConfig(
        model=options.model,
        revision=options.revision,
        dtype="bfloat16",
        quantization=None,
        enforce_eager=True,
        trust_remote_code=options.trust_remote_code,
    )
    load_config = LoadConfig(load_format=options.load_format)
    vllm_config = VllmConfig(model_config=model_config, load_config=load_config)
    model = RoutedExpertsModel(
        vllm_config,
        ep_size=session.ep_size,
        ep_rank=session.role_rank,
        max_num_tokens=options.max_num_batched_tokens,
        device="mcpu",
    )
    loader = get_model_loader(load_config)
    with set_current_vllm_config(vllm_config):
        loader.load_weights(model, model_config)
    if options.load_format == "dummy":
        model.process_dummy_weights_after_loading()
        weight_summary = "dummy weights"
    else:
        audit = model.weight_audit
        assert audit is not None
        weight_summary = (
            f"loaded={audit.loaded_count} "
            f"remote={len(audit.remote_checkpoint_names)}"
        )
    service = ExpertServiceV7(model, session)
    print(f"AF-EP F{session.role_rank} weights ready: {weight_summary}", flush=True)
    service.initialize()
    return service


def _join_mpi_world():
    from vllm_xcpu_plugin.distributed.mpi_world import (
        ClusterType,
        initialize_mpi,
        initialize_mpi_world,
    )

    mpi = initialize_mpi()
    # A-side run_mp_rpc_worker executes this barrier before importing vLLM.
    # F ranks must preserve the same MPMD collective order.
    mpi.COMM_WORLD.Barrier()
    return mpi, initialize_mpi_world(ClusterType.MOE)


def _serve_model_passes(service, max_model_passes: int) -> None:
    completed = 0
    while max_model_passes == 0 or completed < max_model_passes:
        service.execute_model_pass()
        completed += 1
