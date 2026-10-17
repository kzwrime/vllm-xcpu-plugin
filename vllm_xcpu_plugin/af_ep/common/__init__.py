"""AF-EP code shared by A/F roles; import transport versions explicitly."""

AF_EP_BACKENDS = ("mpi_alltoallv_v7", "mpi_alltoallv_v8")
# Used when AF-EP is enabled without an explicit --all2all-backend. V8 uses
# MPI collectives and needs no UCX OSC (RMA) setup, unlike V7.
DEFAULT_AF_EP_BACKEND = "mpi_alltoallv_v8"


def apply_default_af_ep_backend(parallel_config) -> bool:
    """Replace vLLM's default all2all backend with the AF-EP default.

    Only vLLM's own default is replaced; an explicit non-AF backend is left
    for the AF-EP bootstrap to reject. Returns whether the backend changed.
    """
    from vllm.config import ParallelConfig

    vllm_default = ParallelConfig.__dataclass_fields__["all2all_backend"].default
    if parallel_config.all2all_backend != vllm_default:
        return False
    parallel_config.all2all_backend = DEFAULT_AF_EP_BACKEND
    return True
