import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    "VLLM_XCPU_SHARED_KV_CACHE": lambda: (
        os.getenv("VLLM_XCPU_SHARED_KV_CACHE", "0") == "1"
    ),
    # Contiguous TP ranks per physically shared KV pool; unset means the whole
    # TP group. Must divide the TP size.
    "VLLM_XCPU_SHARED_KV_GROUP_SIZE": lambda: (
        int(os.environ["VLLM_XCPU_SHARED_KV_GROUP_SIZE"])
        if os.getenv("VLLM_XCPU_SHARED_KV_GROUP_SIZE")
        else None
    ),
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # If set, vllm-xcpu will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    "VLLM_CPU_USE_MPI": lambda: bool(int(os.getenv("VLLM_CPU_USE_MPI", "0"))),
    # Complete post-routing V5/V6 EP operation; set 0 for the legacy MK path.
    "VLLM_XCPU_ENABLE_FUSED_EP_MOE": lambda: bool(
        int(os.getenv("VLLM_XCPU_ENABLE_FUSED_EP_MOE", "1"))
    ),
    # Move replicated shared MLPs into the fused V5 EP op (round 0, under the
    # dispatch); TORCH_XCPU_FUSED_EP_MOE_V5_ASYNC_DISPATCH=1 overlaps them.
    # Layers on other MoE backends (only V5 fuses) keep the runner's path.
    "VLLM_XCPU_FUSE_SHARED_EXPERTS": lambda: bool(
        int(os.getenv("VLLM_XCPU_FUSE_SHARED_EXPERTS", "1"))
    ),
    # V5/V6 fused EP quota per source/destination pair, in post-SP token rows.
    # Zero disables chunking. This is independent of the V4 total receive cap.
    "VLLM_XCPU_MOE_EP_MAX_RECV_TOKENS_PER_RANK": lambda: int(
        os.getenv("VLLM_XCPU_MOE_EP_MAX_RECV_TOKENS_PER_RANK", "1024")
    ),
    # Enable XCPU AF-EP. Topology is derived from vLLM and the MPMD MPI world.
    "VLLM_XCPU_ENABLE_AF_EP": lambda: bool(
        int(os.getenv("VLLM_XCPU_ENABLE_AF_EP", "0"))
    ),
    # Optional global diagnostic barrier at every full A/F model pass entry.
    "VLLM_XCPU_AF_FORWARD_ALLREDUCE": lambda: bool(
        int(os.getenv("VLLM_XCPU_AF_FORWARD_ALLREDUCE", "0"))
    ),
    # Maximum number of hidden-state tokens that mpi_alltoallv v4 prepare may
    # receive on each rank. A non-positive value keeps the legacy derived size.
    "VLLM_XCPU_MOE_MAX_RECV_TOKENS": lambda: int(
        os.getenv("VLLM_XCPU_MOE_MAX_RECV_TOKENS", "0")
    ),
    # Force Quark checkpoints with dynamic MXFP4 activations to keep their
    # packed MXFP4 weights but execute with BF16 activations (W4A16).
    "VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16": lambda: bool(
        int(os.getenv("VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16", "0"))
    ),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
