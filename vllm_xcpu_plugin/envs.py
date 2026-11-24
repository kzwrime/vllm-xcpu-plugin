import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    # Take the KV cache's device memory in init_device, while the device heap
    # is still empty, and hand it to the KV cache allocation after model
    # loading and profiling. See vllm_xcpu_plugin/kv_cache_reservation.py.
    "VLLM_XCPU_RESERVE_KV_CACHE": lambda: (
        os.getenv("VLLM_XCPU_RESERVE_KV_CACHE", "1") == "1"
    ),
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
    # Run MPI TP all-reduce in place on the input instead of into a new output.
    "VLLM_XCPU_ALLREDUCE_INPLACE": lambda: bool(
        int(os.getenv("VLLM_XCPU_ALLREDUCE_INPLACE", "0"))
    ),
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
    # Enable DeepSeek-V4.1 Engram disaggregation: the Engram tables live on
    # dedicated MPI ranks (python -m vllm_xcpu_plugin.engram_disagg.server)
    # that follow the attention ranks in the MPMD world.
    "VLLM_XCPU_ENABLE_ENGRAM_DISAGG": lambda: bool(
        int(os.getenv("VLLM_XCPU_ENABLE_ENGRAM_DISAGG", "0"))
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
    # Fusion switch (0/1) for the compiled GLM-5.2 MLA front half (q/kv
    # RMSNorm + q_b_proj + W_UK_T + RoPE + KV-cache write): 1 replaces it
    # with torch_xcpu's fused_mla_qnorm_qb_uk_rope_cache via the post-grad
    # pattern pass. The impl version inside the op (A0 unfused baseline,
    # A1..AN mega-kernels, default: the highest) is torch_xcpu's
    # TORCH_XCPU_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE_IMPL. Compiled graphs only.
    "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE": (
        lambda: bool(
            int(
                os.getenv(
                    "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_"
                    "QNORM_QB_UK_ROPE_CACHE",
                    "0",
                )
            )
        )
    ),
    # Same fusion switch for fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache
    # (input add-norm + fused_qkv_a + the MLA front half); runs before the
    # phase-1 pass above, whose work it removes on the layers it fuses.
    # Impl: TORCH_XCPU_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_IMPL
    # (default: the highest).
    "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_"
    "QNORM_QB_UK_ROPE_CACHE": (
        lambda: bool(
            int(
                os.getenv(
                    "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_"
                    "QNORM_QB_UK_ROPE_CACHE",
                    "0",
                )
            )
        )
    ),
    # INT8 [16, 16] UE8M0 variant of the phase-1 MLA front pass (0/1).
    # Emits torch_xcpu::fused_mla_qnorm_qb_uk_rope_cache_int8. Compiled graphs only.
    "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE_"
    "INT8": (
        lambda: bool(
            int(
                os.getenv(
                    "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_"
                    "QNORM_QB_UK_ROPE_CACHE_INT8",
                    "0",
                )
            )
        )
    ),
    # INT8 [16, 16] UE8M0 variant of the phase-2 MLA front pass (0/1).
    # Emits torch_xcpu::fused_mla_add_norm_qkva[_wk_wq]_qnorm_qb_uk_rope_cache_int8.
    "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_"
    "QNORM_QB_UK_ROPE_CACHE_INT8": (
        lambda: bool(
            int(
                os.getenv(
                    "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_"
                    "QNORM_QB_UK_ROPE_CACHE_INT8",
                    "0",
                )
            )
        )
    ),
    # Fusion switch (0/1) for the MoE front segment (post-attention
    # fused_add_rms_norm + FP32 router gate GEMM): 1 replaces eligible pairs
    # with torch_xcpu's fused_add_norm_router_gemm via the post-grad pattern
    # pass. Impl inside the op: TORCH_XCPU_FUSED_ADD_NORM_ROUTER_GEMM_IMPL
    # (A2 default mega-kernel with internal per-op fallback, A1 v1 mega-kernel,
    # A0 unfused). Compiled graphs only.
    "VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM": lambda: bool(
        int(
            os.getenv(
                "VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM",
                "0",
            )
        )
    ),
    # Debug: dump the post-grad FX graph to <value>.<pid>.txt before pattern
    # application (and again after, when patterns changed it).
    "VLLM_XCPU_DUMP_POST_GRAD_GRAPH": lambda: os.getenv(
        "VLLM_XCPU_DUMP_POST_GRAD_GRAPH", ""
    ),
    # Enable fused sequential Markov sampling in XcpuDSparkSpeculator (default 1).
    # Set to 0 (or false/off/no) to manually fall back to the unfused loop.
    "VLLM_XCPU_DSPARK_MARKOV_FUSED": lambda: (
        os.getenv("VLLM_XCPU_DSPARK_MARKOV_FUSED", "1").strip().lower()
        in ("1", "true", "on", "yes")
    ),
    # Precision mode for XcpuDSparkSpeculator fused Markov w2: bf16, fp8, int8.
    "VLLM_XCPU_DSPARK_MARKOV_DTYPE": lambda: (
        os.getenv("VLLM_XCPU_DSPARK_MARKOV_DTYPE", "bf16").strip().lower()
    ),
    # Enable fused XCPU rejection verification/resampling (default 1).
    # Set to 0 (or false/off/no) to fall back to the unfused rejection_sample() path.
    "VLLM_XCPU_REJECTION_SAMPLE_FUSED": lambda: (
        os.getenv("VLLM_XCPU_REJECTION_SAMPLE_FUSED", "1").strip().lower()
        in ("1", "true", "on", "yes")
    ),
    # Synchronous token-range diagnostic checks in spec-decode / rejection ops
    # (default 0).
    "VLLM_XCPU_CHECK_TOKEN_RANGE": lambda: (
        os.getenv("VLLM_XCPU_CHECK_TOKEN_RANGE", "0").strip().lower()
        in ("1", "true", "on", "yes")
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
