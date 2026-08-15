# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Register XCPU Prepare/Finalize implementations with vLLM."""

from typing import Any, ClassVar

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    register_moe_prepare_finalize_factory,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareFinalizeFactory,
)

from .mpi_alltoallv_prepare_finalize_v2 import MpiAlltoallvPrepareAndFinalizeV2
from .mpi_alltoallv_prepare_finalize_v3 import MpiAlltoallvPrepareAndFinalizeV3
from .mpi_alltoallv_prepare_finalize_v4 import MpiAlltoallvPrepareAndFinalizeV4
from .mpi_alltoallv_prepare_finalize_v5 import MpiAlltoallvPrepareAndFinalizeV5
from .mpi_alltoallv_prepare_finalize_v6 import MpiAlltoallvPrepareAndFinalizeV6
from .torch_all_to_all_single_prepare_finalize import (
    TorchAlltoallSinglePrepareAndFinalize,
)

logger = init_logger(__name__)


def _validate_custom_backend(
    moe: FusedMoEConfig,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    backend: str,
) -> None:
    parallel = moe.moe_parallel_config
    if parallel.enable_eplb or routing_tables is not None:
        raise NotImplementedError(
            f"XCPU MoE backend {backend!r} requires linear placement without EPLB"
        )
    if moe.num_experts % parallel.ep_size:
        raise NotImplementedError(
            f"XCPU MoE backend {backend!r} requires a uniform expert partition"
        )
    expected = moe.num_experts // parallel.ep_size
    if moe.num_local_experts != expected:
        raise ValueError(
            f"num_local_experts={moe.num_local_experts}, expected {expected}"
        )

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is not None and vllm_config.parallel_config.enable_dbo:
        raise NotImplementedError(
            f"XCPU MoE backend {backend!r} is synchronous and does not support DBO"
        )


@register_moe_prepare_finalize_factory
class TorchAllToAllSinglePrepareFinalizeFactory(MoEPrepareFinalizeFactory):
    backend_name = "torch_all_to_all_single"
    supports_sequence_parallel = True

    @classmethod
    def create(
        cls,
        *,
        moe: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig | None,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        allow_new_interface: bool,
        use_monolithic: bool,
        eep_stage: bool,
        all2all_manager: Any,
    ) -> FusedMoEPrepareAndFinalize:
        del quant_config, allow_new_interface
        if use_monolithic:
            raise NotImplementedError(
                "XCPU MoE backend 'torch_all_to_all_single' requires Modular MoE"
            )
        if eep_stage:
            raise NotImplementedError(
                "XCPU MoE backend 'torch_all_to_all_single' does not support EEP"
            )
        _validate_custom_backend(moe, routing_tables, cls.backend_name)

        ep_group = get_ep_group().device_group
        if ep_group is None:
            ep_group = all2all_manager.cpu_group
        prepare_finalize = TorchAlltoallSinglePrepareAndFinalize(
            ep_group=ep_group,
            num_local_experts=moe.num_local_experts,
            num_dispatchers=all2all_manager.world_size,
        )
        logger.info_once(
            "Using XCPU MoE backend=%s prepare_finalize=%s",
            cls.backend_name,
            type(prepare_finalize).__name__,
            scope="process",
        )
        return prepare_finalize


class _MpiAlltoallvPrepareFinalizeFactory(MoEPrepareFinalizeFactory):
    version: ClassVar[str]
    implementation: ClassVar[type[FusedMoEPrepareAndFinalize] | None]

    @classmethod
    def create(
        cls,
        *,
        moe: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig | None,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        allow_new_interface: bool,
        use_monolithic: bool,
        eep_stage: bool,
        all2all_manager: Any,
    ) -> FusedMoEPrepareAndFinalize:
        del quant_config, allow_new_interface
        if use_monolithic:
            raise NotImplementedError(
                f"XCPU MoE backend {cls.backend_name!r} requires Modular MoE"
            )
        if eep_stage:
            raise NotImplementedError(
                f"XCPU MoE backend {cls.backend_name!r} does not support EEP"
            )

        if cls.implementation is None:
            raise ValueError(
                f"MPI alltoallv {cls.version} is deprecated and no longer supported; "
                "use --all2all-backend mpi_alltoallv_v2, "
                "mpi_alltoallv_v3, mpi_alltoallv_v4, mpi_alltoallv_v5, "
                "or mpi_alltoallv_v6"
            )

        is_sequence_parallel = moe.moe_parallel_config.is_sequence_parallel
        if is_sequence_parallel and not cls.supports_sequence_parallel:
            raise ValueError(
                f"MPI alltoallv {cls.version} does not support sequence parallelism"
            )
        _validate_custom_backend(moe, routing_tables, cls.backend_name)

        ep_group = get_ep_group().device_group
        if ep_group is None:
            ep_group = all2all_manager.cpu_group
        kwargs: dict[str, Any] = {
            "max_num_tokens": moe.max_num_tokens,
            "ep_group": ep_group,
            "num_experts": moe.num_experts,
            "num_local_experts": moe.num_local_experts,
            "num_dispatchers": all2all_manager.world_size,
            "rank_expert_offset": all2all_manager.rank * moe.num_local_experts,
            "dp_rank": moe.dp_rank,
            "dp_size": moe.dp_size,
        }
        if cls.supports_sequence_parallel:
            kwargs["is_sequence_parallel"] = is_sequence_parallel
            kwargs["sp_size"] = moe.moe_parallel_config.sp_size
        prepare_finalize = cls.implementation(**kwargs)
        logger.info_once(
            "Using XCPU MoE backend=%s version=%s prepare_finalize=%s",
            cls.backend_name,
            cls.version,
            type(prepare_finalize).__name__,
            scope="process",
        )
        return prepare_finalize


@register_moe_prepare_finalize_factory
class MpiAlltoallvV1PrepareFinalizeFactory(_MpiAlltoallvPrepareFinalizeFactory):
    backend_name = "mpi_alltoallv_v1"
    version = "v1"
    implementation = None


@register_moe_prepare_finalize_factory
class MpiAlltoallvV2PrepareFinalizeFactory(_MpiAlltoallvPrepareFinalizeFactory):
    backend_name = "mpi_alltoallv_v2"
    version = "v2"
    implementation = MpiAlltoallvPrepareAndFinalizeV2


@register_moe_prepare_finalize_factory
class MpiAlltoallvV3PrepareFinalizeFactory(_MpiAlltoallvPrepareFinalizeFactory):
    backend_name = "mpi_alltoallv_v3"
    version = "v3"
    implementation = MpiAlltoallvPrepareAndFinalizeV3
    supports_sequence_parallel = True


@register_moe_prepare_finalize_factory
class MpiAlltoallvV4PrepareFinalizeFactory(_MpiAlltoallvPrepareFinalizeFactory):
    backend_name = "mpi_alltoallv_v4"
    version = "v4"
    implementation = MpiAlltoallvPrepareAndFinalizeV4
    supports_sequence_parallel = True


@register_moe_prepare_finalize_factory
class MpiAlltoallvV5PrepareFinalizeFactory(_MpiAlltoallvPrepareFinalizeFactory):
    backend_name = "mpi_alltoallv_v5"
    version = "v5"
    implementation = MpiAlltoallvPrepareAndFinalizeV5
    supports_sequence_parallel = True


@register_moe_prepare_finalize_factory
class MpiAlltoallvV6PrepareFinalizeFactory(_MpiAlltoallvPrepareFinalizeFactory):
    backend_name = "mpi_alltoallv_v6"
    version = "v6"
    implementation = MpiAlltoallvPrepareAndFinalizeV6
    supports_sequence_parallel = True


@register_moe_prepare_finalize_factory
class MpiAlltoallvLegacyPrepareFinalizeFactory(_MpiAlltoallvPrepareFinalizeFactory):
    backend_name = "mpi_alltoallv"
    version = "unversioned backend"
    implementation = None
