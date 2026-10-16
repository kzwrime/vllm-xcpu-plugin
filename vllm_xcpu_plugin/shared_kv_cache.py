"""Node-local replicated GLM cache backing for model runner V2."""

import dataclasses
import os
import socket

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)
_runtime = None


def enabled():
    return os.getenv("VLLM_XCPU_SHARED_KV_CACHE", "0") == "1"


def runtime():
    return _runtime


def validate_config(config):
    if not enabled():
        return
    p = config.parallel_config
    if (
        p.tensor_parallel_size not in (1, 2, 4)
        or p.pipeline_parallel_size != 1
        or p.decode_context_parallel_size != 1
        or not config.use_v2_model_runner
        or p.num_ubatches > 1
        or config.kv_transfer_config is not None
        or config.compilation_config.pass_config.enable_sp
        or os.getenv("VLLM_XCPU_ENABLE_AF_EP", "0") != "0"
    ):
        raise ValueError(
            "Shared KV requires V2, TP1/2/4, PP1/DCP1; disable SP/DBO/AF-EP/KV transfer"
        )
    if config.model_config.hf_text_config.model_type != "glm_moe_dsa":
        raise ValueError("Shared KV currently supports GLM5.2 only")
    if os.getenv("VLLM_XCPU_FUSED_INDEXER_K", "1") == "0":
        raise ValueError("Shared KV requires fused indexer K preprocessing")


def memory_budget(num_layers=0):
    size = int(os.environ["TORCH_MCPU_SHARED_SIZE"], 0)
    base = int(os.getenv("TORCH_MCPU_SHARED_BASE", "0"), 0)
    budget = (size - (-base % 256)) // 256 * 256
    if budget <= 0:
        raise ValueError("Shared KV arena is too small")
    return budget


@torch.library.custom_op("xcpu::shared_kv_publish", mutates_args=("cache",))
def publish(cache: torch.Tensor) -> None:
    # Keep device completion and cross-rank publication inside the compiled graph.
    runtime().fence()


@publish.register_fake
def _publish_fake(cache):
    pass


class SharedKVCache:
    def __init__(self, config, device, draft_layers=()):
        from vllm.distributed import get_pcp_group, get_tp_group

        self.group = get_tp_group()
        self.writer = self.group.rank_in_group == 0
        self.device = device
        self.dp_rank = config.parallel_config.data_parallel_rank
        self.pcp_rank = get_pcp_group().rank_in_group
        self.draft_layers = set(draft_layers)
        self.backing = []
        self.private_caches = []

    def agree(self, value):
        values = [None] * self.group.world_size
        dist.all_gather_object(values, value, group=self.group.cpu_group)
        if any(item != values[0] for item in values):
            raise ValueError(f"Shared KV TP configuration mismatch: {values}")

    def fence(self):
        torch.mcpu.synchronize(self.device)
        if self.group.world_size > 1:
            self.group.barrier()

    def allocate(self, config, device, layout, kernel_block_sizes=None):
        global _runtime
        from torch_mcpu.shared_memory import xmalloc
        from vllm.v1.kv_cache_interface import (
            MLAAttentionSpec,
            UniformTypeKVCacheSpecs,
            create_kv_cache_views,
        )

        if _runtime is not None:
            raise RuntimeError("Shared KV supports one engine per worker process")
        path = os.getenv("TORCH_MCPU_SHARED_SIM_FILE")
        if path:
            path = f"{path}.dp{self.dp_rank}.pcp{self.pcp_rank}"
            os.environ["TORCH_MCPU_SHARED_SIM_FILE"] = path
        self.agree((
            socket.gethostname(),
            path,
            memory_budget(),
            dataclasses.asdict(config),
            sorted(self.draft_layers),
        ))
        sizes = {t.size for t in config.kv_cache_tensors}
        if len(sizes) != 1:
            raise ValueError("Shared KV requires one device cache pool")
        original_size = sizes.pop()
        target_size = max(
            tensor.offset
            + (
                tensor.size
                if tensor.block_stride > tensor.layer_stride
                else tensor.layer_stride * len(tensor.layers)
            )
            for tensor in config.kv_cache_tensors
            if not any(layer in self.draft_layers for layer in tensor.layers)
        )
        size = target_size
        if original_size > memory_budget() or size > memory_budget():
            raise ValueError("KV cache exceeds shared arena capacity")
        buf = xmalloc(size, device.index).view(torch.int8)
        self.backing.append(buf)
        self.fence()
        probe = buf[:16]
        nonce = os.urandom(probe.numel()) if self.writer else None
        nonce = self.group.broadcast_object(nonce, src=0)
        if self.writer:
            probe.copy_(torch.tensor(list(nonce), dtype=torch.uint8).view(torch.int8))
        self.fence()
        observed = bytes(probe.cpu().view(torch.uint8).tolist())
        self.agree((nonce, observed))
        if observed != nonce:
            raise RuntimeError("Shared KV mapping did not expose writer data")
        if self.writer:
            buf.zero_()
        self.fence()
        result = {}
        for tensor in config.kv_cache_tensors:
            name = tensor.layers[0]
            group_id, group = next(
                (i, g)
                for i, g in enumerate(config.kv_cache_groups)
                if name in g.layer_names
            )
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            draft = name in self.draft_layers
            if any((n in self.draft_layers) != draft for n in tensor.layers):
                raise ValueError(
                    "Target and draft layers must have separate cache descriptors"
                )
            if not draft and not isinstance(spec, MLAAttentionSpec):
                raise ValueError("Only replicated MLA/indexer caches may be shared")
            block_size = kernel_block_sizes[group_id] if kernel_block_sizes else None
            if isinstance(spec, MLAAttentionSpec) and spec.storage_block_size:
                block_size = spec.storage_block_size
            raw = buf
            if draft:
                # Draft heads can be TP-sharded. Give them compact, private pages.
                page = spec.page_size_bytes
                stride = config.num_blocks * page
                tensor = dataclasses.replace(
                    tensor,
                    offset=0,
                    block_stride=page,
                    layer_stride=stride,
                    size=stride * len(tensor.layers),
                )
                raw = torch.zeros(tensor.size, dtype=torch.int8, device=device)
            views = create_kv_cache_views(
                raw,
                spec,
                config.num_blocks,
                layout,
                tensor,
                kernel_block_size=block_size,
            )
            result.update(zip(tensor.layers, views))
            if draft:
                self.private_caches.extend(views)
        _runtime = self
        logger.info(
            "Shared KV ready: dp=%d pcp=%d tp=%d physical_bytes=%d "
            "private_TP_bytes=%d file=%s",
            self.dp_rank,
            self.pcp_rank,
            self.group.world_size,
            size,
            size * self.group.world_size,
            path,
        )
        return result
