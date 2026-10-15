"""Conservative TP2 shared MLA cache: one writer, synchronous publication."""

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
        p.tensor_parallel_size != 2
        or p.pipeline_parallel_size != 1
        or p.decode_context_parallel_size != 1
        or p.prefill_context_parallel_size != 1
        or p.use_ubatching
        or not config.model_config.enforce_eager
        or config.use_v2_model_runner
        or config.scheduler_config.async_scheduling
        or config.speculative_config is not None
        or config.kv_transfer_config is not None
        or config.compilation_config.pass_config.enable_sp
        or os.getenv("VLLM_XCPU_ENABLE_AF_EP", "0") != "0"
    ):
        raise ValueError(
            "Shared KV requires TP2/PP1/PCP1/DCP1, eager V1 runner, "
            "--no-async-scheduling, no speculation, KV transfer, SP, DBO or AF-EP"
        )
    if config.model_config.hf_text_config.model_type != "glm_moe_dsa":
        raise ValueError("Shared KV currently supports GlmMoeDsa only")
    if config.attention_config.backend is not None:
        raise ValueError("Shared KV requires the default sparse MLA backend")
    if os.getenv("VLLM_XCPU_FUSED_INDEXER_K", "1") == "0":
        raise ValueError("Shared KV requires fused indexer K preprocessing")


def memory_budget(num_layers):
    size = int(os.environ["TORCH_MCPU_SHARED_SIZE"], 0)
    base = int(os.getenv("TORCH_MCPU_SHARED_BASE", "0"), 0)
    # Reserve worst-case alignment slack for each tensor (MLA + indexer).
    budget = size - (-base % 256) - 256 * (2 * num_layers + 1)
    if budget <= 0:
        raise ValueError("Shared KV arena is too small")
    return budget


class SharedKVCache:
    def __init__(self, config, device):
        from vllm.distributed import get_tp_group

        self.group = get_tp_group()
        self.writer = self.group.rank_in_group == 0
        self.device = device
        self.dp_rank = config.parallel_config.data_parallel_rank
        self.backing = []

    def agree(self, value):
        values = [None] * self.group.world_size
        dist.all_gather_object(values, value, group=self.group.cpu_group)
        if any(item != values[0] for item in values):
            raise ValueError(f"Shared KV TP configuration mismatch: {values}")

    def fence(self):
        # Complete local device work before the host collective. This publishes
        # writes on coherent x86 memory and also drains readers before reuse.
        torch.mcpu.synchronize(self.device)
        self.group.barrier()

    def allocate(self, cache_config):
        global _runtime
        from torch_mcpu.shared_memory import xmalloc
        from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs

        if _runtime is not None:
            raise RuntimeError(
                "Shared KV arena can only be initialized once per worker"
            )
        specs = []
        for group in cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            specs.extend(
                spec.kv_cache_specs.values()
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else [spec]
            )
        if any(not isinstance(spec, MLAAttentionSpec) for spec in specs):
            raise ValueError("Shared KV only supports replicated MLA/indexer cache")
        # File-backed simulation must use one distinct region per DP instance.
        path = os.getenv("TORCH_MCPU_SHARED_SIM_FILE")
        if path:
            path = f"{path}.dp{self.dp_rank}"
            os.environ["TORCH_MCPU_SHARED_SIM_FILE"] = path
        self.agree((
            socket.gethostname(),
            path,
            os.getenv("TORCH_MCPU_SHARED_SIZE"),
            -int(os.getenv("TORCH_MCPU_SHARED_BASE", "0"), 0) % 256,
            dataclasses.asdict(cache_config),
        ))
        tensors = cache_config.kv_cache_tensors
        if not tensors or any(t.size <= 0 for t in tensors):
            raise ValueError("Shared KV requires nonempty cache tensors")
        packed_sizes = {t.size for t in tensors if t.block_stride > 0}
        if len(packed_sizes) > 1:
            raise ValueError(
                "Shared packed cache tensors must have identical backing size"
            )
        required = sum(
            (t.size + 255) // 256 * 256 for t in tensors if t.block_stride == 0
        )
        required += sum((size + 255) // 256 * 256 for size in packed_sizes)
        capacity = int(os.environ["TORCH_MCPU_SHARED_SIZE"], 0)
        base = int(os.getenv("TORCH_MCPU_SHARED_BASE", "0"), 0)
        if required > capacity - (-base % 256):
            raise ValueError(f"Shared KV needs {required} bytes, arena has {capacity}")
        result = {}
        packed = None
        for spec in tensors:
            if spec.block_stride > 0 and packed is not None:
                tensor = packed
            else:
                tensor = xmalloc(spec.size, self.device.index).view(torch.int8)
                self.backing.append(tensor)
                if spec.block_stride > 0:
                    packed = tensor
            for name in spec.shared_by:
                result[name] = tensor
        self.fence()
        # Confirm actual aliasing, not merely equal layouts or file names.
        # The owner publishes a fresh nonce; every rank must read it back.
        probe = self.backing[0][:16]
        nonce = os.urandom(probe.numel()) if self.writer else None
        nonce = self.group.broadcast_object(nonce, src=0)
        if self.writer:
            probe.copy_(torch.tensor(list(nonce), dtype=torch.uint8).view(torch.int8))
        self.fence()
        observed = bytes(probe.cpu().view(torch.uint8).tolist())
        self.agree((nonce, observed))
        if observed != nonce:
            raise RuntimeError("Shared KV mapping probe did not see writer data")
        if self.writer:
            for tensor in self.backing:
                tensor.zero_()
        self.fence()
        _runtime = self
        logger.info(
            "Shared KV ready: dp=%d tp_rank=%d blocks=%d physical_bytes=%d "
            "private_TP2_bytes=%d file=%s",
            self.dp_rank,
            self.group.rank_in_group,
            cache_config.num_blocks,
            required,
            2 * required,
            path,
        )
        return result
