"""Node-local replicated GLM cache backing for model runner V2.

GLM MLA latent and indexer K caches are computed by replicated projections, so
every TP rank produces the same bytes for the same slot. All ranks write them
into the shared pool and read back only what they wrote themselves; concurrent
writes of identical bytes need no ordering, so the forward pass adds no
cross-rank synchronization. Only step-start zeroing and CoW block copies store
different bytes. Those need device-side ordering only: update_requests queues a
TP all-reduce after them, so no rank appends this step's tokens before every
rank has rewritten the blocks. The host waits only during startup.

VLLM_XCPU_SHARED_KV_GROUP_SIZE splits the TP group into domains of contiguous
ranks that map the same physical pool (default: the whole TP group). Fences
still span the whole TP group, which orders every domain.

The x86 simulation backs each domain's pool with a file named after
TORCH_MCPU_SHARED_SIM_FILE, or /dev/shm/xcpu_shared_kv.<job id> when neither it
nor TORCH_MCPU_SHARED_BASE (required by pre-mapped builds) is set. The owner
removes the file once every rank has mapped it, so nothing outlives the job.
"""

import dataclasses
import os
import secrets
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


def group_size(tp_size):
    value = os.getenv("VLLM_XCPU_SHARED_KV_GROUP_SIZE", "")
    size = int(value) if value else tp_size
    if size <= 0 or tp_size % size:
        raise ValueError(
            f"VLLM_XCPU_SHARED_KV_GROUP_SIZE={value} must divide TP size {tp_size}"
        )
    return size


def check_domains(records):
    """Check (domain, owner nonce or None, bytes read) gathered from TP ranks."""
    nonces = {domain: nonce for domain, nonce, _ in records if nonce is not None}
    if len(set(nonces.values())) != len(nonces) or any(
        seen != nonces.get(domain) for domain, _, seen in records
    ):
        raise RuntimeError(
            "VLLM_XCPU_SHARED_KV_GROUP_SIZE does not match the physical shared "
            f"mapping: (domain, written, read) per TP rank = {records}"
        )


def validate_config(config):
    if not enabled():
        return
    p = config.parallel_config
    if (
        p.pipeline_parallel_size != 1
        or p.prefill_context_parallel_size != 1
        or p.decode_context_parallel_size != 1
        or not config.use_v2_model_runner
        or p.num_ubatches > 1
        or config.kv_transfer_config is not None
        or config.compilation_config.pass_config.enable_sp
        or os.getenv("VLLM_XCPU_ENABLE_AF_EP", "0") != "0"
    ):
        raise ValueError(
            "Shared KV requires V2, PP1/PCP1/DCP1; disable SP/DBO/AF-EP/KV transfer"
        )
    if config.model_config.hf_text_config.model_type != "glm_moe_dsa":
        raise ValueError("Shared KV currently supports GLM5.2 only")
    group_size(p.tensor_parallel_size)


def memory_budget():
    size = int(os.environ["TORCH_MCPU_SHARED_SIZE"], 0)
    base = int(os.getenv("TORCH_MCPU_SHARED_BASE", "0"), 0)
    budget = (size - (-base % 256)) // 256 * 256
    if budget <= 0:
        raise ValueError("Shared KV arena is too small")
    return budget


class SharedKVCache:
    def __init__(self, config, device, draft_layers=()):
        from vllm.distributed import get_tp_group

        self.group = get_tp_group()
        self.group_size = group_size(self.group.world_size)
        self.domain, local_rank = divmod(self.group.rank_in_group, self.group_size)
        # The owner only initializes its domain's pool; every rank writes entries.
        self.owner = local_rank == 0
        self.device = device
        self.dp_rank = config.parallel_config.data_parallel_rank
        self.draft_layers = set(draft_layers)
        self.fence_token = torch.zeros(1, device=device)
        self.backing = []
        self.private_caches = []
        self.path = None
        # (st_dev, st_ino) of the simulation file, recorded before its removal.
        self.file_id = None

    def gather(self, value):
        values = [None] * self.group.world_size
        dist.all_gather_object(values, value, group=self.group.cpu_group)
        return values

    def agree(self, common, per_domain):
        # Every rank checks every domain, so all ranks reach the same verdict.
        values = self.gather((self.domain, common, per_domain))
        common_differs = any(item != values[0][1] for _, item, _ in values)
        pairs = {(domain, item) for domain, _, item in values}
        domain_differs = len(pairs) != len({domain for domain, _ in pairs})
        if common_differs or domain_differs:
            raise ValueError(f"Shared KV TP configuration mismatch: {values}")

    def probe(self, buf):
        """Fail if the configured domains differ from the physical mapping."""
        nonce = os.urandom(8) if self.owner else None
        if nonce is not None:
            buf[:8].copy_(torch.frombuffer(bytearray(nonce), dtype=torch.int8))
        self.fence()
        seen = bytes(buf[:8].cpu().numpy())
        # The gather also orders every read before the owner clears the pool.
        check_domains(self.gather((self.domain, nonce, seen)))

    def fence(self):
        """Wait on the host until every rank's queued work is done (startup)."""
        torch.mcpu.synchronize(self.device)
        if self.group.world_size > 1:
            self.group.barrier()

    def device_fence(self):
        """Order queued writes on every rank before later queued work on any rank.

        With MPI the all-reduce is queued on the current stream and completes only
        after every rank reaches it, so the host does not wait.
        """
        self.group.all_reduce(self.fence_token)

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
        # Gather unconditionally: a collective must not depend on per-rank env.
        job = self.gather(secrets.token_hex(8))[0]
        path = os.getenv("TORCH_MCPU_SHARED_SIM_FILE")
        if not path and not os.getenv("TORCH_MCPU_SHARED_BASE"):
            path = f"/dev/shm/xcpu_shared_kv.{job}"
        if path:
            path = f"{path}.dp{self.dp_rank}.tp{self.domain}"
            os.environ["TORCH_MCPU_SHARED_SIM_FILE"] = path
        self.agree(
            (
                self.group_size,
                memory_budget(),
                dataclasses.asdict(config),
                sorted(self.draft_layers),
            ),
            # Only ranks of one domain must share a host and a backing file.
            (socket.gethostname(), path),
        )
        layer_specs = {}
        for group_id, group in enumerate(config.kv_cache_groups):
            for name in group.layer_names:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                layer_specs[name] = group_id, spec
        # Descriptor.size covers the whole pool. Preserve placement and stop at
        # the end of the last target page, including packed-block strides.
        size = max(
            t.offset
            + (len(t.layers) - 1) * t.layer_stride
            + (config.num_blocks - 1) * t.block_stride
            + layer_specs[t.layers[0]][1].page_size_bytes
            for t in config.kv_cache_tensors
            if t.layers[0] not in self.draft_layers
        )
        buf = xmalloc(size, device.index).view(torch.int8)
        self.backing.append(buf)
        if path:
            self.path = path
            stat = os.stat(path)
            self.file_id = stat.st_dev, stat.st_ino
        self.probe(buf)
        if self.owner:
            if path:
                # The probe proved every rank of the domain mapped the file. The
                # mappings keep it alive, and a later crash leaves nothing behind.
                os.unlink(path)
            buf.zero_()
        self.fence()
        result: dict[str, torch.Tensor] = {}
        for tensor in config.kv_cache_tensors:
            name = tensor.layers[0]
            group_id, spec = layer_specs[name]
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
            "Shared KV ready: dp=%d tp=%d domain=%d/%d shared_bytes=%d file=%s",
            self.dp_rank,
            self.group.world_size,
            self.domain,
            self.group.world_size // self.group_size,
            size,
            path,
        )
        return result
