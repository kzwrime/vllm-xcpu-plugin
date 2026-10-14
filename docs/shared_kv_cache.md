# Shared KV cache (initial eager implementation)

Enable with `VLLM_XCPU_SHARED_KV_CACHE=1`. This integrates the persistent
`torch_mcpu.shared_memory.xmalloc` arena with the V1 MCPU model runner for
`GlmMoeDsa` (GLM5.2). Both main MLA and indexer caches are shared within TP2.

Initial supported configuration: TP2, PP=PCP=DCP=1, eager execution,
`VLLM_USE_V2_MODEL_RUNNER=0`, and `--no-async-scheduling`. Speculation, KV
transfer, sequence-parallel compilation, microbatching/DBO, and AF-EP are
rejected. Ordinary DP/EP remains separate from cache ownership: each DP
instance gets its own TP cache pool. Select the default attention backend and
keep `VLLM_XCPU_FUSED_INDEXER_K=1` (the default).

## Memory setup

```sh
export VLLM_XCPU_SHARED_KV_CACHE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export TORCH_MCPU_SHARED_SIZE=1073741824
export TORCH_MCPU_SHARED_SIM_FILE=/dev/shm/unique-run.kv
```

The simulator appends `.dp0`, `.dp1`, etc. to the file path. Both TP ranks must
be on the same host and share the file namespace. Use a fresh run-specific
path; remove the files only after all workers have exited. The allocator does
not reclaim individual allocations. One engine per worker process is supported.

For a pre-mapped device build, configure `TORCH_MCPU_SHARED_BASE` and
`TORCH_MCPU_SHARED_SIZE` for each worker instead; the launcher must supply a
different physical region to each DP group. The current fence implementation
uses device synchronization followed by a TP host barrier. This is verified on
coherent x86 memory; non-coherent hardware needs publication/invalidation
semantics before enabling the feature on that device.

Cache capacity comes from the shared region, with an alignment allowance.
An explicit `--kv-cache-memory-bytes` can cap it further. The old
`VLLM_CPU_KVCACHE_SPACE` setting is ignored when sharing is enabled. Startup
checks identical TP cache configuration, aligned total allocation size, and
actual cross-process visibility with a nonce probe before clearing the cache.
Logs report physical cache bytes once per TP group versus the corresponding
two-copy total (do not add together the two identical rank log messages).

## Execution

Scheduler block allocation and per-worker block tables remain unchanged.
The plugin preserves packed backings and layer aliases from `KVCacheConfig`.
Only TP rank 0 initializes, zeros, copies, and updates shared cache data.

MLA uses the existing separate Q/RoPE computation and cache-insertion path;
cache-writing MLA fusions are disabled for both ranks. Indexer preprocessing
keeps Q computation on both ranks while only rank 0 inserts K. After each
cache update, both ranks finish device work and enter a TP barrier before
reading. Execution completion and state-update fences drain all readers before
blocks can be reused. This intentionally prioritizes correctness over latency.

## Validation commands

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/.venv/bin/python -m pytest \
  /workspace/vllm-xcpu-plugin/tests/plugin_tests/test_shared_kv_cache.py \
  /workspace/vllm-xcpu-plugin/tests/layers/test_mla_rope_kvcache_routing.py -q
```

The two-pair test uses real MCPU tensors and MLA/indexer kernels. Follower KV
inputs deliberately differ to catch accidental duplicate writes. The kit's
`docs/shared-kv-cache/model-smoke.py` runs reduced GLM dummy-weight generation
for shared/private output comparison; it is a functional check, not a quality
evaluation of the full pretrained model.

The kit validation also covers DP2/TP2/EP4 with `mpi_alltoallv_v6`, an external
MPI launcher, and separate `.dp0`/`.dp1` pools. See the kit's
`docs/shared-kv-cache/validation.md` for results and reproducible commands.

Full 78-layer real FP8 and MXFP4 checkpoints have also passed short-request
DP2/TP2/EP4 generation on one x86 host: shared/private tokens and logprobs match
exactly, with half the unique cache backing bytes. See the kit's
`docs/shared-kv-cache/full-validation.md`. This does not validate two physical
nodes, long contexts, performance, or non-coherent device memory.
