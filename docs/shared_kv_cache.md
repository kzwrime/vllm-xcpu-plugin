# Shared KV cache

Set `VLLM_XCPU_SHARED_KV_CACHE=1` to allocate one MLA/indexer cache per TP2
worker group through `torch_mcpu.shared_memory.xmalloc`. TP rank 0 writes; both
ranks read after a device synchronization and TP barrier. Each DP instance keeps
an independent pool.

Supported configuration: GLM5.2 (`GlmMoeDsa`), TP2, PP/PCP/DCP1, eager V1,
synchronous scheduling, default sparse MLA backend, and fused indexer K. The
feature rejects speculation, KV transfer, sequence parallelism, DBO, and AF-EP.

```sh
export VLLM_XCPU_SHARED_KV_CACHE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export TORCH_MCPU_SHARED_SIZE=1073741824
export TORCH_MCPU_SHARED_SIM_FILE=/dev/shm/unique-run.kv
```

The simulator appends `.dp0`, `.dp1`, etc. to the file path. Use a fresh path;
remove it only after all workers exit. For a pre-mapped device build, provide a
separate physical region for each DP group with `TORCH_MCPU_SHARED_BASE` and
`TORCH_MCPU_SHARED_SIZE`.

Scheduler block tables stay per worker. Shared allocation preserves packed
backings and layer aliases. Cache updates use the existing MLA/indexer paths;
only rank 0 performs cache writes, with fences around updates and block reuse.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/.venv/bin/python -m pytest \
  /workspace/vllm-xcpu-plugin/tests/plugin_tests/test_shared_kv_cache.py \
  /workspace/vllm-xcpu-plugin/tests/layers/test_mla_rope_kvcache_routing.py -q
```

The plugin tests cover separate TP pools, packed and unpacked caches, write
ownership, fences, and real MCPU MLA/indexer updates. The kit contains the
reduced and full-model generation checks.
