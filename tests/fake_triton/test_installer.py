# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path


def test_installer_exposes_importable_triton_modules():
    repo = Path(__file__).parents[2]
    code = """
import importlib.util
import json
from vllm_xcpu_plugin.fake_triton import install_fake_triton

first = install_fake_triton()
second = install_fake_triton()
import triton
import triton.backends.compiler
import triton.compiler.compiler
import triton.language as tl
from triton.backends import backends
from triton.compiler.compiler import make_backend
import triton.language.extra.libdevice as libdevice
from triton.runtime.driver import driver

backend = make_backend(driver.active.get_current_target())

print(json.dumps({
    "first": first.installed,
    "second": second.installed,
    "marker": triton.__xcpu_fake_triton__,
    "spec": importlib.util.find_spec("triton") is not None,
    "active": [x.driver.is_active() for x in backends.values()],
    "constexpr": repr(tl.constexpr),
    "dtype_is_type": isinstance(tl.dtype, type),
    "backend_hash": backend.hash(),
    "libdevice": libdevice.__name__,
    "compiler_shim": not hasattr(triton.backends.compiler, "AttrsDescriptor"),
}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "first": True,
        "second": False,
        "marker": True,
        "spec": True,
        "active": [True],
        "constexpr": "tl.constexpr",
        "dtype_is_type": True,
        "backend_hash": "xcpu-fake-triton-backend",
        "libdevice": "triton.language.extra.libdevice",
        "compiler_shim": True,
    }


def test_vllm_target_kernels_are_decorated_before_import():
    repo = Path(__file__).parents[2]
    code = """
import importlib
import json

from vllm.triton_utils import HAS_TRITON, triton
from vllm_xcpu_plugin.fake_triton.runtime import UnknownKernelError

targets = {
    "vllm.v1.worker.utils": ["_zero_kv_blocks_kernel"],
    "vllm.v1.worker.block_table": ["_compute_slot_mapping_kernel"],
    "vllm.v1.worker.gpu.buffer_utils": ["_apply_write_kernel"],
    "vllm.v1.worker.gpu.block_table": [
        "_gather_block_tables_kernel",
        "_compute_slot_mappings_kernel",
    ],
    "vllm.v1.worker.gpu.input_batch": [
        "_prepare_prefill_inputs_kernel",
        "_prepare_pos_seq_lens_kernel",
        "_combine_sampled_and_draft_tokens_kernel",
        "_get_num_sampled_and_rejected_kernel",
        "_post_update_kernel",
        "_post_update_num_computed_tokens_kernel",
        "_expand_idx_mapping_kernel",
    ],
    "vllm.v1.worker.gpu.structured_outputs": [
        "_apply_grammar_bitmask_kernel"
    ],
    "vllm.v1.worker.gpu.sample.bad_words": ["_bad_words_kernel"],
    "vllm.v1.worker.gpu.sample.logit_bias": ["_bias_kernel"],
    "vllm.v1.worker.gpu.sample.logprob": [
        "_topk_log_softmax_kernel",
        "_ranks_kernel",
        "_fill_logprob_token_ids_kernel",
    ],
    "vllm.v1.worker.gpu.sample.min_p": ["_min_p_kernel"],
    "vllm.v1.worker.gpu.sample.penalties": [
        "_penalties_kernel",
        "_bincount_kernel",
    ],
    "vllm.v1.worker.gpu.sample.prompt_logprob": [
        "_prompt_logprobs_token_ids_kernel"
    ],
}

decorated = []
first_kernel = None
for module_name, kernel_names in targets.items():
    module = importlib.import_module(module_name)
    for kernel_name in kernel_names:
        kernel = getattr(module, kernel_name)
        assert type(kernel).__name__ == "FakeJITFunction"
        decorated.append(kernel.qualname)
        first_kernel = first_kernel or kernel

failed_closed = False
try:
    first_kernel[(1,)]()
except UnknownKernelError:
    failed_closed = True

print(json.dumps({
    "has_triton": HAS_TRITON,
    "marker": triton.__xcpu_fake_triton__,
    "count": len(decorated),
    "failed_closed": failed_closed,
}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env["VLLM_PLUGINS"] = "xcpu_platform_plugin"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "has_triton": True,
        "marker": True,
        "count": 22,
        "failed_closed": True,
    }


def test_vllm_registry_is_version_locked_and_dispatches():
    repo = Path(__file__).parents[2]
    torch_mcpu_repo = repo.parent / "torch_mcpu"
    code = """
import json
import torch

from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.block_table import _compute_slot_mapping_kernel
from vllm.v1.worker.gpu.structured_outputs import _apply_grammar_bitmask_kernel
from vllm_xcpu_plugin.fake_triton.runtime import get_registry
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels

register_vllm_kernels()
register_vllm_kernels()
logits = torch.arange(35, dtype=torch.float32, device="mcpu").reshape(1, 35)
indices = torch.tensor([0], dtype=torch.int32, device="mcpu")
bitmask = torch.tensor([[-3, 5]], dtype=torch.int32, device="mcpu")
_apply_grammar_bitmask_kernel[(1, 1)](
    logits,
    logits.stride(0),
    indices,
    bitmask,
    bitmask.stride(0),
    35,
    BLOCK_SIZE=8192,
)
query_start = torch.tensor([0, 2], dtype=torch.int32, device="mcpu")
positions = torch.tensor([0, 5], dtype=torch.int64, device="mcpu")
block_table = torch.tensor([[10, 11]], dtype=torch.int32, device="mcpu")
slot_mapping = torch.full((4,), 999, dtype=torch.int64, device="mcpu")
_compute_slot_mapping_kernel[(2,)](
    2,
    4,
    query_start,
    positions,
    block_table,
    block_table.stride(0),
    4,
    slot_mapping,
    TOTAL_CP_WORLD_SIZE=1,
    TOTAL_CP_RANK=0,
    CP_KV_CACHE_INTERLEAVE_SIZE=1,
    PAD_ID=-1,
    BLOCK_SIZE=1024,
)
torch.mcpu.synchronize()
registry = get_registry()
print(json.dumps({
    "has_triton": HAS_TRITON,
    "registrations": len(registry.registrations()),
    "grammar_launches": registry.launch_counts()[
        "vllm.v1.worker.gpu.structured_outputs._apply_grammar_bitmask_kernel"
    ],
    "v1_slot_launches": registry.launch_counts()[
        "vllm.v1.worker.block_table._compute_slot_mapping_kernel"
    ],
    "v1_slots": slot_mapping.cpu().tolist(),
    "masked": [
        bool(torch.isneginf(logits[0, 1]).item()),
        bool(torch.isneginf(logits[0, 33]).item()),
    ],
}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(torch_mcpu_repo), str(repo)))
    env["VLLM_PLUGINS"] = "xcpu_platform_plugin"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "has_triton": True,
        "registrations": 22,
        "grammar_launches": 1,
        "v1_slot_launches": 1,
        "v1_slots": [40, 45, -1, -1],
        "masked": [True, True],
    }
