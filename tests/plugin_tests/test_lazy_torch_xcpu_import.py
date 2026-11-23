# SPDX-License-Identifier: Apache-2.0
"""Ensure non-worker processes (API server, EngineCore) never import torch_xcpu."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def _run_in_clean_process(
    code: str, env_overrides: dict[str, str] | None = None
) -> str:
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Subprocess exited with {result.returncode}:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


def test_plugin_registration_and_config_never_import_torch_xcpu():
    """Simulate x86 head-only API server / EngineCore where importing torch_xcpu
    raises OSError (or would prematurely initialize an NPU)."""
    out = _run_in_clean_process(
        """
        import sys
        from importlib.abc import MetaPathFinder

        class DisallowTorchXcpu(MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "torch_xcpu" or fullname.startswith("torch_xcpu."):
                    raise OSError(f"unexpected import of {fullname} outside worker")
                return None

        sys.meta_path.insert(0, DisallowTorchXcpu())

        from vllm.platforms import current_platform
        from vllm.plugins import load_general_plugins

        assert current_platform.device_name == "mcpu"
        current_platform.pre_register_and_update()
        current_platform.import_ir_kernels()
        assert current_platform.is_sleep_mode_available() is True

        load_general_plugins()

        # Verify lazy RMSNorm trampoline and GDN patch installed without
        # importing torch_xcpu, and DSpark speculator import did not pull in
        # deepseek_v41.model eagerly.
        from vllm_xcpu_plugin import gdn_patch, rms_norm_dispatch
        assert rms_norm_dispatch._LAZY_LIBRARY is not None
        assert rms_norm_dispatch._REGISTERED is False
        assert gdn_patch._GDN_ATTENTION_PATCHED is True
        assert "vllm_xcpu_plugin.models.dspark.speculator" in sys.modules
        assert "vllm_xcpu_plugin.models.deepseek_v41.model" not in sys.modules

        # ModelRegistry inspection imports DeepSeek-V4.1 model module in
        # non-worker processes; that import must also stay free of torch_xcpu.
        from vllm_xcpu_plugin.models.deepseek_v41 import model as dsv41_model
        assert dsv41_model._INSTALLED is True

        assert "torch_xcpu" not in sys.modules
        print("OK")
        """
    )
    assert "OK" in out


def test_lazy_rms_norm_upgrades_to_native_kernel_on_first_call():
    """Standalone callers using load_general_plugins() without McpuWorker still
    upgrade PrivateUse1 rms_norm to the native C++ kernel on first execution."""
    out = _run_in_clean_process(
        """
        import sys
        import torch
        from vllm.platforms import current_platform  # noqa: F401
        from vllm.plugins import load_general_plugins

        load_general_plugins()
        assert "torch_xcpu" not in sys.modules

        x = torch.randn(4, 16, dtype=torch.bfloat16, device="mcpu")
        w = torch.randn(16, dtype=torch.bfloat16, device="mcpu")
        y = torch.empty_like(x)
        torch.ops._C.rms_norm(y, x, w, 1e-6)

        assert "torch_xcpu" in sys.modules
        dump = "\\n".join(
            line for line in torch._C._dispatch_dump("_C::rms_norm").splitlines()
            if line.startswith("PrivateUse1")
        )
        assert "vllm_rms_norm.cpp" in dump, dump
        print("OK")
        """,
        env_overrides={"TORCH_XCPU_DUMP_SHAPES": "0"},
    )
    assert "OK" in out
