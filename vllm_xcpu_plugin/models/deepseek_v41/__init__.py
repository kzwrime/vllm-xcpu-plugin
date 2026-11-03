# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 on XCPU, following the CUDA SM100 kernel boundaries.

``register_deepseek_v41`` runs at plugin load and only routes the model
architectures to this package. Everything that touches vLLM internals (the
upstream contract check, the Fake Triton adapters, the kernel rebinds) runs
when the model module is imported (``model.install_deepseek_v41``), so a
contract drift fails the DeepSeek-V4.1 load itself with its drift report
instead of leaving the upstream CUDA model in place.
"""

_MODEL_MODULE = "vllm_xcpu_plugin.models.deepseek_v41.model"


def register_deepseek_v41() -> None:
    from vllm import ModelRegistry
    from vllm.platforms import current_platform

    if current_platform.device_name != "mcpu":
        return
    ModelRegistry.register_model(
        "DeepseekV41ForCausalLM", f"{_MODEL_MODULE}:XcpuDeepseekV41ForCausalLM"
    )
    ModelRegistry.register_model(
        "DSparkV41DraftModel", f"{_MODEL_MODULE}:XcpuDSparkDeepseekV41ForCausalLM"
    )
