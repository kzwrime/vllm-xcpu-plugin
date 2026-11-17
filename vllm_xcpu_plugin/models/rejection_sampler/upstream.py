# SPDX-License-Identifier: Apache-2.0
"""Upstream contracts of the fused speculative rejection sampler patch."""

from vllm_xcpu_plugin.fake_triton.provenance import KernelSource
from vllm_xcpu_plugin.upstream_contracts import (
    Manifest,
    Patch,
    UpstreamRef,
    WatchedModule,
)

_V031 = KernelSource(
    tag="v0.31.0",
    local_commit="0670c36056d45530524f498f731bf3847a7d4ce0",
    main_commit="283d76d56a45d9bbead5ba650402fb0957abc30c",
)

MANIFEST = Manifest(
    category="rejection_sampler",
    packages=("vllm_xcpu_plugin.models.rejection_sampler",),
    watched=(
        WatchedModule(
            module="vllm.v1.worker.gpu.spec_decode.rejection_sampler",
            sha256="25759f98e8aace9a556bc4252c206f246535f0536ebc919f97e097f92a72066a",
            source=_V031,
        ),
    ),
    patches=(
        Patch(
            id="rejection_sampler.verify",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.rejection_sampler.patch:_xcpu_rejection_sampler_verify",
                "vllm_xcpu_plugin.models.rejection_sampler.patch:install_rejection_sampler",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.rejection_sampler:RejectionSampler._verify",
                    source_hash="3547f3f2603c167d8d49ab820241c732618c6ed6c12d036f08b6d0dc42219ca9",
                    signature_hash="2b9ad0b9c2e3b847b837e0b320f718f4a572c28cc866c0c459af32d34621914e",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.rejection_sampler:RejectionSampler._watermarking_kwargs",
                    source_hash="c3c2324cf042fe882dd45fa474237e4bb19ee7b40ad811b322e5c24faf8158c6",
                    signature_hash="36aa1b6f54a9dd341fb77459f2b5216fe1ec4ed40da8d2b8d7580c1d73013882",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils:rejection_sample",
                    source_hash="dfedd148a1dc1f02dabbafee112cd263f16ef8757f513c8d711f7b72cf923b83",
                    signature_hash="068a480d38be4093631a780bb07300f395db01254148810b947774a1a6d8df76",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.sample.sampler:Sampler.__init__",
                    source_hash="ceefa00d87c958e5ff72a1cf3bf855434be9af39921d1b8a988e5a926ea39b8e",
                    signature_hash="637e1374b9802f76fbf9bb6841c11bdea2c6a63135d117007ec1c83c3d83a257",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.sample.sampler:Sampler.apply_sampling_params",
                    source_hash="48ce29c7596212c4e6e8f8ab843f4ad2a01582a4db06bce1584304ed66125247",
                    signature_hash="de6f943e18da4521cbee5bbad3f39af4d5f0f9deb9e8c6b21d1a43459b27da33",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.sample.states:SamplingStates.add_request",
                    source_hash="4f8958f4eb90f2aa50421db5833473cc03d46fe279c1924bf0a82e4900e14b5f",
                    signature_hash="cb8002665d09111c5ec568517fec4ed49186bc66cc1e09380754af2d37d9a20a",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.sample.states:SamplingStates.apply_temperature",
                    source_hash="5094707af9f097eb759606de93899c43ad861675c4bb8f6c173a3672bf8fcecb",
                    signature_hash="5f32bcd93cb6d919aa4299e77a03c01ac525d4a24e84c9534dfecd39ba2c8596",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.sample.states:SamplingStates.apply_min_p",
                    source_hash="688911e719ed5b4c50524f73a8d3753d42d84ea89e724744b713cf37c00561ad",
                    signature_hash="5f32bcd93cb6d919aa4299e77a03c01ac525d4a24e84c9534dfecd39ba2c8596",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.sample.states:SamplingStates.apply_top_k_top_p",
                    source_hash="2b2714b776499a040feba5db1e457d8408a698f1d77fc1332fb3e2f1e2ea0b88",
                    signature_hash="5f32bcd93cb6d919aa4299e77a03c01ac525d4a24e84c9534dfecd39ba2c8596",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.sample.logit_bias:LogitBiasState.add_request",
                    source_hash="4bd811e1eed1b9a8a4618c27e3f2869304022df28812ca9e69805ecefea84e0f",
                    signature_hash="cb8002665d09111c5ec568517fec4ed49186bc66cc1e09380754af2d37d9a20a",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.sample.penalties:PenaltiesState.add_request",
                    source_hash="d53d107f217d677d353d381502561a6f23f878460e890f4a83b2bdac98428d30",
                    signature_hash="cb8002665d09111c5ec568517fec4ed49186bc66cc1e09380754af2d37d9a20a",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.sample.bad_words:BadWordsState.add_request",
                    source_hash="bf5af1dbe11de6841533ada0ddd05e2284e85edd01b7d2861be4b178bf063d16",
                    signature_hash="cb8002665d09111c5ec568517fec4ed49186bc66cc1e09380754af2d37d9a20a",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.sample.thinking_budget:ThinkingBudgetState.add_request",
                    source_hash="034ac8bc75bfd34cd4669032ad6019c81f5b058f28a614b0fdd55d4d4fcb22ab",
                    signature_hash="cb8002665d09111c5ec568517fec4ed49186bc66cc1e09380754af2d37d9a20a",
                ),
            ),
            contract=(
                "RejectionSampler._verify runs target logits processing and "
                "rejection_sample. When mcpu row-major BF16/FP16/FP32 logits, "
                "standard (non-block, non-synthetic) verification without "
                "watermarking are used, _xcpu_rejection_sampler_verify bypasses "
                "apply_sampling_params (avoiding FP32 logits copy and standalone "
                "temperature kernel) whenever needs_logits_processing is active "
                "only because of temperature (no min_p/top_k/top_p/penalties/"
                "logit_bias/bad_words/thinking_budget/custom processors/mask/"
                "processed logprobs) and routes directly to "
                "torch_xcpu.ops.vllm_rejection_sample_fused. Re-check: new "
                "built-in logits processors or sampling-state filters in "
                "Sampler.__init__ / apply_sampling_params, or _verify args."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/rejection_sampler.py::vllm_rejection_sample_fused",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/test_rejection_sampler.py",
                "vllm-xcpu-plugin/tests/plugin_tests/test_dspark_phase2.py",
            ),
        ),
    ),
)
