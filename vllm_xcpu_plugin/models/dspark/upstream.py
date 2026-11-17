# SPDX-License-Identifier: Apache-2.0
"""Upstream contracts of the generic DSpark XCPU patches."""

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
    category="dspark",
    packages=("vllm_xcpu_plugin.models.dspark",),
    watched=(
        WatchedModule(
            module="vllm.v1.worker.gpu.spec_decode.dspark.speculator",
            sha256="97bcfaa655c6d4a5ecdfb17ab87b80547e8216b9d15a91c341798ebbb558f146",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.model_executor.models.qwen3_dspark",
            sha256="7be21c094a49520c41f418aceea71b338d5e15b14b3ddd052e6d6f7b7ac8ae79",
            source=_V031,
        ),
    ),
    patches=(
        Patch(
            id="dspark.speculator.greedy_argmax",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.dspark.speculator:XcpuDSparkSpeculator._sample_logits",
                "vllm_xcpu_plugin.models.dspark.speculator:install_dspark",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode:init_speculator",
                    source_hash="bc596126bc5e935c1e0b6ddeb46262d763816240f3faafc67d68a6a30457dbbc",
                    signature_hash="43e795507747ec43f4c34f625ef3f8251f9810b8051e0e846add7c998378cc02",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.dspark.speculator:DSparkSpeculator",
                    expect="vllm.v1.worker.gpu.spec_decode.dspark.speculator:DSparkSpeculator",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.dspark.speculator:DSparkSpeculator._sample_logits",
                    source_hash="9eaa8753ab4b4bf551012baba80e29dab56b6ad82e5257d8c46de5180ea896d7",
                    signature_hash="db2e17616c6001a9e559babd618dd2daeddd6723aa4ef42c636fc7fe91404cf1",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.speculator:DraftModelSpeculator._maybe_predict_acceptance",
                    source_hash="20c3ec594d52a6625f06595febd5cf101fba160a73687197ded6fb03109e6f51",
                    signature_hash="0250181f3d0cf9281f52ab727bca5adc1aa021aca9772fc0cae27e0f23b9ecc0",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.speculator:DraftModelSpeculator.sample_draft",
                    source_hash="1907997f07d984b23b2cf76940220cd09cd8822f7781e7ed399c162aca6738d0",
                    signature_hash="aa4fa87c1252f4e847b97170f37b9096714cc3a60d6052c469a59a2879164110",
                ),
            ),
            contract=(
                "init_speculator lazily imports DSparkSpeculator from "
                "vllm.v1.worker.gpu.spec_decode.dspark.speculator, so "
                "install_dspark() rebinds it before worker startup. "
                "XcpuDSparkSpeculator._sample_logits keeps acceptance prediction "
                "and probabilistic Gumbel sampling (resolving patched "
                "gumbel_module.gumbel_sample), while routing greedy "
                "(draft_logits is None) 2D contiguous 4-byte-aligned even-vocab "
                "BF16 mcpu logits through torch_xcpu.ops.argmax_out and falling "
                "back to logits.argmax(dim=-1) for non-contiguous/odd/FP32 "
                "slices, then calls model.map_draft_to_target(). Re-check: "
                "_sample_logits arguments/signature, map_draft_to_target, d2t "
                "scatter path, gumbel_sample contract."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/argmax.py::argmax_out",
                "vllm-xcpu-plugin/vllm_xcpu_plugin/sampler_patch.py",
            ),
            tests=("vllm-xcpu-plugin/tests/plugin_tests/test_dspark_phase1.py",),
        ),
        Patch(
            id="dspark.speculator.idle_dummy_skip",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.dspark.speculator:XcpuDSparkSpeculator._generate_draft",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.dspark.speculator:DSparkSpeculator._generate_draft",
                    source_hash="babc8d7624b2fd030d5e7187c8828aaed50f7b898dd0533ee441e9349d34668f",
                    signature_hash="61c15fcdc627f4c9c6488a27fdba85f803f0bab90aea21bf6cadb145672e95f1",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.dspark.speculator:DSparkSpeculator._sample_sequential",
                    source_hash="e9d235da1710aefe681bb795931e44a50c9cad59eee0db7fdedf659be896d115",
                    signature_hash="11e5e039624669e60fcfd3560fa2c371c9c3c3aa60b01650e85302c388b2523e",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.dflash.speculator:DFlashSpeculator._run_model",
                    source_hash="3299b8a5946e5042434513b2dc339abb317cffff3a0a24c4f8d983515d74bbff",
                    signature_hash="1b0c1545e5020614ea024d0153ffe7c7a108ce5a36f827e8aa0b43216b8d8d72",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.dflash.speculator:DFlashSpeculator.propose",
                    source_hash="dd98ee612e90b7b69a4c8ca2e91f357c95b9b1ff2eee6c0ce6613a57ef77273f",
                    signature_hash="9d016d35e3909614e06447b41f70e8cefee8e8426ec31fa90e24e4116bb66d98",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu_worker:Worker.execute_dummy_batch",
                    source_hash="3a316417eae8c596fa49846c727b56a1eb3f1e50bc974bfe6b8e59479569a6ed",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
            ),
            contract=(
                "Worker.execute_dummy_batch wraps runtime idle-DP sync batches "
                "with torch_xcpu.ops.set_dummy_run(True) (profiling/warmup/"
                "adaptive-verification leave is_dummy_run() False). When "
                "cudagraph_runtime_mode == NONE and is_dummy_run() holds, "
                "_generate_draft runs _run_model (for cross-DP EP MoE sync) "
                "and skips _sample_sequential (lm_head + intra-TP all_gather + "
                "Markov walk have no cross-DP collectives and all TP ranks of "
                "an idle DP group run execute_dummy_batch together). Re-check: "
                "cross-DP collectives in _sample_sequential, dummy draft_tokens "
                "usage in _dummy_run."
            ),
            tests=("vllm-xcpu-plugin/tests/plugin_tests/test_dspark_phase1.py",),
        ),
        Patch(
            id="dspark.speculator.markov_fused",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.dspark.speculator:XcpuDSparkSpeculator.__init__",
                "vllm_xcpu_plugin.models.dspark.speculator:XcpuDSparkSpeculator.load_draft_model",
                "vllm_xcpu_plugin.models.dspark.speculator:XcpuDSparkSpeculator._resolve_fused_markov_weights",
                "vllm_xcpu_plugin.models.dspark.speculator:XcpuDSparkSpeculator._sample_sequential",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.dspark.speculator:DSparkSpeculator.__init__",
                    source_hash="cc68f8cebf0b4ceb711575d5431c3eb3d9a4a022a58a5d796e71d4ce24a04183",
                    signature_hash="becf74c14dc557426e51d96975cf5f312398aa66833b36a97d68f6bde2b09557",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.dspark.speculator:DSparkSpeculator.load_draft_model",
                    source_hash="fba0258d3761aac8f106c565d4aed19683bad55367e7508f79c82b2d991af903",
                    signature_hash="b2fb736a3e2748f1312cebff1cd412d65460efdd75b4a131b8e92355d1bba995",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.dspark.speculator:DSparkSpeculator._sample_sequential",
                    source_hash="e9d235da1710aefe681bb795931e44a50c9cad59eee0db7fdedf659be896d115",
                    signature_hash="11e5e039624669e60fcfd3560fa2c371c9c3c3aa60b01650e85302c388b2523e",
                ),
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.dspark.speculator:DSparkSpeculator._sample_sequential_topk",
                    source_hash="26ad55603495ce275a15a403f96385f061c528c8decd7bf49da36b7ebd60a91b",
                    signature_hash="11e5e039624669e60fcfd3560fa2c371c9c3c3aa60b01650e85302c388b2523e",
                ),
                UpstreamRef(
                    target="vllm.model_executor.models.qwen3_dspark:DSparkMarkovHead.embed",
                    source_hash="f1a528436c57102e5fe77e0d85cd351cf78d238109c06e06f4bd0b960c6dafc7",
                    signature_hash="306bcdac009bc2a640425e3a14f2b22da2ca1875da6d30b19c92a555b4fa4cc0",
                ),
                UpstreamRef(
                    target="vllm.model_executor.models.qwen3_dspark:DSparkMarkovHead.bias",
                    source_hash="262803c56decb02559715da25fad2df1ec394a51a315daa486476e5173c89f1f",
                    signature_hash="6524747be09014370b50ae76fe4055e61481b91fb154e13dfe39501d51cb41ed",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4ForCausalLM.markov_embed",
                    source_hash="3f78c3c31ad74b94bda89a9da70981dc472042344520ee488dc7c9b008a8250a",
                    signature_hash="306bcdac009bc2a640425e3a14f2b22da2ca1875da6d30b19c92a555b4fa4cc0",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4ForCausalLM.markov_bias",
                    source_hash="0dd5d686b65902e61a1f6c3af507b37458858bf9fce21844da8e3a7377b8f96c",
                    signature_hash="52616eedaafb2cec814beeea2ad3390e83726754310a6373fcd4c68d8a1eb6e2",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4ForCausalLM.compute_confidence",
                    source_hash="311cac9ea151f35ccabf7824aacd40b99548521fab4b805ffc906a61e20305d7",
                    signature_hash="c537a8e67668dcbcfd73b25adbdccaf79c0df42b17a0372014b227f775490f17",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4ForCausalLM.map_draft_to_target",
                    source_hash="b1a3f6f90b765fe6a89799285865bf5d926225668e924e342269453db16229b7",
                    signature_hash="daeeada9a0a858a14f0579903ddc2792eb84299e6adef7635ae3d239c630a657",
                ),
                UpstreamRef(
                    target="vllm.model_executor.models.qwen3_dspark:Qwen3DSparkForCausalLM.map_draft_to_target",
                    source_hash="efb09678cd1e6047bdedc14534c6faa361570e6d7af46bf99243af84198a5cfd",
                    signature_hash="daeeada9a0a858a14f0579903ddc2792eb84299e6adef7635ae3d239c630a657",
                ),
                UpstreamRef(
                    target="vllm.model_executor.models.qwen3_dspark:Qwen3DSparkForCausalLM.markov_bias",
                    source_hash="0dd5d686b65902e61a1f6c3af507b37458858bf9fce21844da8e3a7377b8f96c",
                    signature_hash="52616eedaafb2cec814beeea2ad3390e83726754310a6373fcd4c68d8a1eb6e2",
                ),
            ),
            contract=(
                "XcpuDSparkSpeculator._sample_sequential fuses the n_spec sequential "
                "Markov embedding lookup, w2 linear bias, base_logits addition, "
                "greedy argmax / probabilistic Gumbel sampling, draft_logits cache "
                "write, and optional d2t offset mapping into "
                "torch_xcpu.ops.dspark_markov_sample (accepting INTEL_AMX packed "
                "layout 1, PORTABLE/ACC packed layout 2, and row-major BF16 layout 0), "
                "falling back to in-place .add_() when reduced-vocab scatter, "
                "acceptance estimator, watermarker, or logit scaling is active. "
                "Re-check: DSparkMarkovHead weights/bias, map_draft_to_target offset "
                "convention, and LogitsProcessor scale/soft_cap."
            ),
            impl=("torch_xcpu/torch_xcpu/ops_defs/dspark.py::dspark_markov_sample",),
            tests=(
                "torch_xcpu/test/ops/custom_ops/test_dspark_markov_sample.py",
                "vllm-xcpu-plugin/tests/plugin_tests/test_dspark_phase2.py",
            ),
        ),
        Patch(
            id="dspark.speculator.sp_aux_combine",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.dspark.speculator:XcpuDSparkSpeculator._combine_target_hidden_states",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.v1.worker.gpu.spec_decode.dflash.speculator:DFlashSpeculator._combine_target_hidden_states",
                    source_hash="2e27edda0efb6b4eb4e3b9570d763d671fb29753bbb80bb877a6954fb99d9386",
                    signature_hash="a0e93e80d01bd837f3069ee7939ac87c6d37f16e9e5335a18060a812e34dc58d",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4Model.combine_hidden_states",
                    source_hash="717f5792ab3b82f953159a3ecb8880f2fcf9b5bb3d0a13e801b1613349e35433",
                    signature_hash="331c951c98976528ab552b056abf24f28b117d36ef4ead37aa37fed6befb58b9",
                ),
            ),
            contract=(
                "When target DeepSeek-V4.1 returns "
                "SpShardedAuxHiddenStates(packed_shard, full_num_tokens), "
                "XcpuDSparkSpeculator._combine_target_hidden_states runs "
                "self.model.combine_hidden_states (main_proj + main_norm) on the "
                "TP shard before performing one all_gather_into_tensor_out_v2 into "
                "a dedicated _sp_gather_buf (never self.hidden_states). Re-check: "
                "combine_hidden_states elementwise assumptions under SP."
            ),
            tests=("vllm-xcpu-plugin/tests/plugin_tests/test_dspark_sp_aux.py",),
        ),
    ),
)
