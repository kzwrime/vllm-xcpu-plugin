# SPDX-License-Identifier: Apache-2.0
"""Upstream contracts of the DeepSeek-V4.1 XCPU patches.

Every XCPU patch of this package (OOT layers, overrides, rebinds, Fake Triton
adapters) is a ``Patch`` here: the upstream code it mirrors or relies on, the
semantics it keeps, where those semantics are implemented and tested. The
site lint (``python -m vllm_xcpu_plugin.upstream_contracts lint``) rejects a
patch site that is not listed.

Pins are updated by hand, per reference, after reviewing the upstream diff
against the contract (docs/UPSTREAM_CONTRACTS.md); ``local_commit`` is the
committed vLLM snapshot that holds the reviewed source.
"""

from vllm_xcpu_plugin.fake_triton.provenance import KernelSource
from vllm_xcpu_plugin.upstream_contracts import (
    Manifest,
    Patch,
    UpstreamRef,
    WatchedFile,
    WatchedModule,
)

_V031 = KernelSource(
    tag="v0.31.0",
    local_commit="0670c36056d45530524f498f731bf3847a7d4ce0",
    main_commit="283d76d56a45d9bbead5ba650402fb0957abc30c",
)
_ENGRAM = KernelSource(
    tag="v0.31.0 engram PluggableLayer",
    local_commit="3f08bf786ea1dafd084fb895a880d2e81cbef315",
    main_commit="283d76d56a45d9bbead5ba650402fb0957abc30c",
)

MANIFEST = Manifest(
    category="deepseek_v41",
    packages=("vllm_xcpu_plugin.models.deepseek_v41",),
    checks=(
        "docs/upgrade/11-dsv41-serial.md",
        "docs/upgrade/12-dsv41-dp2tp2-ep.md",
    ),
    watched=(
        WatchedModule(
            module="vllm.models.deepseek_v41.attention",
            sha256="44b870805c1c796715d13029604428aa38a74dd1eebafbe1cc6879817b66d957",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.deepseek_v41.compressor",
            sha256="5807de8cb84b945b088d1f951125e38e0e2ad5581b6ad8de246292782c0e2646",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.deepseek_v41.sparse_mla",
            sha256="1b87aa3f27e7867b6508893da5f52a0f10944b59fd03aa58bd8c568f4e6aef16",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.deepseek_v41.nvidia.model",
            sha256="e54977e4c085c5c184ad28cd8e5063e4c59aefe51921f9ae3887dbc3f35a3222",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.deepseek_v41.decoder_replay_layers",
            sha256="be94bba9c89420e238a31e423e173255d6058fb0b0dbc4983e52d3e857249947",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.common.ops.sequence_parallel",
            sha256="699fde98360fc8d604055e2987d93108ee489e87cc5563949d2a3dcd838eb8bd",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache",
            sha256="a3f81abbab1f507e441176222578f90056e6b4bc0063bb36681c230a84b3d969",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.deepseek_v41.common.ops.indexer_k_store",
            sha256="d89d7043ec7924f1636f8ef257acf5ed0c66687681c7e905ceb16321155360f6",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.deepseek_v41.common.ops.cache_utils",
            sha256="26d039cee17ec3b699a97cb3a8aedb9ca39826766861ed4040adf5bafd657b2c",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.model_executor.kernels.attention.dsa.candidate_blocks",
            sha256="45f1eaa3b5f1d9f6a7b554697709646ce41d79d55f2d82415d008e6b00a1136f",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.model_executor.kernels.mhc.torch",
            sha256="3697085aa6f0070e165c145f9c337ca3fe0623623c8265d29e50c43467e49985",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.v1.attention.backends.mla.sparse_swa",
            sha256="870adaf9bec93dc818a18ba8057035a08d687e7ea415b76c12d9fea28ca0cbef",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.v1.attention.backends.mla.compressor_utils",
            sha256="68868c55e3bb09ce64df9d43ddca6544f24a4342e276e7073b3a42480918420a",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.deepseek_v4.common.ops.fused_indexer_q",
            sha256="ce976c4f744f0e053045e1d74927e8d76e36e1745a19b9a49c23bf80b91bb196",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.deepseek_v41.common.engram",
            sha256="8f375e8a9d753ca117de35140468a70587859ea161d698b8168bbdde402816e4",
            source=_ENGRAM,
        ),
        WatchedModule(
            module="vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router",
            sha256="8cf50cf262bd0c2411fbcbf1d1ac47cfcf9320eefcb1e125f04ac27bdd432ba8",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.deepseek_v41.nvidia.dspark",
            sha256="480149ff32a04ab482fa6ba1e84369bd91a1f59dcb04a6bd43f791f29ecb5bc0",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.model_executor.kernels.mhc.tilelang",
            sha256="204e19209db5b19eb3dd25ea6ae10c84425dee02ef5bc896eca3f83da6149a8e",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.model_executor.kernels.mhc.triton",
            sha256="a48f030e5df658416eb7d0abadad62c4742e4004fcee0f8f0514664e77965ef2",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.model_executor.layers.fused_moe.router.dsv4_topk",
            sha256="14f7c8487b9aef652dd61c91e0e3c3640b643bf9cd0d8a7184157d71f4beb63e",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.deepseek_v41.nvidia.ops.mhc",
            sha256="c8233a70004719017ff6a5826ca538c92ea6319a9c7c0090dc8fc2e603db25e7",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn",
            sha256="d737d96e8d7848b47e4181efb435807f793aca28fc8718afaa83365cc3945ed6",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.model_executor.kernels.linear.mxfp8.emulation",
            sha256="a5b93da2cfdc0ff6b3ba4645b33724de783cd436035523d9afc6fb02a392c0b2",
            source=_V031,
        ),
    ),
    watched_files=(
        WatchedFile(
            path="csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu",
            sha256="a18d7561bc8350664b08fba6a29cedbab21c21d2d126d0bd32836a8a90240a77",
            source=_V031,
        ),
        WatchedFile(
            path="csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu",
            sha256="543d46df95119a64537b47bc7089a238c15255bf60fee361526c64611df27247",
            source=_V031,
        ),
    ),
    patches=(
        Patch(
            id="deepseek_v41.router.topk_hash_softplus_sqrt",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.ops:topk_hash_softplus_sqrt",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm._custom_ops:topk_hash_softplus_sqrt",
                    source_hash="741bbdba0e83e646dcde2e06564c21b7912471c0aa494c2ac2b518fb14e8490f",
                    signature_hash="801aa10f992609abdc27b1ffc5309cae08a9d48201c835d3f6acf6d73b902323",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router:vllm_topk_softplus_sqrt",
                    source_hash="bcf27848bb2a9d1f437adba004f7aa05a1698c88a94e5c98c55a31a605fd34b2",
                    signature_hash="d6da21b0f83467641c9cd99daa61239986fc11c9b8c5d0aa0ed99a05493d9126",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router:_get_padding_mask",
                    source_hash="d4edb0a8f6149e9c8792be47ae447bfea6c6b130f19fae18abe87af1948f3592",
                    signature_hash="bbaca9d85b7578d35157f850653de5d23033a2c84ab7facb710fe725f8d1a1a0",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.router.dsv4_topk:dsv4_topk",
                    source_hash="ea11b51bb975e90174e680cd2595be8bb20f7895d481cd1df4b943aed4755f64",
                    signature_hash="474ffda34a77833210f2e831df3876e56fb7f3b99fd417b364e3dd75713478b5",
                ),
            ),
            contract=(
                "The upstream router (vllm_topk_softplus_sqrt) runs as is; only "
                "the custom-op wrapper it calls is rebound. One XCPU launch fills"
                " topk_weights/topk_indices in place: experts by sqrt(softplus) "
                "score + bias, weighted by the raw scores, renormalized and "
                "scaled; image tokens (five sentinel ids from image_sentinel_lo) "
                "pick with bias_vl; is_padding rows (the SP tail, from "
                "_get_padding_mask) get expert -1. Hash routing is rejected; "
                "token_expert_indices is not written. Re-check: argument order, "
                "new routing inputs, pad-row and sentinel semantics in the CUDA "
                "kernel, output dtypes."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_topk_softplus_sqrt",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_ops.py::test_topk_softplus_sqrt",
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_ops.py::test_topk_softplus_sqrt_image_and_padding_rows",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
                "docs/upgrade/12-dsv41-dp2tp2-ep.md (SP, DP-sharded engram)",
            ),
            upstream_files=("csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu",),
        ),
        Patch(
            id="deepseek_v41.model.attention_class",
            source=_V031,
            downstream=("vllm_xcpu_plugin.models.deepseek_v41.model:_select_attn_cls",),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:_select_dsv4_attn_cls",
                    source_hash="b398a48f8702251fd0be75f8f88f957895ba3f0360b8c74f321b5cbe3558bdbf",
                    signature_hash="f885002db3c231aa84fcdc6c2c3662b2ef990b71fdc564f859dfbd51b5017d3b",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4DecoderLayer.__init__",
                    source_hash="8f53bfbb84070983c3a2f076377b72f2b7f0bb6bd72b2f336bced9019b026ae2",
                    signature_hash="cc45aab73bfa124ee908bd07d4180e9b8f546469f8012eb8b2f78af498d86043",
                ),
            ),
            contract=(
                "The decoder layer builds its attention through "
                "_select_dsv4_attn_cls(vllm_config)(vllm_config, prefix=..., "
                "topk_indices_buffer=..., aux_stream_list=..., "
                "candidate_block_buffer=...). XCPU always returns "
                "XcpuDeepseekV41Attention and rejects an explicit attention "
                "backend. Re-check: new constructor arguments, a second selection"
                " point, backend-dependent behavior moved into the layer."
            ),
            tests=("docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",),
        ),
        Patch(
            id="deepseek_v41.model.mhc",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.ops:mhc_pre_delayed",
                "vllm_xcpu_plugin.models.deepseek_v41.ops:mhc_post",
                "vllm_xcpu_plugin.models.deepseek_v41.ops:mhc_shifted_post_pre",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:mhc_pre_delayed_tilelang",
                    source_hash="af72fabf43d3a4b88cdf716e8e7cda1f7e29f3a341ff8b9de26aa8ac2b24b896",
                    signature_hash="7c4a5bfc06236c5001c74631275f66161d060b62be831d2193be67fd237f7939",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:mhc_post_tilelang",
                    source_hash="8397404ba0eb458583e1bab6a6441d9f18338496eb5d1c6e7e8e719d552a6d66",
                    signature_hash="dad6f030863ab2aee59dfce3eec27ae4a97b40a69caf0f8d6e92c3f78526ab2c",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:mhc_shifted_post_pre",
                    source_hash="416ec7cf492138f1cc76398c8f71f378eb95516bfc5813561d99a1f6cd1b4da1",
                    signature_hash="9858f5c815016bfc36f76451630796fd250cd6595f86e71a8084ea984c473f99",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:mhc_post_tilelang",
                    source_hash="8397404ba0eb458583e1bab6a6441d9f18338496eb5d1c6e7e8e719d552a6d66",
                    signature_hash="dad6f030863ab2aee59dfce3eec27ae4a97b40a69caf0f8d6e92c3f78526ab2c",
                ),
                UpstreamRef(
                    target="vllm.model_executor.kernels.mhc.tilelang:mhc_fused_post_pre_delayed_tilelang",
                    source_hash="92d77ed5f37b14e41f978578ceb900289c92317e37e106c28245ff01f4eab8ba",
                    signature_hash="a4a98a57643d89e5e345fec079b65c249764d43498f9b0f4b7c409208acc2515",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4DecoderLayer.forward",
                    source_hash="8a1b1f3b48d0d281a1ca2cbcd431909fffb5be1d5d668c53eb929bceb47a966b",
                    signature_hash="5978820871604b84cdf691be6fafa70c1c2abebce513ff6e75784e6f4fe2d53f",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model._collapse",
                    source_hash="683aac3775fc3698c444d9ed7dd5cb036e451ace1e7238b5b824951fcd8b6606",
                    signature_hash="a7e1bff9c30125f9179e26cbc59842cf536015eee811420088379c0986423500",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4Model.forward",
                    source_hash="31fea44165b300724f70ed11d062f8750c1e508acec453540837bfff9feab778",
                    signature_hash="9a5bcb0ef7eb5e05f220c34d0dd8d7751c71baf50484755a4fb0f963a8980235",
                ),
            ),
            contract=(
                "mHC pre (delayed sinkhorn mix), post and the shifted post+pre of"
                " the next sublayer are rebound by name in the nvidia model and "
                "DSpark. The XCPU ops keep the tilelang signatures, return tuples"
                " (post_mix, res_mix, x, pre[, aux]) and fp32 mixes; the "
                "<=16-token fused path vs split path is chosen inside the op at "
                "run time. Re-check: argument order at the call sites, new "
                "outputs (aux capture), eps/alpha/sinkhorn parameters, norm "
                "fusion flags."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_mhc_pre_delayed",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_mhc_shifted_post_pre",
                "torch_xcpu/torch_xcpu/ops.py::deepseek_v4_mhc_post",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_ops.py::test_mhc_pre_delayed",
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_ops.py::test_mhc_shifted_post_pre",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.model.hc_collapse",
            source=_V031,
            downstream=("vllm_xcpu_plugin.models.deepseek_v41.ops:hc_collapse",),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:hc_collapse_triton",
                    source_hash="a55f85db717bd88e639defa59688a87d8ad74eb8e985eff6ab3008d8d92d7435",
                    signature_hash="00d7bfbbdfc3733ba9d67c8ad7c6160bd4ff280519d634eb3b1f7996e03e8726",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:hc_collapse_triton",
                    source_hash="a55f85db717bd88e639defa59688a87d8ad74eb8e985eff6ab3008d8d92d7435",
                    signature_hash="00d7bfbbdfc3733ba9d67c8ad7c6160bd4ff280519d634eb3b1f7996e03e8726",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model.forward",
                    source_hash="24aa409a437ae7c6da1e219af1c6c0ab7d36a8da32e9ccefd97eb5b022a6e1dc",
                    signature_hash="7c73aeee0e17a617b3d6038ae156bf93af563da7931df94603f79ffab05c87c3",
                ),
            ),
            contract=(
                "hc_collapse(x [T, hc, D], pre_mix [T, hc]) -> [T, D] weighted "
                "sum of the hc streams, rebound in the model and DSpark. "
                "Re-check: mix layout and dtype, any normalization added to the "
                "collapse."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_hc_collapse",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_ops.py::test_hc_collapse",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.model.sp_reduce_scatter",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.model:_sp_reduce_scatter",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:sp_reduce_scatter",
                    source_hash="3cda5554aa5dfb40131784f7d3a6761e65dfadc67dec0d19b9b8f70d13f63a04",
                    signature_hash="3000125d9016840993fbfb7f0f71cd2c8ef3938fb963a42dd930e6fead173043",
                ),
                UpstreamRef(
                    target="vllm.models.common.ops.sequence_parallel:_custom_collective",
                    source_hash="153902a3f6e9304b5fc20461499cdf42e5ff455911c4a477c1fce3819a9bc7a9",
                    signature_hash="b5a07190ab1523cc03258f2124c5664997e8643201fb338c35de877242a8fbb9",
                ),
                UpstreamRef(
                    target="vllm.models.common.ops.sequence_parallel:tensor_model_parallel_reduce_scatter",
                    source_hash="3ae5201855c59857a5e25106456c3367501b59b9fa08342429ac3bbb59f3bdd6",
                    signature_hash="5f06b237417996a014fdb881912f9ec0273f9441a1ab8f748615bb65ee83914d",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4DecoderLayer.forward",
                    source_hash="8a1b1f3b48d0d281a1ca2cbcd431909fffb5be1d5d668c53eb929bceb47a966b",
                    signature_hash="5978820871604b84cdf691be6fafa70c1c2abebce513ff6e75784e6f4fe2d53f",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._wo_b_proj",
                    source_hash="12c194c3532d75e3469591223a60566960f8124fc32c73b0171e258c26c7d65a",
                    signature_hash="6e05affbd4c64aa537fa31e70f46ed8c2bcbd5b32828a21761335feaf389ed18",
                ),
            ),
            contract=(
                "Sequence-parallel reduce-scatter of [T, D] rows across TP that "
                "pads by the symbolic remainder, so one TRACE_ONCE graph serves "
                "odd token counts (upstream pads only if sp_pad > 0). Only the "
                "decoder layer's name is rebound: DeepseekV4Attention._wo_b_proj "
                "keeps the attention module's own sp_reduce_scatter and is only "
                "reached with a gemm_rs, which XCPU never enables. Re-check: "
                "padding/unpadding contract, the custom collective path, new "
                "callers of sp_reduce_scatter."
            ),
            tests=("docs/upgrade/12-dsv41-dp2tp2-ep.md (SP, DP-sharded engram)",),
        ),
        Patch(
            id="deepseek_v41.model.compiled_layer_stacks",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.model:SpShardedAuxHiddenStates",
                "vllm_xcpu_plugin.models.deepseek_v41.model:XcpuDeepseekV41Model.__init__",
                "vllm_xcpu_plugin.models.deepseek_v41.model:XcpuDeepseekV41Model.source_aux_layer_ids",
                "vllm_xcpu_plugin.models.deepseek_v41.model:XcpuDeepseekV41Model._run_layers",
                "vllm_xcpu_plugin.models.deepseek_v41.model:XcpuDeepseekV41Model._run_replay_layers",
                "vllm_xcpu_plugin.models.deepseek_v41.model:XcpuDeepseekV41Model._collapse",
                "vllm_xcpu_plugin.models.deepseek_v41.model:XcpuDeepseekV41Model.forward",
                "vllm_xcpu_plugin.models.deepseek_v41.model:_XcpuDeepseekV41LayerStack.__init__",
                "vllm_xcpu_plugin.models.deepseek_v41.model:_XcpuDeepseekV41SourceLayers.forward",
                "vllm_xcpu_plugin.models.deepseek_v41.model:_XcpuDeepseekV41ReplayLayers.forward",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV41LLMForCausalLM.model_cls",
                    expect="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model.__init__",
                    source_hash="59a0cd548c9c1b9b23bda068a38483d5e48c9d7f9f2e7c0287de665dd12b6a43",
                    signature_hash="0268aa001d71a5bab9e4761758b7b321b8822f20f4554d61286ec2ca4dd91920",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model.forward",
                    source_hash="24aa409a437ae7c6da1e219af1c6c0ab7d36a8da32e9ccefd97eb5b022a6e1dc",
                    signature_hash="7c73aeee0e17a617b3d6038ae156bf93af563da7931df94603f79ffab05c87c3",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model._run_layers",
                    source_hash="4fa97d6745e44b4f00d19dfc64aad85f486b3ad500b1f5a6e81be0a2545104fa",
                    signature_hash="5551fa45f60436275c5b1a36ea9abeb6206659964dd3a1db4e862c28ca21a4a7",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model._run_replay_layers",
                    source_hash="74282c05206a1792dc7d5b87e40dc9b79b9e73912f49554228cfc3b238947767",
                    signature_hash="f655e16732a4b5dd381a6ea2c54dd65c2a681b8fde4667a19a48bfd84dbe9dfe",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model._decoder_replay_supported",
                    source_hash="6cf6ec982fb01383048c40012b92d413c353e73886e69f9113fb000a337bbaaa",
                    signature_hash="71a4c3fa1de65d26d5c270c8183ba40181c1f5b3d0f75bd701be1dd6cb425232",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model._collapse",
                    source_hash="683aac3775fc3698c444d9ed7dd5cb036e451ace1e7238b5b824951fcd8b6606",
                    signature_hash="a7e1bff9c30125f9179e26cbc59842cf536015eee811420088379c0986423500",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model._mega_gate_metadata",
                    source_hash="1cc3ed4de487400425807b4a7b73a5204f956c8c36a178cc1c91d49b56adc638",
                    signature_hash="a98c0095fa23c28d249e965a743757e6e7bfcde6f46a0c0a2b6ab87395c1e818",
                ),
                UpstreamRef(
                    target="vllm.model_executor.kernels.mhc.tilelang:mhc_post_tilelang",
                    source_hash="8397404ba0eb458583e1bab6a6441d9f18338496eb5d1c6e7e8e719d552a6d66",
                    signature_hash="dad6f030863ab2aee59dfce3eec27ae4a97b40a69caf0f8d6e92c3f78526ab2c",
                ),
            ),
            contract=(
                "DeepseekV4Model.forward stays eager for engram hashing, decoder "
                "replay sub-batch, and SP sharding, while packing aux hidden "
                "states into a preallocated buffer for DSpark (and deferring SP "
                "all-gather across aux layers into SpShardedAuxHiddenStates when "
                "use_sequence_parallel and PP=PCP=1). The decoder layers are "
                "traced as two graphs: layers [start_layer, decoder_replay_start) "
                "and replay layers + collapse. Relies on _run_layers(..., "
                "aux_hidden_by_layer, ...) returning 5 tensors, _collapse(...), "
                "_mega_gate_metadata, mhc_post_tilelang, and _run_replay_layers. "
                "Re-check: argument lists, return arity, aux capture rule, "
                "attributes set in __init__, new eager-only branches in _run_layers."
            ),
            tests=(
                "vllm-xcpu-plugin/tests/plugin_tests/test_dspark_sp_aux.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
                "docs/upgrade/12-dsv41-dp2tp2-ep.md (SP, DP-sharded engram)",
            ),
        ),
        Patch(
            id="deepseek_v41.dspark.context_kv_insert",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.model:_insert_context_kv",
                "vllm_xcpu_plugin.models.deepseek_v41.model:XcpuDSparkDeepseekV41Model.__init__",
                "vllm_xcpu_plugin.models.deepseek_v41.model:XcpuDSparkDeepseekV41Model.combine_hidden_states",
                "vllm_xcpu_plugin.models.deepseek_v41.model:XcpuDSparkDeepseekV41Model.precompute_and_store_context_kv",
                "vllm_xcpu_plugin.models.deepseek_v41.model:XcpuDSparkDeepseekV41ForCausalLM.load_weights",
                "vllm_xcpu_plugin.models.deepseek_v41.model:XcpuDSparkDeepseekV41ForCausalLM.process_weights_after_loading",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4Model",
                    expect="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4Model",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4Model.__init__",
                    source_hash="e32693e1543b42d4191226d01354f1f25f7cf433b2a340c5c6d095639e24afb1",
                    signature_hash="0268aa001d71a5bab9e4761758b7b321b8822f20f4554d61286ec2ca4dd91920",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4Model.combine_hidden_states",
                    source_hash="717f5792ab3b82f953159a3ecb8880f2fcf9b5bb3d0a13e801b1613349e35433",
                    signature_hash="331c951c98976528ab552b056abf24f28b117d36ef4ead37aa37fed6befb58b9",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:_insert_context_kv",
                    source_hash="03397d5d5c05000d9518e718f5a6822cebd9d722779b08dc70f632f5db96e610",
                    signature_hash="601d281fd36741367abe125aa5375c2b103de7ead0f9c63e486841433f857a8e",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4Model.precompute_and_store_context_kv",
                    source_hash="9567877a2e2ee63382f22fc2cc349e8132a2c81f859fed3bff5154045550cce4",
                    signature_hash="ca37bf2241f93a2e40ef31570104253af3358390d0fecb4658f0d7638ded6794",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4ForCausalLM.__init__",
                    source_hash="19d27a794c12036dc94d704054c31699c6d9589990a0ca476032cb3cddca7e8b",
                    signature_hash="0268aa001d71a5bab9e4761758b7b321b8822f20f4554d61286ec2ca4dd91920",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4ForCausalLM.load_weights",
                    source_hash="e1ccc2a56a3f7570bbc12f7fb9079b6b6404a20e640eef8e1f1f51a1bbc8611d",
                    signature_hash="24bcf75dc8091fd46a88cb78817a9f1f45d8eff388d0d84c185e4f7f2c4424da",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4ForCausalLM.process_weights_after_loading",
                    source_hash="5a873008b7f7ebbcba694b0c626ac9b0c8d4dc60b5d82f03660af443bf49af58",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention.__init__",
                    source_hash="c39e8002fdd59dfc601443545aa26e60eea3ae6c452fcbfa83dff5248ffb4599",
                    signature_hash="8fd601110da1a879ac85a8e1e8d9067b86cdeeb97da54849fa5cf47f2d4ce089",
                ),
            ),
            contract=(
                "DSpark context KV: (1) _insert_context_kv is the KV half of "
                "fused Q/KV insert writing attn.swa_cache_layer.kv_cache at "
                "slot_mapping with the layer's rotary cos/sin cache; "
                "(2) XcpuDSparkDeepseekV41Model marks fused_wqa_wkv's "
                "[q_lora_rank, head_dim] wkv offset before quant/bf16 packing "
                "and stacks the 3 layers' unpacked wkv slices once in post-quant "
                "process_weights_after_loading (skipped during load_weights's "
                "pre-quant call via _loading_checkpoint_weights) into one "
                "torch_xcpu INT8/FP8/BF16 linear, freeing temporary raw stashes; "
                "(3) under torch_xcpu.ops.is_dummy_run() (runtime idle-DP sync "
                "batches), combine_hidden_states and unslotted "
                "precompute_and_store_context_kv return without GEMMs. "
                "Re-check: fused_wqa_wkv shard order, q_lora_rank scale-block "
                "alignment, load_weights vs model_loader "
                "process_weights_after_loading order, and KV record layout."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_qnorm_rope_kv_insert",
                "vllm-xcpu-plugin/vllm_xcpu_plugin/layers/int8_ue8m0.py::XcpuInt8Ue8m0LinearMethod",
                "vllm-xcpu-plugin/vllm_xcpu_plugin/layers/mxfp8_linear.py::XcpuMxfp8LinearKernel",
                "vllm-xcpu-plugin/vllm_xcpu_plugin/layers/fp8_linear.py::XcpuFp8BlockScaledMMLinearKernel",
            ),
            tests=(
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
                "vllm-xcpu-plugin/tests/plugin_tests/test_dspark_phase1.py",
            ),
            upstream_files=(
                "csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu",
            ),
        ),
        Patch(
            id="deepseek_v41.attention.kv_records",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.model:_use_v41_mxfp8_kv_record",
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Attention.packed_kv_cache_dtype",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:_use_v41_mxfp8_kv_record",
                    source_hash="6b6b5dd639d3d86b1f9984781f2f130d8b4b7dc4ed52b85ed83b360dafde0b77",
                    signature_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention.packed_kv_cache_dtype",
                    source_hash="78eef1cd187ec7029648512aa848e1e6c9511386f039b30f90436cd16f3da8cc",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention.__init__",
                    source_hash="c39e8002fdd59dfc601443545aa26e60eea3ae6c452fcbfa83dff5248ffb4599",
                    signature_hash="8fd601110da1a879ac85a8e1e8d9067b86cdeeb97da54849fa5cf47f2d4ce089",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4IndexerCache.__init__",
                    source_hash="e73d798a19c6a15941886ab30a6ca07da5be8fe92b75c37f51431c8f8b913979",
                    signature_hash="79b59f9e7b58d369c44f8f0e5554645d7bde8f3501815b1ac99e952d3bafb49a",
                ),
            ),
            contract=(
                "XCPU stores the SM100 record set: V4.1 MXFP8 SWA record (512 fp8"
                " + 16 UE8M0, 528 B), NVFP4 compressed KV (288 B, packed dtype "
                "nvfp4_ds_mla) and the 68 B MXFP4 indexer key (see "
                "config.indexer_k_cache). Re-check: record byte layouts, page "
                "rounding, where the predicate is consulted (cache shapes, "
                "backend choice)."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_qnorm_rope_kv_insert",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_compress_rope_insert",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_indexer_k_norm_rope_store",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_qnorm_rope_kv_insert.py",
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_compress_rope_insert.py",
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_indexer_k_norm_rope_store.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
            upstream_files=(
                "csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu",
            ),
        ),
        Patch(
            id="deepseek_v41.config.indexer_k_cache",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.config:update_deepseek_v41_config",
                "vllm_xcpu_plugin.models.deepseek_v41.model:_indexer_uses_fp4",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:dsa_indexer_uses_fp4",
                    source_hash="6d1440e9352ccc01a4f8e9d1bdcb13fb6aa7bb580f97b08181e8fb29a446f101",
                    signature_hash="f885002db3c231aa84fcdc6c2c3662b2ef990b71fdc564f859dfbd51b5017d3b",
                ),
                UpstreamRef(
                    target="vllm.config.attention:AttentionConfig.resolve_indexer_kv_dtype",
                    source_hash="68a981383e5008d7998786d895a77878af6d59bbc0c7813ee2a90cd565ccfc2c",
                    signature_hash="0d41d197bb3700637c41fcac1b7d9050eb6b6ebdb43f0ccda4ef7947f0264bd3",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Indexer.__init__",
                    source_hash="095caeebf56d83e2ea5409a77ffce4281dbcbed4a6542c9e4a8881017608b389",
                    signature_hash="9a3f3f7231531fd3d227f8a2087a7cf00c09e79faaf38a22471dc4928d5bbf51",
                ),
            ),
            contract=(
                "XCPU stores only the 68 B MXFP4 indexer key record. The platform"
                " hook resolves an unset indexer_kv_dtype to mxfp4 and rejects "
                "any other dtype and indexer_sparse_logits (which makes candidate"
                " consumers build the CUDA SparseMQAIndexer through mqa_cls). "
                "dsa_indexer_uses_fp4 is rebound in the V4.1 attention module to "
                "True because upstream also gates MXFP4 on SM10x GPUs and would "
                "reject MCPU. Re-check: new consumers of indexer_kv_dtype / "
                "indexer_sparse_logits, new checks in dsa_indexer_uses_fp4, "
                "indexer op selection in DeepseekV4Indexer.__init__."
            ),
            tests=("docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",),
        ),
        Patch(
            id="deepseek_v41.attention.layer_boundary",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Attention.__init__",
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Attention.get_padded_num_q_heads",
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Attention._opaque_prepare_and_attn",
                "vllm_xcpu_plugin.models.deepseek_v41.attention:_xcpu_deepseek_v41_prepare_and_attn",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention.__init__",
                    source_hash="c39e8002fdd59dfc601443545aa26e60eea3ae6c452fcbfa83dff5248ffb4599",
                    signature_hash="8fd601110da1a879ac85a8e1e8d9067b86cdeeb97da54849fa5cf47f2d4ce089",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention.get_padded_num_q_heads",
                    source_hash="a33443e4d3d156a3738cd18d7511308856ba4a4ae5e58aafa1b7d88551cf1df0",
                    signature_hash="cf258453d569405b2d41b8b4ec9dbc277a347eeaac462019ee4f2de83c11495c",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention.forward",
                    source_hash="6fbfaafcbab9b177d5c40ca068500527b4ce10fd81e0c192c6d51e84b282a764",
                    signature_hash="31afdfe9093faef0390443559ec092558184d999fdca331e5a52e02dc93c79c7",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._prepare_and_attn",
                    source_hash="7f930e52258a4c45b8fd66faa5c588ef6113ba5db2ddff660cb0449760cbaf32",
                    signature_hash="cd6c6c5b3b75c5fba8568efa29b6ccb65299bf90ba7bb3863bb188a6e9908187",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._prepare_and_attn_eager",
                    source_hash="b35cffbb4d3ade3fcf94c46aa1b065a98d1753d33badc57aab8892d2b75f76a0",
                    signature_hash="cd6c6c5b3b75c5fba8568efa29b6ccb65299bf90ba7bb3863bb188a6e9908187",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._sparse_indexer_and_attn",
                    source_hash="46dd231988524c328b3ad756b0c2b978552471a5cffd09f3e79eabf7ad8acab2",
                    signature_hash="fc88d8f72ea14e33925b1fe5f1481494d69766366c1b2c8e93c03421880e975d",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._alloc_attn_out",
                    source_hash="495883d48c2728827bc79dcab13efb593762756d4ecaca194e57fbf80fc75af0",
                    signature_hash="792309d92d17033810dc3b2a67a028eba3832851f76debdeeca4b008598b5d85",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._wq_b_proj",
                    source_hash="b9439009daebf29808697219052b690838871ee229770087259c24859c51ab54",
                    signature_hash="c8b41db18639df77a4e8589e528c3c16a2f6c51c39e2d133e7305e11324133ed",
                ),
                UpstreamRef(
                    target="vllm.utils.multi_stream_utils:maybe_execute_in_parallel",
                    source_hash="19ad7589aa19874f46816ead238e87d8c64c2852d0eed8a675ed44899419afe0",
                    signature_hash="b5298dc8c55ed5276a95870b581da15e1bcdfc4f88e2c669d6d1a74e88278a1c",
                ),
                UpstreamRef(
                    target="vllm.compilation.breakable_cudagraph:eager_break_during_capture",
                    source_hash="b52e64b92b0bc67d9789485db5320303150d55cdcf72c4e849427cad77100e45",
                    signature_hash="ebf11689d57f9a4ec07a0c8cc1ba0373fa56611af32ab75eca2e66e8acc476f9",
                ),
            ),
            contract=(
                "The upstream forward and _prepare_and_attn run as is. XCPU "
                "points the _prepare_and_attn_fn hook at the opaque op "
                "vllm::xcpu_deepseek_v41_prepare_and_attn (the CUDA eager break: "
                "the cache writes are invisible to torch.compile and the single "
                "TRACE_ONCE trace has no metadata), which runs the upstream "
                "_prepare_and_attn. With aux_stream_list None every "
                "maybe_execute_in_parallel runs fn0 then fn1 on the current "
                "stream: wq_b, qnorm/RoPE/SWA insert, compressor, indexer (K "
                "write before scoring), compressed insert, indexer top-k, sparse "
                "MLA. Heads are not padded; wo_a runs grouped (is_bmm=False); the"
                " compressor wkv/wgate keeps an FP32 accumulator. Re-check: new "
                "steps or CUDA-only calls in _prepare_and_attn, the hook and its "
                "argument list, stream/event use when aux is None."
            ),
            tests=("docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",),
        ),
        Patch(
            id="deepseek_v41.attention.projections",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Attention._run_parallel_input_projections",
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Attention._split_qkv_and_norm",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._run_parallel_input_projections",
                    source_hash="58e3d0388dcbecb59ddd3582bf09ae9b0b44c5324baedb2d04b116c26ecc068c",
                    signature_hash="3dd7730ef51b9703e095ab53c9dee32f3d978a40f9f32e35922754ac96ca08e1",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._split_qkv_and_norm",
                    source_hash="1fb238f961d0d67a7e58d5744c2fc1f8d563f4f8f2bea49d6a7481876d7bb761",
                    signature_hash="b95e7ce3223b5fcc8d639e474a275f09ac2ab722328412af3493af63097da5cd",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._fused_wqa_wkv_gemm",
                    source_hash="215d26eb0cff69ecde4bf6ad327484dd597167d82ca163c64cf6e1a51373e30d",
                    signature_hash="3dd7730ef51b9703e095ab53c9dee32f3d978a40f9f32e35922754ac96ca08e1",
                ),
            ),
            contract=(
                "One fused wq_a|wkv GEMM, the compressor wkv/wgate and the "
                "indexer weights_proj from the same hidden states; the q/kv split"
                " at q_lora_rank and the two RMSNorms fused (fused_q_kv_rmsnorm)."
                " Returns (qr, None, kv): BF16, no qr scale. Re-check: projection"
                " set and order, split widths, norm eps, quantized activations."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_fused_q_kv_rmsnorm",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_fused_q_kv_rmsnorm.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.attention.swa_insert",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Attention._fused_qnorm_rope_kv_insert",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._fused_qnorm_rope_kv_insert",
                    source_hash="cb55958ee1b5a5f9b585e89c63faf89d333ba3be08e0fda0d24ef0feb26b6038",
                    signature_hash="6c52b3064d395496a21d111652f07bda97a6cc08f00aab29f04b2b7e342390d5",
                ),
            ),
            contract=(
                "Per-head q RMSNorm + RoPE and the KV RoPE + MXFP8 SWA record "
                "write at the SWA slot mapping, padded_heads query heads. Without"
                " metadata (profile) q passes through. Re-check: which rotary "
                "cache and positions, record layout, norm placement."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_qnorm_rope_kv_insert",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_qnorm_rope_kv_insert.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
            upstream_files=(
                "csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu",
            ),
        ),
        Patch(
            id="deepseek_v41.attention.compressor",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.ops:fused_save_compress_norm",
                "vllm_xcpu_plugin.models.deepseek_v41.ops:rope_quant_insert",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.compressor:fused_save_compress_norm",
                    source_hash="c1303425db10eb933ee409e776ff582b2a13ffc041cdbd5f2bbb8bb7d317c4b3",
                    signature_hash="286e9f86bc4c5dd254958a342d0a7f9a5bcb692b6341e8f125e81ddbbceb6a9c",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.compressor:rope_quant_insert",
                    source_hash="8afda256761a665bac17fa6d974e3a6086744fd2f4b45a49c600efb3f7347660",
                    signature_hash="c343b58718aa87e26627c05146acc68cf2ae4e6759e1abe45aa9338f63d25f86",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.compressor:DeepseekCompressor.forward",
                    source_hash="53ca239f7c83abfcceecc42c5083278a445e08bf86a1e6612dc7364b5df0752c",
                    signature_hash="7c61a9d94556bdc7f63916440e3c4440d6e691fe56a16fe80580d1ad76b4f200",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.compressor:DeepseekCompressor.insert_cache",
                    source_hash="06aa6b7b7319ffbdfbb07505e2652d02a3cf7c0ea6a5b616d76a3c83a2a40007",
                    signature_hash="100230c8c291505434a3d620df611b67ef5df41a1beca0c7e4d6c127be9cde9d",
                ),
            ),
            contract=(
                "The upstream DeepseekCompressor.forward / insert_cache run as "
                "is; the two kernel names they call are rebound. State save + "
                "softmax pooling + RMSNorm write the BF16 latent of every group a"
                " slotted token closes into latent_out (state cache via the state"
                " metadata for ratio 2, the k-cache slot mapping for ratio 1); "
                "then RoPE + NVFP4/MXFP8 record insert into the uint8 compressed "
                "cache (no fp8_scale). Re-check: argument lists, state cache "
                "semantics, pooling window/ratio, norm precision, record format."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_compress_save_norm",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_compress_rope_insert",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_compress_*.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.attention.sparse_mla",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Attention.forward_mqa",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention.forward_mqa",
                    source_hash="6473933118ae28419562712480c6688a6ee7e34b1bec37523812173eaf040839",
                    signature_hash="2602ae9277be6acda599b04acbc171526a07c620ea9d6a193296f51c48e27952",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._compressed_kv_cache",
                    source_hash="639f31499372857a5b94ae89f356e30f4203758341fb3aa941a2c5bc529433bf",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn:DeepseekV4MegaAttnAttention.forward_mqa",
                    source_hash="666e650ce79aea518ffef892aecf205d3b294176b1c0aa1ace55dcc563d2837c",
                    signature_hash="e9050e14ae8aa8253592e2f2fde1a5167bdb341a23a75e0f0182e642f4877f64",
                ),
            ),
            contract=(
                "Sparse MLA over the SWA window and the indexer-selected "
                "compressed entries, with the attention sink and the inverse RoPE"
                " fused; decode and prefill rows are separate launches with their"
                " own SWA index lists and step metadata; rows past the step are "
                "zeroed. Global top-k indices are mapped through the compressed "
                "block table. Re-check: index/len semantics, sink, softmax scale,"
                " which rows are decode, output padding."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_sparse_mla",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_compute_global_topk_indices_and_lens",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_sparse_mla.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.attention.o_proj",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Attention._o_proj",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._o_proj",
                    source_hash="1990605073df409e9620d0e1c1c55f41db93950def84c38c610dec206b91cee8",
                    signature_hash="4ce8d294b648f6c0fd3ab9dd64d424f167d303df2925e5d7bd2183f97892a691",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._wo_b_proj",
                    source_hash="12c194c3532d75e3469591223a60566960f8124fc32c73b0171e258c26c7d65a",
                    signature_hash="6e05affbd4c64aa537fa31e70f46ed8c2bcbd5b32828a21761335feaf389ed18",
                ),
            ),
            contract=(
                "Grouped wo_a (einsum tgd,grd->tgr over n_local_groups; W8A16 "
                "MXFP8 or INT8 UE8M0 packed, or BF16 fallback) then _wo_b_proj. "
                "Re-check: group layout, head slicing, wo_b reduction."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops.py::einsum_fp8_tgd_grd_to_tgr",
                "torch_xcpu/torch_xcpu/ops.py::einsum_int8_tgd_grd_to_tgr",
            ),
            tests=("docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",),
        ),
        Patch(
            id="deepseek_v41.attention.indexer",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Attention._indexer_cls",
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Indexer._produce_k",
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Indexer.forward_q",
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41Indexer.forward",
                "vllm_xcpu_plugin.models.deepseek_v41.attention:XcpuDeepseekV41IndexerOp.forward",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Attention._indexer_cls",
                    source_hash="8ba5687245d81d7734d7acc2e5caa88dfe76a1d102b4a59c99e9105c564b0249",
                    signature_hash="e84996dbc10c662a1bc7959afa57ae376092c4231cc8ae6da8ad2df236921aff",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Indexer.__init__",
                    source_hash="095caeebf56d83e2ea5409a77ffce4281dbcbed4a6542c9e4a8881017608b389",
                    signature_hash="9a3f3f7231531fd3d227f8a2087a7cf00c09e79faaf38a22471dc4928d5bbf51",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Indexer._produce_k",
                    source_hash="0adb14f6caf786422a79c208e753a52ae938eda3b598ddfb0294fe59e27b5dcd",
                    signature_hash="1c3e4df2582e78973032c2aac0c7e4a491bb7f30fa7e16b363a7b30023a79190",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Indexer.forward_q",
                    source_hash="a157f95b2e397c412b676270ed1c1e187b3107bd0ceefb4fcd37c935e4c0d500",
                    signature_hash="e533a9dcb6ce998bd8376f365e834943d03edb40517dc0dd036e2bd49b7df450",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Indexer.forward",
                    source_hash="e823ec911e02015c6b41f614b71186f28a8b4b04e589ea3208085b7c70c33503",
                    signature_hash="d8c01e77f4e5217728069b541e23759b89c84b00ebe7bdc81c154344ee10dc5a",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Indexer._wq_b_proj",
                    source_hash="9171bc9b6a9ccd092f8eace18a6f72e6c2b56d7a435c55ca379e73a76a60e69f",
                    signature_hash="c8b41db18639df77a4e8589e528c3c16a2f6c51c39e2d133e7305e11324133ed",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4Indexer.attn_cls",
                    expect="vllm.model_executor.layers.sparse_attn_indexer:SparseAttnIndexer",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.ops.indexer_k_store:indexer_k_norm_rope_store",
                    source_hash="3cb98160a488a5c76c7ed3fe7f06f22b585f012d6dcddd086a1c9fd6d7e2b406",
                    signature_hash="cc1df0c78ed409068fef93d3c02df64ade2957853edbafed10dfb40892974ef6",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v4.common.ops.fused_indexer_q:fused_indexer_q_rope_quant",
                    source_hash="938c0b9a60a56d2d97cb2210c4aeff5b0e5c9883db96c40d2e7ca4e21c495a06",
                    signature_hash="f9a52e3d4980835e775b009c2a94b329955d60fd9aa7d8492769af9705ed5bc6",
                ),
                UpstreamRef(
                    target="vllm.model_executor.kernels.attention.dsa.candidate_blocks:select_candidate_blocks",
                    source_hash="20ab4278116135c9b293e0ae1d1d5229a8a88630e1eb436f95f3d7772f7617c3",
                    signature_hash="20552f205f9ab58f18bbfd39ea9b12131f1150242338af6495b8fd68e252056b",
                ),
            ),
            contract=(
                "Lightning indexer: the K owner writes norm+RoPE MXFP4 keys "
                "before scoring; q gets RoPE in BF16 (no FP4 Q) with weights "
                "scaled by softmax_scale * n_head^-0.5; scores + top-k per token "
                "straight off the paged cache for prefill and decode alike, "
                "chunked by the logits budget; candidate source layers write "
                "blocks, consumers share one metadata per step. Short contexts "
                "(max_seq_len // ratio <= topk) fill all candidates. Re-check: "
                "scaling, candidate block size/flags, top-k tie/order, "
                "short-context rule, the SparseAttnIndexer constructor arguments "
                "the indexer passes."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_indexer_topk",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_indexer_q_rope",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_indexer_k_norm_rope_store",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_fill_short_context_topk",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_indexer_*.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.backends.swa",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41SWAMetadataBuilder.__init__",
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41SWAMetadataBuilder.build_tile_scheduler",
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41SWAMetadataBuilder.build",
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41SWABackend.get_name",
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41SWABackend.get_builder_cls",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.sparse_mla:DeepseekV41SparseSWAMetadataBuilder.__init__",
                    source_hash="00b0ec21310e6a67b1f4d29da9fb58b19eaafad74283220e64535e5c6874ee42",
                    signature_hash="3037c3ba3a932c5f37a4b4d7a744f12737b140c3a1b9a5db95bf35362400046f",
                ),
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.sparse_swa:DeepseekSparseSWAMetadataBuilder.__init__",
                    source_hash="25f548526a11936ac32b0b1c1e12e4b89cd5bfc4cdc68c32c51048919301b59a",
                    signature_hash="3037c3ba3a932c5f37a4b4d7a744f12737b140c3a1b9a5db95bf35362400046f",
                ),
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.sparse_swa:DeepseekSparseSWAMetadataBuilder.build",
                    source_hash="50678e14c55d827bc9fdc5f290ac6e8c2c505952678a22f894dbdf6f255f8c28",
                    signature_hash="94eb50ba000c11f1c7cbbcfc97bd5b4e89c62e7a3f3d653634ec0afdf04464bd",
                ),
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.sparse_swa:DeepseekSparseSWAMetadataBuilder.build_tile_scheduler",
                    source_hash="96d3fb5f4215aba8a133a0e26aed91265bdccc35469994415b2f61791d92c973",
                    signature_hash="231262944dd12bee0cfb99a85e95b8ead4efd751bbc14b8885ba4422359c1c35",
                ),
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.sparse_swa:DeepseekSparseSWABackend.get_name",
                    source_hash="686fc66a7bc1ce466b2bbaa8562f3dd24361dd16de503f855e3e3b26681f298f",
                    signature_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                ),
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.sparse_swa:DeepseekSparseSWABackend.get_builder_cls",
                    source_hash="bfe1f411aa43f902adef0600b36782ac23467026547bfaa6449c57ee52b45352",
                    signature_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                ),
            ),
            contract=(
                "Reuses the upstream SWA metadata (decode/prefill SWA index lists"
                " in paged coordinates, lens, token_to_req_indices, "
                "is_valid_token) and adds xcpu_sparse_mla step metadata per "
                "(compress_ratio, decode|prefill); the FlashMLA tile scheduler is"
                " replaced by empty entries. Widths: window or DSpark non-causal "
                "width for decode, prefill index width, plus index_topk for "
                "compressed layers. Re-check: metadata field names/meaning, width"
                " attributes, new build arguments, CUDA graph assumptions."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_sparse_mla_metadata",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_compute_swa_metadata",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_metadata_ops.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.backends.sparse_mla",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41SparseMLABackend.get_name",
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41SparseMLABackend.get_supported_kernel_block_sizes",
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41SparseMLABackend.get_builder_cls",
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41SparseMLABackend.supports_compute_capability",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.sparse_mla:DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes",
                    source_hash="278e5bb9e24bfa246945214f2de2d6c02f7c0f0b86eeb2fbd9c752611c7318f6",
                    signature_hash="a2030ab2008839cab466a7af931752b0ab4a121d3132120ee49e42a6deab7d8d",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.sparse_mla:DeepseekV4SparseMLABackend.get_builder_cls",
                    source_hash="f2e45e8bbbe067e2c66f344b501ca1fc1678fe7f6b4ad2bbe538455f0ed2bafd",
                    signature_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.sparse_mla:DeepseekV4SparseMLABackend.supports_compute_capability",
                    source_hash="ea34add3bfa1484f24b1bc49e6b979637d8935c5bea4ae2b29916bde31ac0319",
                    signature_hash="56a33d2fe5706714b5d182f1d588e34aab59a0ecf7ddac142e21bcc25193e799",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.sparse_mla:DeepseekV4SparseMLAMetadataBuilder.build",
                    source_hash="f3cd6af1ce3e0519f20121054ae51f678c5eb74550fee3e2c6faa24630d66654",
                    signature_hash="9d4f5c110f7433149d1af616a33a3415d877c9c5991b9d079fed64fd223ff915",
                ),
            ),
            contract=(
                "Compressed-KV backend: same metadata as FLASHMLA_SPARSE_DSV41, "
                "128-token pages holding block_size // compress_ratio entries, no"
                " CUDA graphs. Re-check: page size constraints, metadata fields "
                "consumed by forward_mqa (block_table)."
            ),
            tests=("docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",),
        ),
        Patch(
            id="deepseek_v41.backends.indexer",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41IndexerMetadataBuilder.__init__",
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41IndexerMetadataBuilder.build",
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41IndexerBackend.get_name",
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41IndexerBackend.get_supported_kernel_block_sizes",
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41IndexerBackend.get_builder_cls",
                "vllm_xcpu_plugin.models.deepseek_v41.backends:XcpuDeepseekV41IndexerCache.get_attn_backend",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.indexer:DeepseekV41IndexerBackend.get_name",
                    source_hash="60adb8c2ec3b5947d718fc669c0ac129cac21315b27ed9db1733b0bd7bdc7c06",
                    signature_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                ),
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.indexer:DeepseekV41IndexerBackend.get_supported_kernel_block_sizes",
                    source_hash="278e5bb9e24bfa246945214f2de2d6c02f7c0f0b86eeb2fbd9c752611c7318f6",
                    signature_hash="a2030ab2008839cab466a7af931752b0ab4a121d3132120ee49e42a6deab7d8d",
                ),
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.indexer:DeepseekV32IndexerBackend.get_builder_cls",
                    source_hash="6910e221b52ead2e4367a8bc274e6ff7ca52e9d144f67c14cacb4acac19d60ed",
                    signature_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                ),
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.indexer:DeepseekV32IndexerMetadataBuilder.build",
                    source_hash="62f1065265356dedcf7fcdfd649cbce10cce7dccf74983f67a1f87a162dd7002",
                    signature_hash="9d4f5c110f7433149d1af616a33a3415d877c9c5991b9d079fed64fd223ff915",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4IndexerCache",
                    source_hash="598bc88638aea4dacc0a8170af277ad7729212d54ae86ccf9aef5569d438c11d",
                    signature_hash="2e3340e65247480abdd99151760bcc4f1d72187fb20b30c58af5ebc5f881d1ad",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.attention:DeepseekV4IndexerCache.get_attn_backend",
                    source_hash="67977471d38fd379b7db23c8c455834022deeb90b2e634cf2491217f4f152924",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.compressor_utils:get_compressed_slot_mapping",
                    source_hash="70f9aaafd52697db2d84e4a0eeeb37a294d2c7a64c8d241bbad7588176fe4704",
                    signature_hash="1fb1ca05f88f6f80ceed95f4973a2adac7ccffd5a089a09f1cbdbdaff4e82b63",
                ),
                UpstreamRef(
                    target="vllm.v1.attention.backend:AttentionMetadataBuilder._init_reorder_batch_threshold",
                    source_hash="255fc9009e2198ad74ac571dc988e2a1e68b93c079f8a9b679610d2b61c5eb80",
                    signature_hash="542bda1d4becd95ac8e281901fde08a79afa8bfd23814815ec8f00351237ed8f",
                ),
                UpstreamRef(
                    target="vllm.v1.attention.backends.utils:split_decodes_and_prefills",
                    source_hash="21b78ee405f260ca1faf720cac023103e278d378e45b2610f35c99bc265820e1",
                    signature_hash="441a21b245c30963704e4bcc4aa2a1aa4c994b575954c3318efa2b3d045b08e5",
                ),
            ),
            contract=(
                "Own indexer metadata: per-token paged indexer over slot mapping,"
                " block table and token->request map, with row chunks sized by "
                "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB; ratio 2 takes the upstream "
                "get_compressed_slot_mapping (as "
                "DeepseekV32IndexerMetadataBuilder does, through the "
                "CompressedSlotMappingKernel adapter). Pages are 128 tokens "
                "(checked). The DeepseekV4IndexerCache name is rebound to the "
                "XCPU subclass, which reports this backend. Re-check: compressed "
                "slot rule and its arguments, decode threshold / spec-as-decode, "
                "fields the upstream builder adds, the cache constructor."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_indexer_metadata",
            ),
            tests=("docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",),
        ),
        Patch(
            id="deepseek_v41.engram.hash",
            source=_ENGRAM,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuNgramHashState.forward",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:NgramHashState.forward",
                    source_hash="a2622e3c0f1a8d323525a009866aceb3e9f59e35d914601951b7db7c1ee0400a",
                    signature_hash="dde1d2a5a1575316e7851f1b72d87f2da5e7a58313a35d80823e82f7ed7e1998",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:NgramHashState.__init__",
                    source_hash="d2b26de057f0e365eccd7550c030943994d9a2fc1fa36bebf790862863715a29",
                    signature_hash="3bfa25d206318e4f81a65790561b5386129227e1ed272e6adac13be82e570349",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:NgramHashState.ensure_cache",
                    source_hash="b14726ea8fb306dc334500bd5618b7011e228a650c40a4f70c6daebbb99d75e5",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:_write_hash_cache_kernel",
                    source_hash="5ade07d356a4aa06cee793eb86d000c53b13461f07c97588bccf7d6761f4f76c",
                    signature_hash="47a195a3d98cefb15c2f0123389523d716bec2ab8c2c07dcc3cb2676a14308c9",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:_hash_ids_kernel",
                    source_hash="113958d8fbd5c0d16fa47a5a9529f5c19d16f23e988dca7a2b9d7316e71d1c91",
                    signature_hash="138895928a31270daeb27248d1a9588e79c1ae7988385381c61e7410cfb5a362",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:DEAD_ID",
                    source_hash="1bad6b8cf97131fceab8543e81f7757195fbb1d36b376ee994ad1cf17699c464",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model.forward",
                    source_hash="24aa409a437ae7c6da1e219af1c6c0ab7d36a8da32e9ccefd97eb5b022a6e1dc",
                    signature_hash="7c73aeee0e17a617b3d6038ae156bf93af563da7931df94603f79ffab05c87c3",
                ),
            ),
            contract=(
                "[T, layers, (max_ngram-1)*heads] int32 n-gram hashes in one "
                "fused op: on the V1 runner the batch's compressed ids are "
                "written to the slot-keyed cache before any lookback reads it; "
                "history comes from the chunk, then lookback_token_ids, then the "
                "slot cache; DEAD_ID (-1) breaks n-grams, pad_id fills before the"
                " sequence start. Re-check: history precedence, cache keying, "
                "hash formula (multipliers, primes, offsets), output layout, "
                "dead/pad handling."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_engram_hash",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_engram_hash.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.engram.lookup",
            source=_ENGRAM,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuParallelEngramEmbedding.lookup",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:ParallelEngramEmbedding.lookup",
                    source_hash="a323841a2c62c093e339fe9d29d13f4890354500f797809fbd3739a9b5ed0113",
                    signature_hash="38c76af81009e7438a4c092528cc55af5b1d7bd1af5f1c23aff8e795b5cd3aa8",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:ParallelEngramEmbedding.__init__",
                    source_hash="bfcffab420a3e3f17bb61193d924374c4ea0f4808b34afe94fa3f36595b204b0",
                    signature_hash="9e155dd9685c456547b8ff806e0c2e061ae933181230422c260bf80e704965e7",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:ParallelEngramEmbedding._storage",
                    source_hash="42e11ae89724c1c9bdece92c1b72fd8d17f98e0a348ee877fe5f8c4a690b5c96",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:_engram_lookup_kernel",
                    source_hash="35b43f6397681806746837ea97796342113a3274a88575fcdb57f71cf5d1fa0d",
                    signature_hash="08d8e02008b8e77fb0de84618f7835b724d325a16107771cf8914901bc14286f",
                ),
            ),
            contract=(
                "Looks up this rank's head buckets [head_start, head_start + "
                "part_n_hash_cols) of [T, n_hash_cols] ids into [T, local_heads, "
                "dim] BF16: FP8 rows times UE8M0 block scales, ids outside "
                "[vocab_start, vocab_end) give zero rows. The table is a device "
                "tensor (no offload, so no sort/background). Re-check: sharding "
                "of heads and rows, scale format, out-of-range rule, out layout."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_engram_lookup",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_engram_lookup.py",
                "docs/upgrade/12-dsv41-dp2tp2-ep.md (SP, DP-sharded engram)",
            ),
        ),
        Patch(
            id="deepseek_v41.engram.embed",
            source=_ENGRAM,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuEngram.embed",
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuEngram._fill_rows",
                "vllm_xcpu_plugin.models.deepseek_v41.engram:_xcpu_deepseek_v41_engram_embed",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram.embed",
                    source_hash="096d87da2e2fc292816b4532edbf1f43f349ab5dae977bb7868069dc1397c03f",
                    signature_hash="635bcc5d9cad7e4ff6b0206fd3f90c0480e89c878a3c443ea5a583d919baabab",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram._ready_rows",
                    source_hash="55c5273007c1a22fd1ce61e5224c5a652261acf3091dc21b0ac0fc227570a577",
                    signature_hash="70d119e18a5b2782fcd94897d8123e5e4f228dd290cfeb62f42ed9bd07267ab5",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram.prepare_embeddings",
                    source_hash="e5ddf2233511f32f11ca6efd99a938e5041036ea4d3d94cc4899fc80f88968c6",
                    signature_hash="635bcc5d9cad7e4ff6b0206fd3f90c0480e89c878a3c443ea5a583d919baabab",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:_gather_engram_rows",
                    source_hash="f42adadcf75e2b425e5df0a8ddd1da1f7b16162bbe50b367a57c3e45dd60749c",
                    signature_hash="917cce6a8d7d0e91cefa46d2f0fe1986aa45dc48878b789f800012a8be10f34e",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:engram_gathered_num_tokens",
                    source_hash="c3d55d49100199e396233789399aae061a4e839c1760065d39c01f670c7d65c2",
                    signature_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                ),
            ),
            contract=(
                "Under torch.compile Engram.embed runs inside the opaque op "
                "vllm::xcpu_deepseek_v41_engram_embed (DP-sharded tables exchange"
                " staged rows using the forward context's DP token slot, which a "
                "graph must not bake in); the op's impl is XcpuEngram._fill_rows, "
                "which writes directly into out via deepseek_v41_engram_select_rows"
                " (or copy_ when not SP) or, with Engram "
                "disaggregation (deepseek_v41.engram.remote), decodes the remote "
                "rows in eager and compile alike. The op's output is allocated in "
                "the graph as [ceil(T/tp) if SP else T, n_hash_cols, dim] BF16 and "
                "must equal Engram.embed's result shape. Re-check: embed's output"
                " shape/dtype, new forward-context reads, staging buffer "
                "lifetime."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_engram_select_rows",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_metadata_ops.py",
                "docs/upgrade/12-dsv41-dp2tp2-ep.md (SP, DP-sharded engram)",
            ),
        ),
        Patch(
            id="deepseek_v41.engram.post_wkv",
            source=_ENGRAM,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuEngram._post_wkv",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram.forward",
                    source_hash="4ffb0b91a0df2585f42bf5abfa97cbda138737b46bf2e79d75047b801beab818",
                    signature_hash="f3d5597443acc5b4c16744830dc2cfc59b8f49d15175bc11332f38bb6e407f97",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram._local_kv_tokens",
                    source_hash="85f5c23034904b5731c081969f8d5a0f577285d468929b91229303bfaa4e6bf3",
                    signature_hash="3e174853e2b5a4fc65339c92725d36fca88e004ee852119b139038555321d759",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram._post_wkv",
                    source_hash="4505ff17e125eba286d966554086b8c89bbbb3c3127f9a6a4c196e5a9dcc115c",
                    signature_hash="596787ef0502a388fa801da4e7d94abf783163cae1db365058d3ad7dd0d3f468",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:_fused_engram_post_wkv_kernel",
                    source_hash="d53f6ccaf11b2fcfb6d28629402edd860cc30646761afc77f542f7c0f6b41d0c",
                    signature_hash="02b5ff8607868ce6c0130b5f2d80e388d498f3093a3b12e1202132a2ca6e6369",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4DecoderLayer.forward",
                    source_hash="8a1b1f3b48d0d281a1ca2cbcd431909fffb5be1d5d668c53eb929bceb47a966b",
                    signature_hash="5978820871604b84cdf691be6fafa70c1c2abebce513ff6e75784e6f4fe2d53f",
                ),
            ),
            contract=(
                "Gate of each hc stream: normalized dot of the stream with its "
                "key (q_weight/k_weight RMS-style norms, clamp 1e-6), signed "
                "sqrt, sigmoid; adds gate * shared value. Rows at or past "
                "num_kv_tokens (SP shard padding) and False token_mask rows pass "
                "through. num_kv_tokens is a SymInt in compiled graphs, so no "
                "shape asserts. Re-check: gate formula, which rows get kv, mask "
                "polarity, eps/clamp."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_engram_post_wkv",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_engram_post_wkv.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
                "docs/upgrade/12-dsv41-dp2tp2-ep.md (SP, DP-sharded engram)",
            ),
        ),
        Patch(
            id="deepseek_v41.engram.layer_registration",
            source=_ENGRAM,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuEngram.__init__",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram.__init__",
                    source_hash="5f0a0ab3985a27821e2adeba4c360cfab674fe345a2095f9a89a10fe2bdd9d0c",
                    signature_hash="fd6c9bba29099ca63b82093ca293fa3c8c722c07558dc26338c4cba98672d28a",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram._create_embedding",
                    source_hash="eac41d3628e73a2ab298961cba28fb6eb8ed3bb0a242ecfb6178f293729bda09",
                    signature_hash="d2c60c3cf954f6a1545bdf4930a29adfe6e711073892317a7bfe6913b8c797fd",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram._init_staging",
                    source_hash="b29bfb924a88e9f9ca7f8ff30ba9be0f6e92f176151e02efdf5b7df53189ce4e",
                    signature_hash="dee731e4e979a7dc14376ee36e25d2a6a70b255c48f98a15dc018902e5a17ac7",
                ),
            ),
            contract=(
                "XcpuEngram replaces Engram through PluggableLayer.register_oot "
                "and registers itself in static_forward_context under its prefix "
                "for the opaque embed op. Re-check: constructor signature (prefix"
                " keyword), staging allocation, embedding construction."
            ),
            tests=("docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",),
        ),
        Patch(
            id="deepseek_v41.engram.remote",
            source=_ENGRAM,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.model:_gather_engram_hashes",
                "vllm_xcpu_plugin.models.deepseek_v41.model:_get_engram_dp_size",
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuParallelEngramEmbedding.__init__",
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuParallelEngramEmbedding._get_shard_info",
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuParallelEngramEmbedding._allocate_weights",
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuEngram._create_embedding",
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuEngram._init_staging",
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuEngram.prepare_embeddings",
                "vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuEngram._fill_rows",
                "vllm_xcpu_plugin.engram_disagg.attn.client:RemoteEngramClient",
                "vllm_xcpu_plugin.engram_disagg.attn.model_loader:EngramAttentionModelLoader",
                "vllm_xcpu_plugin.engram_disagg.common:EngramShardLayout",
                "vllm_xcpu_plugin.engram_parallel.client:EngramParallelClient",
                "vllm_xcpu_plugin.engram_parallel.model_loader:EngramParallelModelLoader",
                "vllm_xcpu_plugin.engram_parallel.layout:EngramParallelLayout",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:gather_engram_hashes",
                    expect="vllm.models.deepseek_v41.common.engram:gather_engram_hashes",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:get_engram_dp_size",
                    expect="vllm.distributed.parallel_state:get_engram_dp_size",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:gather_engram_hashes",
                    source_hash="21daa026b191221c1d481ee125ca08a3a1cbe931729c839744a90ab23846250b",
                    signature_hash="a325571fa5ec51c53b3c03ec612de949bff785ec581dc9c19c8874b56dfb62ba",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:NgramHashState.dummy_hashes",
                    source_hash="da7c9d9ee8d789e231bdcfceaaa1c7d1fb08f848cb2e4579ca719a994586c605",
                    signature_hash="8ea60dfd46731fa6898943145481862f2a4136af57a2c60af2eb5cef1fa4bb53",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:EngramLayout.__init__",
                    source_hash="6ef54c9198d26e113271c14d1b72bff56e9dc383bbaf83b71cccc42fa71158a8",
                    signature_hash="7cb76bfb394aa0cc3337c6ebae2c28058821c1740333dfa9d86d43d98cef6b75",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:ParallelEngramEmbedding.__init__",
                    source_hash="bfcffab420a3e3f17bb61193d924374c4ea0f4808b34afe94fa3f36595b204b0",
                    signature_hash="9e155dd9685c456547b8ff806e0c2e061ae933181230422c260bf80e704965e7",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:ParallelEngramEmbedding._get_shard_info",
                    source_hash="145b91e67a036384c327dbd49be99b35e999fe4ecda03a1b42baa041bc55bc22",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:ParallelEngramEmbedding._allocate_weights",
                    source_hash="7ae3df14f95f855b7722c877851cebdf2667cbb3ca641e672d10005848e75f99",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:_engram_head_shard_weight_loader",
                    source_hash="eb371335ecb24390c8a642201b9300f7902cd2b41fd52a5b1485dd16d4361a45",
                    signature_hash="3b8fb179f3ef6294d6fb814577afbbc0db4337b596fd42364921c12726a6a18e",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram.__init__",
                    source_hash="5f0a0ab3985a27821e2adeba4c360cfab674fe345a2095f9a89a10fe2bdd9d0c",
                    signature_hash="fd6c9bba29099ca63b82093ca293fa3c8c722c07558dc26338c4cba98672d28a",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram._create_embedding",
                    source_hash="eac41d3628e73a2ab298961cba28fb6eb8ed3bb0a242ecfb6178f293729bda09",
                    signature_hash="d2c60c3cf954f6a1545bdf4930a29adfe6e711073892317a7bfe6913b8c797fd",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram._init_staging",
                    source_hash="b29bfb924a88e9f9ca7f8ff30ba9be0f6e92f176151e02efdf5b7df53189ce4e",
                    signature_hash="dee731e4e979a7dc14376ee36e25d2a6a70b255c48f98a15dc018902e5a17ac7",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram.prepare_embeddings",
                    source_hash="e5ddf2233511f32f11ca6efd99a938e5041036ea4d3d94cc4899fc80f88968c6",
                    signature_hash="635bcc5d9cad7e4ff6b0206fd3f90c0480e89c878a3c443ea5a583d919baabab",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:Engram._local_kv_tokens",
                    source_hash="85f5c23034904b5731c081969f8d5a0f577285d468929b91229303bfaa4e6bf3",
                    signature_hash="3e174853e2b5a4fc65339c92725d36fca88e004ee852119b139038555321d759",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model.forward",
                    source_hash="24aa409a437ae7c6da1e219af1c6c0ab7d36a8da32e9ccefd97eb5b022a6e1dc",
                    signature_hash="7c73aeee0e17a617b3d6038ae156bf93af563da7931df94603f79ffab05c87c3",
                ),
                UpstreamRef(
                    target="vllm.model_executor.model_loader.default_loader:DefaultModelLoader.get_all_weights",
                    source_hash="4047d24ba68dd2e0e4779319f87ee1e2ce25899f014246c9d8759a4b4fb662e0",
                    signature_hash="925a6be0c5912b150b2cd6e47bd5655c93c556411127b369bec5a122e3d1cc05",
                ),
                UpstreamRef(
                    target="vllm.model_executor.model_loader.default_loader:DefaultModelLoader._get_weights_iterator",
                    source_hash="fd976676c02d33b9442dd84e83fa1d2bfd69cf9660b97eeed4a5f731de7e9550",
                    signature_hash="976275b36860318cd0eaaa620727a4bc3c8a2f0ef9766707af347989b2dff7e9",
                ),
                UpstreamRef(
                    target="vllm.model_executor.model_loader.default_loader:DefaultModelLoader._prepare_weights",
                    source_hash="c5ccb7b24500c4f91585f42471d4f6ad0bfc6d514522bb60c86454805d9df4ce",
                    signature_hash="e3d6c58df8e7715c3e5e73b50fe3927755a77fffbb9378f2432d659865d2e445",
                ),
                UpstreamRef(
                    target="vllm.model_executor.model_loader.weight_utils:safetensors_weights_iterator",
                    source_hash="061bb65f7bdf0d6b5796bd47794ec375537675450597337245d24dc765f2bf41",
                    signature_hash="d159df39d1c847ed24167463f38d4964ec27e2e2334d397bd2f33441937bc27b",
                ),
                UpstreamRef(
                    target="vllm.model_executor.model_loader.ep_weight_filter:should_skip_weight",
                    source_hash="fb7a5a121254ca4762650c93721a465011b5681f05f16cf65fbb0361643b5701",
                    signature_hash="4daaca1beef199835162c11d8cf86341b743bd118b9e5e4b6a80a80048ec0a12",
                ),
            ),
            contract=(
                "Engram distributed sharding (VLLM_XCPU_ENABLE_ENGRAM_DISAGG or "
                "VLLM_XCPU_ENABLE_ENGRAM_PARALLEL): under disaggregation the tables "
                "live on dedicated Engram MPI ranks and the attention rank's "
                "ParallelEngramEmbedding holds zero-row parameters; under "
                "EngramParallel the tables are 2D-sharded across all DP x TP ranks "
                "and each rank holds its [lo, hi) row slice per layer, loaded via "
                "EngramParallelModelLoader (safe_open slice). Per-step "
                "MPI_Ialltoallv requests size each peer slot dynamically from "
                "ForwardContext.dp_metadata.num_tokens_across_dp_cpu when "
                "data_parallel_size > 1 (rejecting skip_dp_coordination). Both "
                "modes bypass "
                "upstream column sharding (shard info (1, 0)) and keep "
                "Engram._init_staging/prepare_embeddings as no-ops. "
                "Exactly one request per forward: DeepseekV4Model.forward hashes "
                "the full token stream before sp_shard and calls "
                "gather_engram_hashes once, also for forwards without attention "
                "metadata via the get_engram_dp_size() > 1 dummy_hashes branch "
                "(the rebound get_engram_dp_size reports 2); the compiled "
                "_run_layers dummy branch requests too. Each engram layer's embed "
                "then fills [ceil(T/tp) if SP else T, n_hash_cols, dim] rows for "
                "the tokens of Engram._local_kv_tokens, layer index 0 collecting "
                "the reply first. Relies on EngramLayout.primes order for the "
                "column -> rank mapping and on dummy_hashes using DEAD ids "
                "(zero rows). Re-check: the forward's hash/gather/prepare order "
                "and its dummy branch condition, _local_kv_tokens' shard formula, "
                "ParallelEngramEmbedding.__init__'s use of _get_shard_info / "
                "_allocate_weights / _weight_loader, safetensors iterator "
                "signature, should_skip_weight's name pattern and its "
                "scale-tensor exception, checkpoint tensor names of the tables."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_engram_remote_request",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_engram_remote_decode",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_engram_serve",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_engram_parallel_request",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_engram_parallel_collect",
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_engram_parallel_decode",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_engram_remote_decode.py",
                "torch_xcpu/test/mpi_custom_ops/test_engram_remote_roundtrip_mpi.py",
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_engram_parallel.py",
                "torch_xcpu/test/mpi_custom_ops/test_engram_parallel_roundtrip_mpi.py",
                "vllm-xcpu-plugin/tests/engram_disagg",
                "vllm-xcpu-plugin/tests/engram_parallel",
                "vllm-xcpu-plugin/tests/mpi/engram_parallel_model_mpi.py",
                "docs/upgrade/13-dsv41-engram-disagg.md",
                "docs/upgrade/14-dsv41-engram-parallel.md",
            ),
        ),
        Patch(
            id="deepseek_v41.engram.no_offload",
            source=_ENGRAM,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.config:update_deepseek_v41_config",
                "vllm_xcpu_plugin.models.deepseek_v41.config:_update_engram_parallel_config",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.config:EngramConfig",
                    expect="vllm.config.engram:EngramConfig",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:ParallelEngramEmbedding._allocate_weights",
                    source_hash="7ae3df14f95f855b7722c877851cebdf2667cbb3ca641e672d10005848e75f99",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
            ),
            contract=(
                "MCPU device memory already is host memory and has no UVA view, "
                "so the platform hook forces engram_config.cpu_offload, "
                "dp_shared_memory and use_thp off; the table stays an ordinary "
                "device tensor sharded over the node-local engram DP group. "
                "Re-check: new EngramConfig fields that assume CUDA host "
                "mappings, allocation paths."
            ),
            tests=("docs/upgrade/12-dsv41-dp2tp2-ep.md (SP, DP-sharded engram)",),
        ),
        Patch(
            id="deepseek_v41.metadata.compute_swa_indices_and_lens",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.triton_adapters:_compute_swa_indices_and_lens",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.sparse_swa:_compute_swa_indices_and_lens_kernel",
                    source_hash="f53940d12d8e66b71e00d0c8321bc100c9d057559f80bfd172a30e92d2209b2e",
                    signature_hash="ccd1db6e5a55665240645dfdb2fdd73af3faaaeb121caedbbca2b67b3f02ed04",
                ),
            ),
            contract=(
                "Fake Triton adapter: routes the kernel to "
                "torch_xcpu.ops.deepseek_v41_compute_swa_metadata on the launch "
                "stream and validates its launch arguments. Re-check "
                "the kernel body, every launch site, grid and constexprs "
                "(docs/TRITON_COMPATIBILITY_GUIDE.md 3.6)."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_compute_swa_metadata",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_metadata_ops.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.metadata.compute_dspark_noncausal_swa_indices",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.triton_adapters:_compute_dspark_noncausal_swa_indices",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.sparse_swa:ComputeDSparkNoncausalSWAIndicesKernel.kernel",
                    source_hash="c5d245e07d940f30c11079b91e3be75086c49b853eb06bf79085ad3db6e086fb",
                    signature_hash="e17c2cf8c919ec4259584c99658d8d5b2627b34fae15140dd8df031def957668",
                ),
            ),
            contract=(
                "Fake Triton adapter: routes the kernel to "
                "torch_xcpu.ops.deepseek_v41_compute_swa_metadata (non_causal=True)"
                " on the launch stream and validates its launch arguments. "
                "Re-check the kernel body, every launch site, grid and constexprs "
                "(docs/TRITON_COMPATIBILITY_GUIDE.md 3.6)."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_compute_swa_metadata",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_metadata_ops.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.metadata.compute_prefill_metadata",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.triton_adapters:_compute_prefill_metadata",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.sparse_swa:ComputePrefillMetadataKernel.kernel",
                    source_hash="02bb032558f058b35ebfd7d8c95eb2c08c81572ddf95ae4d46b76dd191cadcb8",
                    signature_hash="a2f53c18aeabc4ad65e7a74ff2443f7507850aed2d6ce66d2eae693880e159d8",
                ),
            ),
            contract=(
                "Fake Triton adapter: routes the kernel to "
                "torch_xcpu.ops.deepseek_v41_compute_prefill_metadata on the "
                "launch stream and validates its launch arguments. Re-check "
                "the kernel body, every launch site, grid and constexprs "
                "(docs/TRITON_COMPATIBILITY_GUIDE.md 3.6)."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_compute_prefill_metadata",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_metadata_ops.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.metadata.compressed_slot_mapping",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.triton_adapters:_compressed_slot_mapping",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.v1.attention.backends.mla.compressor_utils:CompressedSlotMappingKernel.kernel",
                    source_hash="44f6e974bcc801ced8e7f79b3ef4060687aedaab1b605c09fb9518587584bfce",
                    signature_hash="c1886f73275a8d5758d399688ba0cdfdd8372c734672edfdea8f7e63e82ee292",
                ),
            ),
            contract=(
                "Fake Triton adapter: routes the kernel to "
                "torch_xcpu.ops.deepseek_v41_compressed_slot_mapping on the "
                "launch stream and validates its launch arguments. Re-check "
                "the kernel body, every launch site, grid and constexprs "
                "(docs/TRITON_COMPATIBILITY_GUIDE.md 3.6)."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_compressed_slot_mapping",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_metadata_ops.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.metadata.ring_slot_mapping",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.triton_adapters:_ring_slot_mapping",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.compressor:_ring_slot_mapping_kernel",
                    source_hash="6335cc75a9aeb566b6019ed04fb771ae40727858a6aacc903b69513bd1fa54d7",
                    signature_hash="66dc54aa92d6f81ad8cc69356d8fe0eb058d9dd26c8fcbb23b3b9a2013165782",
                ),
            ),
            contract=(
                "Fake Triton adapter: routes the kernel to "
                "torch_xcpu.ops.deepseek_v41_ring_slot_mapping on the launch "
                "stream and validates its launch arguments. Re-check "
                "the kernel body, every launch site, grid and constexprs "
                "(docs/TRITON_COMPATIBILITY_GUIDE.md 3.6)."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_ring_slot_mapping",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_metadata_ops.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.metadata.gather_lookback",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.triton_adapters:_gather_lookback",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model_state:_gather_lookback_kernel",
                    source_hash="ead01b9268cd69d4f1afa6dd2e7fc135cd8dbf7f005f0bfa019a3794dc3d3562",
                    signature_hash="350b5bf1c66b757c18e08d99e08ce8ab51bd3a30c67c610302089241943ef892",
                ),
            ),
            contract=(
                "Fake Triton adapter: routes the kernel to "
                "torch_xcpu.ops.deepseek_v41_gather_lookback on the launch "
                "stream and validates its launch arguments. Re-check "
                "the kernel body, every launch site, grid and constexprs "
                "(docs/TRITON_COMPATIBILITY_GUIDE.md 3.6)."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_gather_lookback",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_metadata_ops.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.metadata.pad_replayed_slots",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.triton_adapters:_pad_replayed_slots",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model_state:_pad_replayed_slots_kernel",
                    source_hash="fafdf630e37825ed19c0fdd3691ca7457b4da7390a983500e25886338590e6f9",
                    signature_hash="9df248841db11dcf9cfc6da250231cd1688c26268fcc950032340754478f5619",
                ),
            ),
            contract=(
                "Fake Triton adapter: re-expresses the kernel with ATen ops on "
                "the launch stream and validates its launch arguments. Re-check "
                "the kernel body, every launch site, grid and constexprs "
                "(docs/TRITON_COMPATIBILITY_GUIDE.md 3.6)."
            ),
            tests=("docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",),
        ),
        Patch(
            id="deepseek_v41.metadata.gather_replay_batch",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.triton_adapters:_gather_replay_batch",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model_state:_gather_replay_batch_kernel",
                    source_hash="1864dfdd08cd19ef6a1b1b70400b0c0087beccd2ae4f1d54f34a299be0b001bd",
                    signature_hash="bab1c1ae014f6cd4bb8c5199dccfc5fa26fd1038b9053fea965bd5041905ea5c",
                ),
            ),
            contract=(
                "Fake Triton adapter: re-expresses the kernel with ATen ops on "
                "the launch stream and validates its launch arguments. Re-check "
                "the kernel body, every launch site, grid and constexprs "
                "(docs/TRITON_COMPATIBILITY_GUIDE.md 3.6)."
            ),
            tests=("docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",),
        ),
        Patch(
            id="deepseek_v41.metadata.engram_select_rows",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.triton_adapters:_engram_select_rows",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.common.engram:_engram_select_rows_kernel",
                    source_hash="52e43e4499a3308d769213685fd966c1d06a82f05230366f8bc824c2912ac161",
                    signature_hash="1a2683ab57bf65296dd5e3b576c76ecb88da557e7322825b9abb9e9de145fc07",
                ),
            ),
            contract=(
                "Copy one token window out of a rank-major gathered engram "
                "buffer: output[t, c] = gathered[c // LOCAL_WIDTH][token_start + "
                "t][c % LOCAL_WIDTH], zero past ``num_tokens``. Fake Triton "
                "adapter: routes the kernel to "
                "torch_xcpu.ops.deepseek_v41_engram_select_rows on the launch "
                "stream and validates its launch arguments. Re-check the kernel "
                "body, every launch site, grid and constexprs "
                "(docs/TRITON_COMPATIBILITY_GUIDE.md 3.6)."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_engram_select_rows",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/deepseek_v41/test_metadata_ops.py",
                "docs/upgrade/11-dsv41-serial.md (eager + compile + DSpark)",
            ),
        ),
        Patch(
            id="deepseek_v41.vl_model.streaming_load_weights",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.model:XcpuDeepseekV41ForCausalLM.load_weights",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.vl_model:DeepseekV41ForCausalLM.__init__",
                    source_hash="eee102dcf160a9a21e93fe6bbc857920e4d5c79276dfb46fc7bb34d995ad8ce1",
                    signature_hash="41e09ea153ede580447e3e22bd8e67225987f82aa5682a96a2ac3bb3d3bdba35",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.vl_model:DeepseekV41ForCausalLM.load_weights",
                    source_hash="996e63ebdf9e37f48163ebb32ee51fdcfb16acacc2b433ad7e058f86bb926057",
                    signature_hash="24bcf75dc8091fd46a88cb78817a9f1f45d8eff388d0d84c185e4f7f2c4424da",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV41LLMForCausalLM.load_weights",
                    source_hash="bc3a6bdc6bbc5cc77e88340c25413b8d8c55ff77dc397bffb18ee5e7d047f7ee",
                    signature_hash="24bcf75dc8091fd46a88cb78817a9f1f45d8eff388d0d84c185e4f7f2c4424da",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV4Model.load_weights",
                    source_hash="cc15fc0ddec18dd1d58abbf2980aa46e4b2d0af9232bb17cd82d7fed7d0fb31f",
                    signature_hash="24bcf75dc8091fd46a88cb78817a9f1f45d8eff388d0d84c185e4f7f2c4424da",
                ),
                UpstreamRef(
                    target="vllm.model_executor.models.utils:AutoWeightsLoader._load_module",
                    source_hash="783af64185df51d18546a46666189a7ee740744fd9047c9bb2cb6c629fa5b3e9",
                    signature_hash="53c04ff398a6fb3ce7ae410166e52ed12f0d653b8a08a844eb5247a3de0e698a",
                ),
            ),
            contract=(
                "DeepseekV41ForCausalLM.load_weights upstream materializes and "
                "sorts all mapped weights in memory so language_model.* forms "
                "one contiguous block for AutoWeightsLoader._load_module (calling"
                " DeepseekV41LLMForCausalLM.load_weights once). XCPU streams "
                "language_model.* directly without sorting/buffering, drops "
                "StageMissingLayer tower tensors, and clones+sorts only the "
                "small non-language_model tensors at the end, setting "
                "_weights_finalized=True. Re-check: new top-level modules or "
                "non-idempotent rules in hf_to_vllm_mapper, post-load "
                "finalization order in DeepseekV41LLMForCausalLM.load_weights."
            ),
            tests=("vllm-xcpu-plugin/tests/engram_parallel/test_layout_and_loader.py",),
        ),
        Patch(
            id="deepseek_v41.quant.linear_scale_param_name",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.models.deepseek_v41.model:_linear_scale_param_name",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:_linear_scale_param_name",
                    source_hash="2f6c4b5f50b3ed124cb59b7983cc4b59d71e3cac59103ad1012b42879c4851e4",
                    signature_hash="d14dbd05f16a8200539de0a3bd08ccec604c2dec8b0266121203a21b060bf101",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.vl_model:_linear_scale_param_name",
                    source_hash="2f6c4b5f50b3ed124cb59b7983cc4b59d71e3cac59103ad1012b42879c4851e4",
                    signature_hash="d14dbd05f16a8200539de0a3bd08ccec604c2dec8b0266121203a21b060bf101",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:_linear_scale_param_name",
                    source_hash="2f6c4b5f50b3ed124cb59b7983cc4b59d71e3cac59103ad1012b42879c4851e4",
                    signature_hash="d14dbd05f16a8200539de0a3bd08ccec604c2dec8b0266121203a21b060bf101",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.model:DeepseekV41LLMForCausalLM.__init__",
                    source_hash="54bf9a148b88ef4d5824068463591b15418a5e0f71f56cb3eb92b280a924402a",
                    signature_hash="0268aa001d71a5bab9e4761758b7b321b8822f20f4554d61286ec2ca4dd91920",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.vl_model:DeepseekV41ForCausalLM.__init__",
                    source_hash="eee102dcf160a9a21e93fe6bbc857920e4d5c79276dfb46fc7bb34d995ad8ce1",
                    signature_hash="41e09ea153ede580447e3e22bd8e67225987f82aa5682a96a2ac3bb3d3bdba35",
                ),
                UpstreamRef(
                    target="vllm.models.deepseek_v41.nvidia.dspark:DSparkDeepseekV4ForCausalLM.__init__",
                    source_hash="19d27a794c12036dc94d704054c31699c6d9589990a0ca476032cb3cddca7e8b",
                    signature_hash="0268aa001d71a5bab9e4761758b7b321b8822f20f4554d61286ec2ca4dd91920",
                ),
            ),
            contract=(
                "Upstream _linear_scale_param_name returns 'weight_scale' whenever "
                "weight_block_size == [32, 32] and expert_dtype == 'fp4' (assuming "
                "ModelOpt MXFP8). XcpuInt8Ue8m0LinearMethod always registers "
                "'weight_scale_inv', so XcpuInt8Ue8m0Config overrides all three "
                "call sites (model, vl_model, dspark) to 'weight_scale_inv' "
                "while delegating other quant configs upstream. Re-check: new "
                "call sites importing _linear_scale_param_name or scale suffix "
                "changes in _make_deepseek_v4_weights_mapper."
            ),
            impl=(
                "vllm-xcpu-plugin/vllm_xcpu_plugin/layers/int8_ue8m0.py::XcpuInt8Ue8m0LinearMethod",
            ),
            tests=("vllm-xcpu-plugin/tests/layers/test_int8_ue8m0.py",),
        ),
    ),
)
