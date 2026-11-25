# SPDX-License-Identifier: Apache-2.0
"""Upstream contracts of XCPU EPLB router, communicator, and layer hooks."""

from vllm_xcpu_plugin.fake_triton.provenance import KernelSource
from vllm_xcpu_plugin.upstream_contracts import (
    Manifest,
    Patch,
    UpstreamRef,
    WatchedModule,
)

_V031 = KernelSource(
    tag="v0.31.0",
    local_commit="3f5fbba1c3027469e80def9878eefc80b38ec9c4",
    main_commit="283d76d56a45d9bbead5ba650402fb0957abc30c",
)

MANIFEST = Manifest(
    category="eplb",
    packages=(
        "vllm_xcpu_plugin.eplb",
        "vllm_xcpu_plugin.distributed.mpi_eplb_communicator",
    ),
    watched=(
        WatchedModule(
            module="vllm.model_executor.layers.fused_moe.router.base_router",
            sha256="c909924fcfa461ccdaa6290d215d4d35bd4062fb02feeb23619b234da993f6bf",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.model_executor.layers.fused_moe.routed_experts",
            sha256="a532b01b38ac508ab7f8db84ea5b373cadb736917a668091585da5a5419b3a86",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.distributed.eplb.rebalance_execute",
            sha256="53af2d0429813c44256bae99d91177437ae492802eca284c18f1dd65703dbad9",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.distributed.eplb.eplb_utils",
            sha256="f3215b17d6073b77079e9b0369404153d68753e1d86a1bec296c43ee7c8cc57a",
            source=_V031,
        ),
        WatchedModule(
            module="vllm.distributed.eplb.async_worker",
            sha256="b1b10db5181fb5a704c8656a6a4a35af7e79b8fb4721183b8f2548fb72c5b042",
            source=_V031,
        ),
    ),
    patches=(
        Patch(
            id="eplb.router.map_to_physical",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.eplb.patch:_xcpu_eplb_map_to_physical_and_record",
                "vllm_xcpu_plugin.eplb.patch:install_eplb",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.router.base_router:eplb_map_to_physical_and_record",
                    expect="vllm.model_executor.layers.fused_moe.router.base_router:eplb_map_to_physical_and_record",
                    signature_hash="754e02547928ccf23f2437ece90929022f61c63add1530d449a425111d1c29ae",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.router.base_router:BaseRouter._validate_eplb_state",
                    source_hash="1f0a626c6f96695b579ae37c09d949978fa2ee36c281ed5a6fe74fdcf28a6fb6",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.router.base_router:BaseRouter._apply_eplb_mapping",
                    source_hash="5d3581c03d69cc012861755234093ea7d44fc0b033237e20554586e49667c559",
                    signature_hash="e6b4be56ca397c9d543b969355f44aff1bcdda97ca799ddf0f961c43628cce7a",
                ),
            ),
            contract=(
                "XCPU rebinds base_router.eplb_map_to_physical_and_record to "
                "torch_xcpu.ops.eplb_map_to_physical_and_record, preserving the "
                "exact signature, Knuth multiplicative replica hash "
                "(((token_idx * 2654435769) & 0xFFFFFFFF) % replica_count), -1 "
                "out-of-range pass-through, and conditional atomic int32 load "
                "recording guarded by record_enabled & num_unpadded_tokens."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/eplb.py::eplb_map_to_physical_and_record",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/test_eplb_map_to_physical.py",
                "vllm-xcpu-plugin/tests/layers/fused_moe/test_eplb_v5.py",
                "vllm-xcpu-plugin/tests/mpi/eplb_v5_mpi.py",
                "vllm-xcpu-plugin/tests/upstream_contracts/test_upstream_contracts_eplb.py",
            ),
        ),
        Patch(
            id="eplb.communicator.mpi",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.distributed.mpi_eplb_communicator:MpiEplbCommunicator.needs_profile_buffer_reservation",
                "vllm_xcpu_plugin.distributed.mpi_eplb_communicator:MpiEplbCommunicator.set_stream",
                "vllm_xcpu_plugin.distributed.mpi_eplb_communicator:MpiEplbCommunicator.set_transfer_context",
                "vllm_xcpu_plugin.distributed.mpi_eplb_communicator:MpiEplbCommunicator.add_send",
                "vllm_xcpu_plugin.distributed.mpi_eplb_communicator:MpiEplbCommunicator.add_recv",
                "vllm_xcpu_plugin.distributed.mpi_eplb_communicator:MpiEplbCommunicator.execute",
                "vllm_xcpu_plugin.platform:McpuPlatform.get_default_eplb_communicator",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_communicator:EplbCommunicator",
                    source_hash="ac630f0828ae958df0b05c53bc6fb55008dd73d277dc4cd9e306071b1080b45f",
                    signature_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_communicator:EplbCommunicator.add_send",
                    source_hash="827e40eaf6e96b5b231793728119ad594e58f5ed275b960ed2339637cb874f22",
                    signature_hash="fbc731a00cdfd55fbca53055a399556dc2f604ab04382852d64485b224ba619b",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_communicator:EplbCommunicator.add_recv",
                    source_hash="f4d5294a51baa3afba49a1b5629d7d149ea7443fe48a451ecb074e89407d92d4",
                    signature_hash="b9781f51c5a3c8f5c4be9a8236dbfd6dcb2f666e6ea24f64115beb25fb5489c8",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_communicator:EplbCommunicator.execute",
                    source_hash="c697171086f8cbb36b90ab3a65efdd1ed844e6bc034e3e54b18f8d80cf377612",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_communicator:EplbCommunicator.set_transfer_context",
                    source_hash="77fdfac057dd2dd4c06bbfcf141eefab86beebb32bbdc2345e21d097d0e3f2a3",
                    signature_hash="3c13c4e85976b6a4308ba3e9fe1b2559673072245b5dd5ca7c3c6ea5fe84f13e",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_communicator:EplbCommunicator.set_stream",
                    source_hash="bb55c10fb1870ac127500099283cadef47fbfa8165ede4b1596ba5f4c0747104",
                    signature_hash="fc362ee5ce28e9aa52260597fa982147798ea916d22e5ceb9b6f3dd6d2e2f6f7",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_communicator:EplbCommunicator.needs_profile_buffer_reservation",
                    source_hash="1fcd55a0df9acb9fa1832167c63cc286d33c84205a7db8c797b2e665169988af",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_communicator:register_eplb_communicator_factory",
                    source_hash="78d047cdcd8b6a096da7054432c6ee664017c69b6660a38c82170e83d32b094e",
                    signature_hash="bad17aec2f9e3e99eb02b27ec3edd0b797a2013bc401b5f5324d84acdfcb5020",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_communicator:create_eplb_communicator",
                    source_hash="caecac2ec7ca853c3a5c8de6b551f82d7d4189eabd9bf773b754127fd96a4d08",
                    signature_hash="8a51a9d178ccb7b1a033b2bda99750c5fcf750ca8d2fa47ebb8ebc843bccf7cd",
                ),
                UpstreamRef(
                    target="vllm.platforms.interface:Platform.get_default_eplb_communicator",
                    source_hash="d562e593c7f76120127a386b54e8ba85fe044078d1ef1d8ee35ded32650a4d8e",
                    signature_hash="724dff91766268c6d2a6cd581f48b81b65c61a3937d5633b3046b1b5133b1b44",
                ),
                UpstreamRef(
                    target="vllm.config.parallel:ParallelConfig.__post_init__",
                    source_hash="4a0ffec3898a447f48b6d179af3ebac4edd886fd7d4aef4a751b8b36f82dccca",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.config.parallel:EPLBConfig._validate_eplb_config",
                    source_hash="3e86d4c3bb07d0172c7abd51d1066ddfb13bb5d5ff5aa2376563a9fc0faeb204",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_state:EplbState.__init__",
                    source_hash="abba98d01c96611af21317aaf47956635e492bacb4eac224610e33b1076fdf54",
                    signature_hash="abfbd946c86496dce50ea9815cbb1e47816ee179224d6bee3f4e6a0384c41849",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_state:EplbState.step",
                    source_hash="79fd6a54190a31fb76f205b16180347620ebd63ca7c6945bd558591b85689806",
                    signature_hash="0c1e098acacc59f353bd5106bb8db8217615ab4e37162d4f718ad419ada20b39",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_state:EplbState.update_mapping",
                    source_hash="716fb88f7feb5d52827bd52e6757f59dc20b6291570f463089d65d2f1348935d",
                    signature_hash="f288c21128bb7d4bed190f62998b28df106827b093cccd10c48baf062392ac34",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_state:EplbState.prepare_forward",
                    source_hash="22ce39fea02cc0bd49d84162359a00ac19f12209dacacfc1050b682049421d31",
                    signature_hash="0c5e1bb4dad94a616575f09e379d509ce7b0a19bfbc01812eda808d54345d75c",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.rebalance_execute:rearrange_expert_weights_inplace",
                    source_hash="8ae828efe86746cce318e4a2115e71409cb6af98a9834005318b8bbf70afa2c8",
                    signature_hash="f0f9a3502c50e0f2c6c12decaaac663c3f731e5282ce574ec96028f30184cb8c",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_utils:device_stream",
                    source_hash="7fe040a7ae452bb7bafb2712d378d4d136bbebc48807981a2fde5b0016bc42df",
                    signature_hash="adb17574b0529a94b99247cd81332fc330824e855582f0d023ff66dc008a80a5",
                ),
            ),
            contract=(
                "MpiEplbCommunicator implements EplbCommunicator over the "
                "_EPLB GroupCoordinator's CpuMPICommunicator and batched "
                "torch_mpi_ext.ops.batch_isend_irecv, encoding collision-checked "
                "(layer_idx, expert_id, seq, tensor_idx) MPI tags within "
                "MPI_TAG_UB, returning needs_profile_buffer_reservation=False, "
                "and honoring device_stream(self._stream) with "
                "MPI_THREAD_MULTIPLE checks."
            ),
            impl=(
                "torch_mpi_ext/torch_mpi_ext/ops.py::batch_isend_irecv",
            ),
            tests=(
                "vllm-xcpu-plugin/tests/layers/fused_moe/test_eplb_v5.py",
                "vllm-xcpu-plugin/tests/mpi/eplb_v5_mpi.py",
                "vllm-xcpu-plugin/tests/upstream_contracts/test_upstream_contracts_eplb.py",
            ),
        ),
        Patch(
            id="eplb.routed_experts.get_expert_weights",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.layers.fused_moe.routed_experts:XcpuRoutedExperts.get_expert_weights",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.routed_experts:RoutedExperts.get_expert_weights",
                    source_hash="aec103e4b62ffcf97c4a74971d68c02663f91b6e4bce0b0c5f915ddb592656ee",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
            ),
            contract=(
                "XcpuRoutedExperts.get_expert_weights delegates parameter "
                "filtering/contiguity to super().get_expert_weights(), replaces "
                "gemm1/gemm2 packed_weight views with their full "
                "MoeGroupedGemmParams.packed_weight_storage backing (supporting "
                "padded storages), and exposes every weight/scale as a contiguous "
                "torch.uint8 view (local_num_experts, -1) so both MPI and gloo "
                "handle BF16/FP8/MXFP4 cleanly."
            ),
            impl=(
                "torch_xcpu/torch_xcpu/ops_defs/moe_grouped_gemm.py::MoeGroupedGemmParams",
            ),
            tests=(
                "torch_xcpu/test/ops/custom_ops/test_eplb_map_to_physical.py",
                "vllm-xcpu-plugin/tests/layers/fused_moe/test_eplb_v5.py",
                "vllm-xcpu-plugin/tests/mpi/eplb_v5_mpi.py",
                "vllm-xcpu-plugin/tests/upstream_contracts/test_upstream_contracts_eplb.py",
            ),
        ),
        Patch(
            id="eplb.routed_experts.load_weights_fused_redundant",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.layers.fused_moe.routed_experts:XcpuRoutedExperts.load_weights",
                "vllm_xcpu_plugin.layers.fused_moe.routed_experts:_load_redundant_fused_slices",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.routed_experts:RoutedExperts.load_weights",
                    source_hash="a456877fcfc12f8d877eed2304ada923c9f7f8e5aa15711b6527d4accf302a64",
                    signature_hash="24bcf75dc8091fd46a88cb78817a9f1f45d8eff388d0d84c185e4f7f2c4424da",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.routed_experts:RoutedExperts._map_global_expert_id_to_local_expert_id",
                    source_hash="1035bd7dd13a8e99a43a2cd8519111d260d666d500b0ded10a069f4ae0634945",
                    signature_hash="6ff338ed63789b3be7646cfe432ada9f28749d197b1fd2ba842f06226dcd79b0",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.routed_experts:RoutedExperts._orient_fused_weight",
                    source_hash="6751e61d051a73d7492bed33382de8ed1e3dab8ce18f786d5563d1e3ed397c89",
                    signature_hash="837dd0d44b55dbd90fa768c8ee20118424aa1ba85d030bbd8c947cbd59e00e2a",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.routed_experts:RoutedExperts.get_expert_mapping",
                    source_hash="82843333082c345d50cacb8a34aeb14fbcaf05c94b644da6c909d1a97cdaf08e",
                    signature_hash="2a1cc675b5d4db1920e52d34518e298ef1cae606ccc6f7c2edce32410be9b4c7",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.routed_experts:RoutedExperts.build_expert_params_mapping",
                    source_hash="efcc631504408a1cc43ef8bfe9b321b892fee2a92600e162041eaa3d73f6da6c",
                    signature_hash="416ac7cefbd2813d4521582fe274d312362fef6c1b317a258a65f2b90b26754f",
                ),
                UpstreamRef(
                    target="vllm.distributed.eplb.eplb_state:EplbState.build_initial_global_physical_to_logical_map",
                    source_hash="dead87c9c9095f9280ad5c5dff9bd9f191b93e19cce0b76867714230023968ed",
                    signature_hash="c4227e2419ea238d6c6cd5e5de6f3ba68145d5258f5b4f17b8baac80282e06a7",
                ),
            ),
            contract=(
                "RoutedExperts.load_weights unbinds 3D fused checkpoint weights "
                "(e.g. gate_up_proj, down_proj) strictly by logical index "
                "0..num_logical-1; XcpuRoutedExperts.load_weights captures only "
                "local redundant logical slices and replays param.weight_loader "
                "using EplbState.build_initial_global_physical_to_logical_map "
                "so physical redundant slots do not stay uninitialized."
            ),
            tests=(
                "vllm-xcpu-plugin/tests/layers/fused_moe/test_eplb_v5.py",
                "vllm-xcpu-plugin/tests/upstream_contracts/test_upstream_contracts_eplb.py",
            ),
        ),
        Patch(
            id="eplb.moe_methods.supports_eplb",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.layers.fused_moe.unquantized_fused_moe_method:XcpuUnquantizedFusedMoEMethod.supports_eplb",
                "vllm_xcpu_plugin.layers.fp8_moe:XcpuFp8MoEMethod.supports_eplb",
                "vllm_xcpu_plugin.layers.mxfp4_moe:XcpuMxfp4MoEMethod.supports_eplb",
                "vllm_xcpu_plugin.layers.mxfp4_moe:XcpuCompressedTensorsMxfp4MoEMethod.supports_eplb",
                "vllm_xcpu_plugin.layers.fused_moe.grouped_gemm_experts:XcpuGroupedGemmExperts._supports_parallel_config",
                "vllm_xcpu_plugin.platform:McpuPlatform.check_and_update_config",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method:UnquantizedFusedMoEMethod.supports_eplb",
                    source_hash="d62135db5d3c46e7e4cb7d9cd6629c1107e1764aadf4cf48c628bb0d972e976c",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.quantization.fp8:Fp8MoEMethod.supports_eplb",
                    source_hash="d62135db5d3c46e7e4cb7d9cd6629c1107e1764aadf4cf48c628bb0d972e976c",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.quantization.mxfp4:Mxfp4MoEMethod.supports_eplb",
                    source_hash="d62135db5d3c46e7e4cb7d9cd6629c1107e1764aadf4cf48c628bb0d972e976c",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4:CompressedTensorsW4A4Mxfp4MoEMethod.supports_eplb",
                    source_hash="941445231bccb2184f5d93b4000ab2b22de510293b95a04d0f94b657b5a2071b",
                    signature_hash="d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
                ),
            ),
            contract=(
                "BF16/FP8/MXFP4 XCPU MoE methods and XcpuGroupedGemmExperts "
                "enable supports_eplb when paired with mpi_alltoallv_v5/v6 and "
                "linear physical-to-local placement without routing_tables."
            ),
            tests=(
                "vllm-xcpu-plugin/tests/layers/fused_moe/test_eplb_v5.py",
                "vllm-xcpu-plugin/tests/layers/fused_moe/test_moe_communication_control_plane.py",
                "vllm-xcpu-plugin/tests/mpi/eplb_v5_mpi.py",
                "vllm-xcpu-plugin/tests/upstream_contracts/test_upstream_contracts_eplb.py",
            ),
        ),
    ),
)
