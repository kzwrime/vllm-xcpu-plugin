# AF-EP 最小功能回归

覆盖以下行为：

| 文件 | 覆盖 |
|---|---|
| `test_attention_bootstrap.py` | A 完成模型加载、排空设备任务后初始化传输；允许 A compile，拒绝 CUDAGraph capture |
| `test_attention_client_v7.py` | vLLM 选择远程专家方法；A 不持有专家权重；不同层返回独立结果，且不重复 all-reduce；开启检查时可 tracing V7 dispatch/combine |
| `test_expert_service_v7.py` | F 完成多个 MoE 层的接收、计算、返回，保持有效行数和输出对应关系；BF16/FP32 无输出 combine-send 在 DCE 后保留有序副作用 |
| `test_routed_expert_weights.py` | 仅加载本地 EP 专家并保留量化张量名称；缺失专家层权重时拒绝启动（2 个用例） |

从 devkit 根目录运行：

```bash
.venv/bin/python -m pytest -q vllm-xcpu-plugin/tests/af_ep
```

这些是 CPU 上的插件功能测试，MPI 端点和专家计算使用替身。tracing 用例使用真实 V7 FakeTensor 实现。真实通信数值由 [torch_xcpu V7 round-trip](../../../torch_xcpu/test/mpi_custom_ops/test_moe_af_v7_roundtrip_mpi.py) 覆盖；设置 `XCPU_AF_TEST_COMPILE_ATTENTION=1` / `XCPU_AF_TEST_COMPILE_EXPERT=1` 可分别编译 A / F 侧端点。完整模型、量化和服务启动使用 [AFD E2E 入口](../../../vllm_scripts/e2e/run_afd_crossnode_matrix.sh)。

架构/import 约束、诊断 allreduce 开关、内部字段、正则枚举、重复的初始化细节和开发期 bootstrap smoke 不再单独测试。生产功能和检查逻辑保持原样。
