# GDN 首个 prefill chunk 审计（2026-10-06）

## 源码身份

| 字段 | 值 |
|---|---|
| 沿用的人工版本标签 | `v0.25.1` |
| 本地 vLLM 已提交源码 | `7649db558cbc2ec0ba786bd6857cf2f5b66580d0` |
| 上游修复 | `79468c20ef23227d4e051f807e10fd54fb24e24f`（#51565） |
| source_version | `v0.25.1; local=7649db558cbc; upstream=79468c20ef23` |
| build source hash | `5db0fbae76ee85cae95d97aef25ec0521fc54322987331e2602cf11948d3e6e9` |
| build signature hash | `6a9c419a47084c40387d3cf2cbdb4ea3bc55a2a15a4a237dab55d3e71585ea5a` |

只更新 `GDNAttentionMetadataBuilder.build` 的来源记录。其签名未变，源码从
`94479e086d8e8348ec804400967f650bea96bbe243fb57193c2e073cb1541a79`
变为表中指纹。其余10个GDN metadata记录和fake Triton kernel记录未改。
标签沿用已有人工确认范围；不是新建release tag，也不是宣称整个上游版本已兼容。

## 审阅内容及适配

已对照上游commit的完整build修改及测试，阅读本地
`vllm/v1/attention/backends/gdn_attn.py`、`backends/utils.py` 的
`split_decodes_and_prefills`、V2 `model_states/mamba_hybrid.py::prepare_attn`、
`qwen_gdn_linear_attn.py` 的prefill/decode分支，以及插件
`gdn_metadata_patch.py`、`torch_xcpu/csrc/gdn_metadata.cpp` 的分配/构造两条路径。

新请求的query_len=1不意味着已有历史。CPU调度字段共同确定
`no_prior_state = query_len>0 & seq_len_upper_bound<=query_len & is_prefilling`。
它必须走prefill并得到has_initial_state=False，避免读取复用状态。
续算的1-token chunk仍decode；没有prefill标记的graph capture仍decode；
尾部padding不计入prefill请求数或token数。

插件不会仅依赖被替换的Python实现：同一CPU mask传给C++ output allocator
和builder，避免“分配按decode、填充按prefill”的不一致。C++只同步读取CPU
bool向量，支持stride，不添加device readback。两个schema均以可选末尾参数扩展，
旧调用仍可用；新插件必须部署支持新参数的torch_xcpu。

另审阅conv prefill/decode交接，发现torch_xcpu错误地将整个物理扩展窗口当成
prefill历史。上游逻辑历史恒为width-1，位于头部；spec decode用accepted-1偏移。
修复在torch_xcpu内实现。上游conv函数及签名未改，因此不刷新其来源指纹。

## 验证

配套torch_xcpu提交：卷积修复`7519361dab3a9472fba7d313ccf166e313083cb8`，
元数据适配`dd4d3781f6ae78903503c5c08cc10fb097ccc6fd`。必须包含两者并重新编译安装。

- vLLM新增测试修复前2 failed/4 passed，修复后完整GDN metadata文件15 passed。
- 插件`tests/plugin_tests/test_gdn_patch.py`：224 passed，覆盖first/resumed、
  none/align/all、strided和graph padding，与vLLM CPU builder逐字段比较。
- torch_xcpu卷积交接新增9项修复前全失败；修复后update文件17 passed，fn矩阵270 passed。
- Qwen3.8-27B-FP8 TP2 DFlash2，prefix OFF，12次完全相同的1-token greedy请求：
  原版本12种乱码，配套修复后12次一致。

全套TUI、compile和调度实验持续记录在开发套件
`docs/investigations/dflash2_no_prefix_cache/`。这里记录的是代码审计与本地x86证据，
不是对尚未运行的NPU后端作兼容性保证。
