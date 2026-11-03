# 上游契约（upstream contracts）

插件对 vLLM 的每一处改动（OOT 子类、方法覆盖、模块属性重绑、Fake Triton
adapter）都依赖某些上游代码的语义。上游契约把这种依赖写成可校验的清单：

```
上游符号（module:qualname，指纹）
   └─ Patch：下游 patch 点 + 语义契约 + 实现（torch_xcpu）+ 测试 + 审计来源
```

目标：vLLM 升级时，任何影响 patch 的上游变化都以统一形态的错误
`[XCPU-UPSTREAM-DRIFT]` 停下，并直接给出需要阅读的上下文；而不是因为常规路径
还能跑，把语义变化留到边缘情况才暴露。

## 1. 组成

| 位置 | 内容 |
| --- | --- |
| `vllm_xcpu_plugin/upstream_contracts/` | 引擎：数据模型、`rebind()`、校验、报告、lint、CLI |
| `vllm_xcpu_plugin/models/<category>/upstream.py` | 类别清单 `MANIFEST`（当前：`deepseek_v41`） |
| `upstream_contracts.MANIFESTS` | 类别 → 清单模块 |
| `upstream_compatibility.py` | 旧的逐算子/逐文件钉住；`verify_upstream_compatibility()` 先校验已迁移类别的契约 |

清单条目：

- `Patch`
  - `downstream`：下游 patch 点（`module:qualname`），lint 保证可解析；
  - `upstream`：依赖的上游符号 `UpstreamRef`。`target` 写的是 patch 实际触达
    的路径（例如重绑的是 `nvidia.model:mhc_post_tilelang` 这个名字，而不是
    tilelang 里的定义），用 `source_hash`/`signature_hash` 钉住定义，或用
    `expect` 钉住"该路径仍指向某对象"；
  - `contract`：下游镜像或假设的上游语义，以及上游 diff 中需要重点核对的点；
  - `impl` / `tests`：语义实现（torch_xcpu 算子等）和覆盖它的测试、E2E 文档，
    格式为相对工作区根目录的 `path[::symbol]`，lint 检查其存在；
  - `source`：`KernelSource`，规则与 `TRITON_COMPATIBILITY_GUIDE.md` 2.1 相同。
    `local_commit` 必须是**已提交**、包含被审阅源码的 vLLM 快照，报告用它恢复
    审阅时的源码做 diff。
- `WatchedModule`：整体镜像的上游文件（SHA256）。文件变化时，报告用
  `local_commit` 快照逐符号比较，把每个变化的函数、方法、类体、常量或模块级
  语句（import）映射回引用它的 patch；没有 patch 引用的变化标为 unbound。

- `WatchedFile`：XCPU 算子镜像的非 Python 上游源码（CUDA kernel 等，路径相对 vLLM
  仓库根目录），按 SHA256 钉住，变化时给出整文件 diff；`Patch.upstream_files`
  声明依赖它的 patch。

指纹算法与 Fake Triton 相同（规范化 AST、signature）。常量按 `repr` 取指纹。

## 2. 校验时机与报错形态

- 运行时：`register_deepseek_v41()`（插件加载）和 `install_deepseek_v41()`
  （模型导入）在安装任何 patch 之前调用 `verify_upstream_compatibility`，
  契约不成立时抛出 `UpstreamDriftError`（`KernelVersionError` 子类）。
- 重绑一律使用 `rebind(owner, name, value)`：属性不存在即报错（上游改名时
  不会静默新增属性），原对象记录下来，之后的校验仍按原对象比较。
- Fake Triton adapter 的期望指纹从清单读取，dispatch 时的不匹配报错会指向清单。

报错示例（缩略）：

```
[XCPU-UPSTREAM-DRIFT] deepseek_v41: 3 upstream change(s) reach XCPU patches (1 bound, 2 unbound).
...
[1] SOURCE CHANGED: vllm.models.deepseek_v41.common.engram:NgramHashState.forward
    upstream: .../vllm/vllm/models/deepseek_v41/common/engram.py:511
    reviewed: v0.31.0 ...; local=...; main=...
    patch deepseek_v41.engram.hash
      downstream: vllm_xcpu_plugin.models.deepseek_v41.engram:XcpuNgramHashState.forward
                  .../vllm_xcpu_plugin/models/deepseek_v41/engram.py:55
      impl:       torch_xcpu/torch_xcpu/ops_defs/deepseek_v41/__init__.py::deepseek_v41_engram_hash
      contract:   ...history comes from the chunk, then lookback_token_ids, then the slot cache...
      test:       torch_xcpu/test/ops/custom_ops/deepseek_v41/test_engram_hash.py
    diff (reviewed -> current):
      +    lookback_token_ids = lookback_token_ids[:, -self.lookback_depth :]
[2] WATCHED MODULE CHANGED: vllm.models.deepseek_v41.nvidia.model:DeepseekV4DecoderLayer._new_upstream_helper
    detail:   added in watched module; no patch references it: ...
```

漂移类型：`MISSING`（改名/移动/删除）、`REBOUND ELSEWHERE`（路径指向了别的
对象）、`INTERFACE CHANGED`（签名）、`SOURCE CHANGED`（实现）、
`WATCHED MODULE CHANGED`（监视文件中的符号级变化，可能是 unbound）、
`WATCHED SOURCE CHANGED`（钉住的 CUDA 等源文件变化）。

## 3. CLI

在 `vllm-xcpu-plugin/` 目录下运行（在工作区根目录下，`./vllm` 仓库目录会被当作
命名空间包遮蔽真正的 vllm 包）：

```bash
cd /shared/vllm-xcpu-dev-kit-0.31-v3/vllm-xcpu-plugin
python -m vllm_xcpu_plugin.upstream_contracts check [CATEGORY ...] [--json]  # 汇总，有漂移时退出码 1
python -m vllm_xcpu_plugin.upstream_contracts report CATEGORY                # 含完整 diff
python -m vllm_xcpu_plugin.upstream_contracts fingerprint MODULE:QUALNAME    # 审阅后用于更新钉住值
python -m vllm_xcpu_plugin.upstream_contracts lint [CATEGORY ...]            # 未声明的 patch 点、失效引用
python -m vllm_xcpu_plugin.upstream_contracts provenance [CATEGORY ...]      # 钉住值与 local_commit 快照一致
python -m vllm_xcpu_plugin.upstream_contracts sites CATEGORY                 # 发现的 patch 点及其所属 patch
```

测试：`pytest tests/upstream_contracts`（lint、check、provenance、引擎行为）。

## 4. 处理漂移（升级流程）

1. 一次性获取全部漂移：`check --json` 或 `report CATEGORY`。不要逐个启动服务排错。
2. 对每一项：读 diff，对照 `contract` 判断语义是否变化。
   - 语义未变（重构、注释、无关分支）：只更新这一条引用的指纹。
   - 语义变化：修改下游 patch、`impl` 中的 torch_xcpu 实现及其 reference，补充
     `tests` 中的用例；需要时更新 `contract` 文本。
   - `MISSING`：在新版本中找到对应符号（`git log -S`、grep），更新 `target`，
     必要时调整下游。
   - unbound：判断是否有 patch 隐式依赖它；有则补 `UpstreamRef`（或新 `Patch`），
     否则只更新该监视模块的 SHA256。
3. 提交被审阅的 vLLM 状态，把相关条目的 `KernelSource.local_commit`（及
   `main_commit`/`backport_commit`）指向它；用 `fingerprint` 读出新指纹并逐条
   手工更新。**禁止未读 diff 就批量刷新指纹。**
4. 运行 `lint`、`provenance`、`pytest tests/upstream_contracts`、`tests` 中列出的
   算子测试，以及清单 `checks` 中的 E2E 文档。

## 5. 新增或修改 patch

- 覆盖 vLLM 类的方法、重绑上游属性、替换实例 `__class__`，都必须在清单中声明；
  `lint` 会拒绝未声明的 patch 点和直接赋值形式的重绑（必须用 `rebind()`）。
- 覆盖的方法若属于监视模块，所在 patch 必须钉住被覆盖的上游方法。
- 下游读取上游的私有可调用成员（`self._helper` 定义在 vLLM 基类中，或经 import
  名访问 `module._x` / `Class._x`）时，lint 要求钉住该成员（`private-use`）。
- 依赖一个上游函数的**行为**（例如复用其输出形状、依赖其调用顺序）而不覆盖它时，
  同样加入 `UpstreamRef`，并在 `contract` 中写明依赖点。调用点（例如
  `DeepseekV4DecoderLayer.forward` 调用被重绑的 mhc 函数）也属于依赖。
- 新类别：新建 `models/<category>/upstream.py` 定义 `MANIFEST`，登记到
  `upstream_contracts.MANIFESTS`，把 `upstream_compatibility.py` 中该类别的
  文件级钉住迁移为 `WatchedModule`。
