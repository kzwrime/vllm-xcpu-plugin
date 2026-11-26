# SPDX-License-Identifier: Apache-2.0
"""An upstream change must surface as a drift that names the downstream site,
its contract and the reviewed-vs-current diff (what an agent acts on)."""

import importlib
import subprocess
import sys
import textwrap
import types

import pytest

from vllm_xcpu_plugin import upstream_contracts as uc
from vllm_xcpu_plugin.fake_triton.provenance import KernelSource
from vllm_xcpu_plugin.fake_triton.runtime import KernelVersionError
from vllm_xcpu_plugin.upstream_contracts import _source
from vllm_xcpu_plugin.upstream_contracts.report import format_text, to_json

UPSTREAM = "xcpu_contract_fixture_upstream"

REVIEWED = textwrap.dedent(
    """
    DEAD_ID = -1


    def gate(x, eps):
        return x / (x + eps)


    class Table:
        rows = 4

        def lookup(self, ids):
            return [i % self.rows for i in ids]
    """
)


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    """A one-file upstream package in its own git repository."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    path = tmp_path / f"{UPSTREAM}.py"
    path.write_text(REVIEWED)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "reviewed")
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.syspath_prepend(str(tmp_path))
    # Same-size edits within one second would reuse a stale .pyc.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    sys.modules.pop(UPSTREAM, None)
    module = importlib.import_module(UPSTREAM)

    source = KernelSource(tag="test", local_commit=commit, main_commit=commit)

    def pin(target):
        source_hash, signature_hash = uc.fingerprint(uc.resolve(target))
        return uc.UpstreamRef(
            target=target, source_hash=source_hash, signature_hash=signature_hash
        )

    manifest = uc.Manifest(
        category="fixture",
        packages=(),
        checks=("rerun the fixture E2E",),
        watched=(
            uc.WatchedModule(
                module=UPSTREAM,
                sha256=_source.sha256_text(REVIEWED),
                source=source,
            ),
        ),
        patches=(
            uc.Patch(
                id="fixture.gate",
                downstream=("tests.downstream:xcpu_gate",),
                upstream=(pin(f"{UPSTREAM}:gate"), pin(f"{UPSTREAM}:DEAD_ID")),
                source=source,
                contract="gate(x, eps) = x / (x + eps); eps keeps the ratio finite.",
                tests=("tests/test_gate.py",),
            ),
            uc.Patch(
                id="fixture.lookup",
                downstream=("tests.downstream:XcpuTable.lookup",),
                upstream=(pin(f"{UPSTREAM}:Table.lookup"),),
                source=source,
                contract="ids wrap modulo Table.rows.",
            ),
        ),
    )
    manifest_module = types.ModuleType("xcpu_contract_fixture_manifest")
    manifest_module.MANIFEST = manifest
    monkeypatch.setitem(sys.modules, manifest_module.__name__, manifest_module)
    monkeypatch.setitem(uc.MANIFESTS, "fixture", manifest_module.__name__)

    fixture = types.SimpleNamespace(module=module, commit=commit)

    def change(text):
        path.write_text(textwrap.dedent(text))
        _source.file_symbols.cache_clear()
        sys.modules.pop(UPSTREAM, None)
        fixture.module = importlib.import_module(UPSTREAM)

    fixture.change = change
    uc._verified.cache_clear()
    yield fixture
    uc._verified.cache_clear()
    sys.modules.pop(UPSTREAM, None)
    _source.file_symbols.cache_clear()


def test_unchanged_upstream_holds(upstream):
    assert uc.check("fixture") == []
    uc.verify("fixture")


def test_source_change_names_patch_contract_and_diff(upstream):
    upstream.change(REVIEWED.replace("x / (x + eps)", "x / (x + 2 * eps)"))
    drifts = {d.target: d for d in uc.check("fixture")}
    drift = drifts[f"{UPSTREAM}:gate"]
    assert drift.kind == "source"
    assert [p.id for p in drift.patches] == ["fixture.gate"]
    assert "+    return x / (x + 2 * eps)" in drift.diff
    assert upstream.commit[:12] in drift.reviewed
    # The watched-module breakdown does not report the same symbol twice.
    assert list(drifts) == [f"{UPSTREAM}:gate"]

    text = format_text("fixture", list(drifts.values()))
    assert text.startswith(uc.DRIFT_MARKER)
    assert "tests.downstream:xcpu_gate" in text
    assert "eps keeps the ratio finite" in text
    assert "rerun the fixture E2E" in text
    assert to_json("fixture", list(drifts.values()))["drifts"][0]["diff"]

    with pytest.raises(uc.UpstreamDriftError) as excinfo:
        uc.verify("fixture")
    assert isinstance(excinfo.value, KernelVersionError)
    assert excinfo.value.drifts[0].target == f"{UPSTREAM}:gate"


def test_signature_change_is_an_interface_change(upstream):
    upstream.change(REVIEWED.replace("def gate(x, eps)", "def gate(x, eps, scale=1)"))
    (drift,) = uc.check("fixture")
    assert drift.kind == "signature"


def test_constant_change_is_bound(upstream):
    upstream.change(REVIEWED.replace("DEAD_ID = -1", "DEAD_ID = -2"))
    drifts = {d.target: d for d in uc.check("fixture")}
    drift = drifts[f"{UPSTREAM}:DEAD_ID"]
    assert [p.id for p in drift.patches] == ["fixture.gate"]
    assert "+DEAD_ID = -2" in drift.diff and "WARNING" not in drift.reviewed


def test_unreferenced_change_in_watched_module_is_reported_unbound(upstream):
    upstream.change(
        REVIEWED.replace("    rows = 4", "    rows = 8")
        + "\n\ndef new_helper():\n    return 0\n"
    )
    drifts = {d.target: d for d in uc.check("fixture")}
    helper = drifts[f"{UPSTREAM}:new_helper"]
    assert helper.kind == "module" and not helper.patches
    assert "no patch references it" in helper.detail
    # A class-body change points at the patches bound inside the class.
    table = drifts[f"{UPSTREAM}:Table"]
    assert [p.id for p in table.related] == ["fixture.lookup"]
    assert "rows = 8" in table.diff


def test_removed_symbol_is_missing(upstream):
    upstream.change(REVIEWED.replace("def gate(x, eps)", "def gate_v2(x, eps)"))
    drifts = {d.target: d for d in uc.check("fixture")}
    assert drifts[f"{UPSTREAM}:gate"].kind == "missing"


def test_rebind_keeps_the_original_resolvable(upstream):
    original = upstream.module.gate
    uc.rebind(upstream.module, "gate", lambda x, eps: 0)
    try:
        assert uc.resolve(f"{UPSTREAM}:gate") is original
        assert uc.check("fixture") == []
    finally:
        upstream.module.gate = original


def test_rebind_of_a_vanished_attribute_fails(upstream):
    with pytest.raises(KernelVersionError, match="no longer exists upstream"):
        uc.rebind(upstream.module, "gate_renamed", lambda x, eps: 0)


def _register_watched_file_manifest(
    monkeypatch,
    *,
    rel_path: str,
    sha256: str,
    commit: str,
    category: str = "watched_file_fixture",
):
    source = KernelSource(tag="test", local_commit=commit, main_commit=commit)
    manifest = uc.Manifest(
        category=category,
        packages=(),
        watched_files=(
            uc.WatchedFile(path=rel_path, sha256=sha256, source=source),
        ),
        patches=(
            uc.Patch(
                id=f"{category}.kernel",
                downstream=("tests.downstream:xcpu_kernel",),
                upstream=(),
                upstream_files=(rel_path,),
                source=source,
                contract="Mirrors the upstream CUDA kernel.",
            ),
        ),
    )
    mod = types.ModuleType(f"xcpu_contract_{category}_manifest")
    mod.MANIFEST = manifest
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(uc.MANIFESTS, category, mod.__name__)
    uc._verified.cache_clear()
    return manifest


def _mock_vllm_module_path(monkeypatch, vllm_init_path):
    orig_module_path = _source.module_path

    def fake_module_path(module: str):
        if module == "vllm":
            return vllm_init_path
        return orig_module_path(module)

    monkeypatch.setattr(_source, "module_path", fake_module_path)


def test_watched_file_checked_in_source_checkout(tmp_path, monkeypatch):
    from vllm_xcpu_plugin.upstream_contracts.lint import lint, provenance

    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "vllm").mkdir()
    vllm_init = tmp_path / "vllm" / "__init__.py"
    vllm_init.write_text("")
    kernel = tmp_path / "csrc" / "kernel.cu"
    kernel.parent.mkdir(parents=True)
    kernel_v1 = "__global__ void k() { /* v1 */ }\n"
    kernel.write_text(kernel_v1)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "v1")
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    _mock_vllm_module_path(monkeypatch, vllm_init)
    manifest = _register_watched_file_manifest(
        monkeypatch,
        rel_path="csrc/kernel.cu",
        sha256=_source.sha256_text(kernel_v1),
        commit=commit,
    )

    assert _source.vllm_repo_root() == tmp_path.resolve()
    assert _source.vllm_git_root() == tmp_path.resolve()
    assert uc.check(manifest.category) == []
    uc.verify(manifest.category)
    assert provenance(manifest) == []

    # Modifying the watched file surfaces a 'file' drift with diff.
    kernel.write_text("__global__ void k() { /* v2 */ }\n")
    (drift,) = uc.check(manifest.category)
    assert drift.kind == "file"
    assert drift.target == "vllm:csrc/kernel.cu"
    assert [p.id for p in drift.patches] == [f"{manifest.category}.kernel"]
    assert "+__global__ void k() { /* v2 */ }" in (drift.diff or "")

    # Deleting the watched file in a source checkout must still report missing.
    kernel.unlink()
    (drift,) = uc.check(manifest.category)
    assert drift.kind == "missing"
    assert drift.target == "vllm:csrc/kernel.cu"
    lint_problems = lint(manifest)
    assert "watched file csrc/kernel.cu does not exist" in lint_problems


def test_watched_file_checked_in_non_git_source_checkout(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='vllm'\n")
    (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.26)\n")
    (tmp_path / "vllm").mkdir()
    vllm_init = tmp_path / "vllm" / "__init__.py"
    vllm_init.write_text("")
    kernel = tmp_path / "csrc" / "kernel.cu"
    kernel.parent.mkdir(parents=True)
    kernel_text = "__global__ void k() {}\n"
    kernel.write_text(kernel_text)

    _mock_vllm_module_path(monkeypatch, vllm_init)
    manifest = _register_watched_file_manifest(
        monkeypatch,
        rel_path="csrc/kernel.cu",
        sha256=_source.sha256_text(kernel_text),
        commit="0" * 40,
        category="watched_file_nogit",
    )

    assert _source.vllm_repo_root() == tmp_path.resolve()
    assert _source.vllm_git_root() is None
    assert uc.check(manifest.category) == []

    kernel.unlink()
    (drift,) = uc.check(manifest.category)
    assert drift.kind == "missing"


def test_watched_file_skipped_for_non_editable_vllm_install(tmp_path, monkeypatch):
    from vllm_xcpu_plugin.upstream_contracts.lint import lint, provenance

    # Place site-packages inside a parent git repo (like dev-kit/.venv) to verify
    # the parent repo is never mistaken for a vLLM checkout.
    _git(tmp_path, "init", "-q")
    site_packages = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    vllm_dir = site_packages / "vllm"
    vllm_dir.mkdir(parents=True)
    vllm_init = vllm_dir / "__init__.py"
    vllm_init.write_text("")

    _mock_vllm_module_path(monkeypatch, vllm_init)
    manifest = _register_watched_file_manifest(
        monkeypatch,
        rel_path="csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu",
        sha256=_source.sha256_text("irrelevant"),
        commit="1" * 40,
        category="watched_file_wheel",
    )

    assert _source.vllm_repo_root() is None
    assert _source.vllm_git_root() is None
    assert (
        _source.vllm_repo_path(
            "csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu"
        )
        is None
    )
    assert _source.reviewed_text(vllm_init, "1" * 40) is None
    assert uc.check(manifest.category) == []
    uc.verify(manifest.category)
    # lint() skips repo-only watched file existence in wheel mode.
    assert not any("does not exist" in p for p in lint(manifest))
    # provenance() fails closed with a clear message when vllm has no git repo.
    prov = provenance(manifest)
    assert len(prov) == 1 and "git checkout" in prov[0]


def test_watched_file_in_package_data_still_checked_in_wheel(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    vllm_dir = site_packages / "vllm"
    data_file = vllm_dir / "transformers_utils" / "chat_templates" / "demo.jinja"
    data_file.parent.mkdir(parents=True)
    vllm_init = vllm_dir / "__init__.py"
    vllm_init.write_text("")
    data_file.write_text("v1")

    _mock_vllm_module_path(monkeypatch, vllm_init)
    rel = "vllm/transformers_utils/chat_templates/demo.jinja"
    manifest = _register_watched_file_manifest(
        monkeypatch,
        rel_path=rel,
        sha256=_source.sha256_text("v1"),
        commit="2" * 40,
        category="watched_file_pkgdata",
    )

    assert _source.vllm_repo_root() is None
    assert _source.vllm_repo_path(rel) == data_file
    assert _source.vllm_repo_path("vllm/unshipped.cu") is None
    assert uc.check(manifest.category) == []

    data_file.write_text("v2")
    (drift,) = uc.check(manifest.category)
    assert drift.kind == "file"
    assert drift.target == f"vllm:{rel}"


def test_lint_workspace_root_handles_non_editable_plugin(tmp_path):
    from vllm_xcpu_plugin.upstream_contracts.lint import _workspace_root

    ws = tmp_path / "devkit"
    (ws / "vllm-xcpu-plugin" / "vllm_xcpu_plugin").mkdir(parents=True)
    scripts_dir = ws / "vllm_scripts"
    scripts_dir.mkdir()
    site_lint = (
        ws
        / ".venv"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "vllm_xcpu_plugin"
        / "upstream_contracts"
        / "lint.py"
    )
    site_lint.parent.mkdir(parents=True)
    site_lint.write_text("")

    # Editable anchor resolves directly from __file__.
    editable_lint = (
        ws / "vllm-xcpu-plugin" / "vllm_xcpu_plugin" / "upstream_contracts" / "lint.py"
    )
    assert _workspace_root(anchor=editable_lint, cwd=tmp_path) == ws.resolve()
    # Non-editable site-packages anchor falls back to cwd ancestors.
    assert _workspace_root(anchor=site_lint, cwd=scripts_dir) == ws.resolve()
    # Outside any workspace, returns None instead of pointing into site-packages.
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert _workspace_root(anchor=site_lint, cwd=outside) is None
