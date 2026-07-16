# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.machinery
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import FakeJITFunction, jit, passthrough_decorator


class FakeLanguageSymbol:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"tl.{self.name}"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        del kwargs
        return args[0] if len(args) == 1 else self


class FakeDType:
    """Type-compatible stand-in for ``triton.language.dtype``.

    TorchDynamo adds this object to an ``isinstance`` type tuple while setting
    up guards, so it must be an actual Python type rather than a generic
    language symbol.
    """


class FakeLanguageModule(types.ModuleType):
    constexpr: FakeLanguageSymbol
    dtype: type[FakeDType]
    core: FakeLanguageModule
    view: Callable[..., Any]
    extra: types.ModuleType

    def __getattr__(self, name: str) -> FakeLanguageSymbol:
        if name.startswith("__"):
            raise AttributeError(name)
        symbol = FakeLanguageSymbol(name)
        setattr(self, name, symbol)
        return symbol


class FakeConfig:
    def __init__(self, kwargs: dict[str, Any], **options: Any) -> None:
        self.kwargs = kwargs
        for name, value in options.items():
            setattr(self, name, value)


class FakeDriver:
    @staticmethod
    def is_active() -> bool:
        return True

    @staticmethod
    def get_current_target() -> FakeTarget:
        # TorchInductor asks the Triton driver for a target while constructing
        # its autotune-cache key.  Fake Triton never compiles a Triton kernel,
        # but still needs to provide a stable target for this metadata path.
        return FakeTarget()


@dataclass(frozen=True)
class FakeTarget:
    """Minimal target consumed by TorchInductor's Triton compatibility code."""

    backend: str = "xcpu"
    arch: str = "fake"


class FakeBackend:
    """Metadata-only backend returned by ``triton.compiler.make_backend``."""

    @staticmethod
    def hash() -> str:
        return "xcpu-fake-triton-backend"


def _make_backend(target: FakeTarget) -> FakeBackend:
    del target
    return FakeBackend()


class FakeCompilerObject:
    """Import-only stand-in; Fake Triton never compiles kernels."""


class FakeTritonCompilerError(RuntimeError):
    pass


def _fake_triton_key() -> str:
    return "xcpu-fake-triton"


def _language_view(*args: Any, _semantic: Any = None) -> Any:
    del _semantic
    return args[0] if args else None


@dataclass(frozen=True)
class FakeTritonInstallation:
    installed: bool


def _module(name: str, *, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, is_package=package
    )
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def _cdiv(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        return 1
    return 1 << (value - 1).bit_length()


def _discover_real_triton_paths() -> dict[str, list[str]]:
    """Find compiler-support packages without importing the real runtime."""
    paths: dict[str, list[str]] = {}
    spec = importlib.machinery.PathFinder.find_spec("triton", sys.path)
    if spec is not None and spec.submodule_search_locations is not None:
        paths["triton"] = list(spec.submodule_search_locations)

    for package, relative in (
        ("triton.backends", "backends"),
        ("triton.language", "language"),
        ("triton.language.extra", "language/extra"),
        ("triton.runtime", "runtime"),
    ):
        candidates = [Path(root) / relative for root in paths.get("triton", [])]
        existing = [str(candidate) for candidate in candidates if candidate.is_dir()]
        if existing:
            paths[package] = existing
    return paths


def install_fake_triton(*, replace_existing: bool = False) -> FakeTritonInstallation:
    """Install the XCPU Fake Triton modules into the current process."""
    existing = sys.modules.get("triton")
    package_paths = _discover_real_triton_paths()
    if existing is not None:
        if getattr(existing, "__xcpu_fake_triton__", False):
            return FakeTritonInstallation(installed=False)
        if not replace_existing:
            raise RuntimeError(
                "Cannot install XCPU Fake Triton after another Triton runtime "
                "has already been imported"
            )
        replaced_modules = {
            "triton",
            "triton.backends",
            "triton.language",
            "triton.language.core",
            "triton.language.extra",
            "triton.language.extra.libdevice",
            "triton.runtime",
            "triton.runtime.driver",
        }
        for name in replaced_modules:
            module = sys.modules.get(name)
            path = getattr(module, "__path__", None)
            if path is not None:
                package_paths[name] = list(path)
        for name in replaced_modules:
            sys.modules.pop(name, None)

    triton = _module("triton", package=True)
    triton.__path__ = package_paths.get("triton", [])  # type: ignore[attr-defined]
    triton.__xcpu_fake_triton__ = True  # type: ignore[attr-defined]
    triton.__version__ = "3.4.0+xcpu.fake"  # type: ignore[attr-defined]
    triton.jit = jit  # type: ignore[attr-defined]
    triton.autotune = passthrough_decorator  # type: ignore[attr-defined]
    triton.heuristics = passthrough_decorator  # type: ignore[attr-defined]
    triton.Config = FakeConfig  # type: ignore[attr-defined]
    triton.cdiv = _cdiv  # type: ignore[attr-defined]
    triton.next_power_of_2 = _next_power_of_2  # type: ignore[attr-defined]

    language = FakeLanguageModule("triton.language")
    language.__spec__ = importlib.machinery.ModuleSpec(
        "triton.language", loader=None, is_package=True
    )
    language.__path__ = []  # type: ignore[attr-defined]
    language.__path__ = package_paths.get(  # type: ignore[attr-defined]
        "triton.language", []
    )
    language.constexpr = FakeLanguageSymbol("constexpr")
    language.dtype = FakeDType
    language.core = language
    language.view = _language_view
    triton.language = language  # type: ignore[attr-defined]

    language_extra = _module("triton.language.extra", package=True)
    language_extra.__path__ = package_paths.get(  # type: ignore[attr-defined]
        "triton.language.extra", []
    )
    libdevice = FakeLanguageModule("triton.language.extra.libdevice")
    libdevice.__spec__ = importlib.machinery.ModuleSpec(
        "triton.language.extra.libdevice", loader=None
    )
    language_extra.libdevice = libdevice  # type: ignore[attr-defined]
    language.extra = language_extra

    backends = _module("triton.backends", package=True)
    backends.__path__ = package_paths.get(  # type: ignore[attr-defined]
        "triton.backends", []
    )
    backends.backends = {  # type: ignore[attr-defined]
        "cpu": types.SimpleNamespace(driver=FakeDriver)
    }
    triton.backends = backends  # type: ignore[attr-defined]

    runtime = _module("triton.runtime", package=True)
    runtime.__path__ = package_paths.get(  # type: ignore[attr-defined]
        "triton.runtime", []
    )
    driver = _module("triton.runtime.driver")
    driver.active = FakeDriver  # type: ignore[attr-defined]
    # TorchInductor imports ``driver`` from this module and then accesses its
    # ``active`` driver.  Keep the module-level shim above for older callers,
    # while exposing the object-level ABI used by current TorchInductor.
    driver.driver = types.SimpleNamespace(active=FakeDriver)  # type: ignore[attr-defined]
    driver.set_active_to_cpu = lambda: None  # type: ignore[attr-defined]
    runtime.driver = driver  # type: ignore[attr-defined]
    triton.runtime = runtime  # type: ignore[attr-defined]

    compiler_modules = {}
    if not package_paths.get("triton"):
        # TorchInductor imports these modules while initializing even in eager
        # CPU processes.  Empty modules select its built-in compatibility path;
        # actual compilation deliberately remains unsupported by Fake Triton.
        backend_compiler = _module("triton.backends.compiler")
        backends.compiler = backend_compiler  # type: ignore[attr-defined]
        compiler = _module("triton.compiler", package=True)
        compiler_impl = _module("triton.compiler.compiler")
        compiler.CompiledKernel = FakeCompilerObject  # type: ignore[attr-defined]
        compiler_impl.ASTSource = FakeCompilerObject  # type: ignore[attr-defined]
        compiler_impl.make_backend = _make_backend  # type: ignore[attr-defined]
        compiler.compiler = compiler_impl  # type: ignore[attr-defined]
        triton.compiler = compiler  # type: ignore[attr-defined]

        autotuner = _module("triton.runtime.autotuner")
        autotuner.Autotuner = FakeCompilerObject  # type: ignore[attr-defined]
        autotuner.Heuristics = FakeCompilerObject  # type: ignore[attr-defined]
        autotuner.Config = FakeConfig  # type: ignore[attr-defined]
        autotuner.OutOfResources = FakeTritonCompilerError  # type: ignore[attr-defined]
        autotuner.PTXASError = FakeTritonCompilerError  # type: ignore[attr-defined]
        runtime.autotuner = autotuner  # type: ignore[attr-defined]

        runtime_jit = _module("triton.runtime.jit")
        runtime_jit.JITFunction = FakeJITFunction  # type: ignore[attr-defined]
        runtime_jit.KernelInterface = FakeCompilerObject  # type: ignore[attr-defined]
        runtime.jit = runtime_jit  # type: ignore[attr-defined]
        triton.JITFunction = FakeJITFunction  # type: ignore[attr-defined]

        runtime_cache = _module("triton.runtime.cache")
        runtime_cache.triton_key = _fake_triton_key  # type: ignore[attr-defined]
        runtime.cache = runtime_cache  # type: ignore[attr-defined]
        triton.knobs = types.SimpleNamespace(  # type: ignore[attr-defined]
            autotuning=types.SimpleNamespace(print=False),
            runtime=types.SimpleNamespace(jit_post_compile_hook=None),
        )
        compiler_modules = {
            "triton.backends.compiler": backend_compiler,
            "triton.compiler": compiler,
            "triton.compiler.compiler": compiler_impl,
            "triton.runtime.autotuner": autotuner,
            "triton.runtime.jit": runtime_jit,
            "triton.runtime.cache": runtime_cache,
        }

    modules = {
        "triton": triton,
        "triton.backends": backends,
        "triton.language": language,
        "triton.language.core": language,
        "triton.language.extra": language_extra,
        "triton.language.extra.libdevice": libdevice,
        "triton.runtime": runtime,
        "triton.runtime.driver": driver,
    }
    modules.update(compiler_modules)
    sys.modules.update(modules)
    # Submodules imported by the real Triton package remain in sys.modules so
    # TorchInductor can keep using compiler/runtime helpers.  Since their old
    # parent packages were replaced above, restore the attributes that Python's
    # import machinery would normally install while loading a child module.
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("triton.") or name in modules:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and "." not in child_name:
            setattr(parent, child_name, module)
    return FakeTritonInstallation(installed=True)
