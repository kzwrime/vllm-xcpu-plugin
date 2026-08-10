# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import update_wrapper
from typing import Any


class FakeTritonError(RuntimeError):
    """Base error raised by the XCPU Fake Triton runtime."""


class UnknownKernelError(FakeTritonError):
    """Raised when an unregistered kernel is launched."""


class KernelVersionError(FakeTritonError):
    """Raised when a registered kernel does not match its manifest."""


class InvalidLaunchError(FakeTritonError):
    """Raised when grid or launch arguments are invalid."""


_TRITON_LAUNCH_METADATA = frozenset({
    "enable_warp_specialization",
    "launch_cooperative_grid",
    "maxnreg",
    "num_ctas",
    "num_stages",
    "num_warps",
})


def _source_fingerprint(fn: Callable[..., Any]) -> str:
    try:
        source = textwrap.dedent(inspect.getsource(fn))
        normalized = ast.dump(
            ast.parse(source),
            annotate_fields=True,
            include_attributes=False,
        )
    except (OSError, TypeError, SyntaxError):
        code = fn.__code__
        normalized = repr((
            code.co_code,
            code.co_consts,
            code.co_names,
            code.co_varnames,
        ))
    return hashlib.sha256(normalized.encode()).hexdigest()


def _signature_fingerprint(signature: inspect.Signature) -> str:
    fields = []
    for parameter in signature.parameters.values():
        fields.append((
            parameter.name,
            parameter.kind.name,
            repr(parameter.default),
            repr(parameter.annotation),
        ))
    return hashlib.sha256(repr(fields).encode()).hexdigest()


def _is_constexpr(annotation: Any) -> bool:
    if getattr(annotation, "name", None) == "constexpr":
        return True
    return isinstance(annotation, str) and annotation.rsplit(".", 1)[-1] == "constexpr"


@dataclass(frozen=True)
class KernelLaunch:
    """A validated Fake Triton kernel launch passed to an adapter."""

    kernel: FakeJITFunction
    grid: tuple[int, ...]
    arguments: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def argument(self, name: str) -> Any:
        try:
            return self.arguments[name]
        except KeyError as exc:
            raise InvalidLaunchError(
                f"{self.kernel.qualname}: missing bound argument {name!r}"
            ) from exc


KernelDispatcher = Callable[[KernelLaunch], Any]


class _ArgumentBinder:
    """Bind the simple signatures used by Triton kernels without inspect.bind.

    Triton kernels overwhelmingly use positional-or-keyword parameters plus
    defaults for constexpr arguments.  ``inspect.Signature.bind`` is useful for
    diagnostics, but walking ``inspect.Parameter`` objects on every launch is
    disproportionately expensive.  Keep it as the exact error/fallback path
    and use precomputed names and defaults for valid launches.
    """

    def __init__(self, signature: inspect.Signature) -> None:
        self._signature = signature
        parameters = tuple(signature.parameters.values())
        self._fast_path = all(
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            for parameter in parameters
        )
        self._names = tuple(parameter.name for parameter in parameters)
        self.parameter_names = frozenset(self._names)
        self._positional_names = tuple(
            parameter.name
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        self._keyword_names = frozenset(
            parameter.name
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        )
        self._required_names = tuple(
            parameter.name
            for parameter in parameters
            if parameter.default is inspect.Parameter.empty
        )
        self._defaults = {
            parameter.name: parameter.default
            for parameter in parameters
            if parameter.default is not inspect.Parameter.empty
        }

    def __call__(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not self._fast_path or len(args) > len(self._positional_names):
            return self._bind_with_inspect(args, kwargs)

        arguments = dict(zip(self._positional_names, args, strict=False))
        for name, value in kwargs.items():
            if name not in self._keyword_names or name in arguments:
                return self._bind_with_inspect(args, kwargs)
            arguments[name] = value

        if any(name not in arguments for name in self._required_names):
            return self._bind_with_inspect(args, kwargs)

        return {
            name: arguments[name] if name in arguments else self._defaults[name]
            for name in self._names
        }

    def _bind_with_inspect(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)


@dataclass(frozen=True)
class KernelRegistration:
    dispatcher: KernelDispatcher
    expected_source_hash: str
    expected_signature_hash: str
    allowed_metadata: frozenset[str] = field(default_factory=frozenset)
    allowed_grid_dims: frozenset[int] = field(
        default_factory=lambda: frozenset({1, 2, 3})
    )
    owner: str = "torch_mcpu"
    source_version: str = "unversioned"


class KernelRegistry:
    """Strict registry mapping decorated Triton kernels to XCPU adapters."""

    def __init__(self) -> None:
        self._registrations: dict[str, KernelRegistration] = {}
        self._launch_counts: dict[str, int] = {}

    def clear(self) -> None:
        self._registrations.clear()
        self._launch_counts.clear()

    def register(
        self,
        kernel: FakeJITFunction,
        dispatcher: KernelDispatcher,
        *,
        expected_source_hash: str,
        expected_signature_hash: str | None = None,
        allowed_metadata: Sequence[str] = (),
        allowed_grid_dims: Sequence[int] = (1, 2, 3),
        owner: str = "torch_mcpu",
        source_version: str = "unversioned",
        defer_version_check: bool = False,
    ) -> None:
        expected_signature_hash = expected_signature_hash or kernel.signature_hash
        if not defer_version_check and kernel.source_hash != expected_source_hash:
            raise KernelVersionError(
                f"{kernel.qualname}: source fingerprint mismatch; "
                f"expected {expected_source_hash}, got {kernel.source_hash}. "
                f"The XCPU replacement was reviewed against {source_version}. "
                "Manually review the upstream Triton source diff and semantic "
                "changes, update the XCPU implementation and differential tests, "
                "then update only this kernel's hashes and source_version in "
                "vllm_xcpu_plugin.fake_triton.vllm_kernels._KERNELS."
            )
        if not defer_version_check and kernel.signature_hash != expected_signature_hash:
            raise KernelVersionError(
                f"{kernel.qualname}: signature fingerprint mismatch; "
                f"expected {expected_signature_hash}, got {kernel.signature_hash}. "
                f"The XCPU replacement was reviewed against {source_version}. "
                "Manually review the upstream Triton signature and semantics, "
                "update the adapter and differential tests, then update this "
                "kernel's hashes and source_version in "
                "vllm_xcpu_plugin.fake_triton.vllm_kernels._KERNELS."
            )
        unknown_metadata = set(allowed_metadata) - _TRITON_LAUNCH_METADATA
        if unknown_metadata:
            raise ValueError(
                f"Unsupported Triton launch metadata: {sorted(unknown_metadata)}"
            )
        grid_dims = frozenset(allowed_grid_dims)
        if not grid_dims or not grid_dims <= {1, 2, 3}:
            raise ValueError("allowed_grid_dims must be a non-empty subset of 1, 2, 3")

        registration = KernelRegistration(
            dispatcher=dispatcher,
            expected_source_hash=expected_source_hash,
            expected_signature_hash=expected_signature_hash,
            allowed_metadata=frozenset(allowed_metadata),
            allowed_grid_dims=grid_dims,
            owner=owner,
            source_version=source_version,
        )
        previous = self._registrations.get(kernel.qualname)
        if previous is not None and previous != registration:
            raise KernelVersionError(f"Conflicting registration for {kernel.qualname}")
        self._registrations[kernel.qualname] = registration
        self._launch_counts.setdefault(kernel.qualname, 0)

    def dispatch(
        self,
        kernel: FakeJITFunction,
        grid: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        registration = self._registrations.get(kernel.qualname)
        if registration is None:
            raise UnknownKernelError(
                f"Fake Triton kernel is not registered: {kernel.qualname}"
            )
        if (
            kernel.source_hash != registration.expected_source_hash
            or kernel.signature_hash != registration.expected_signature_hash
        ):
            raise KernelVersionError(
                f"{kernel.qualname}: decorated kernel changed after registration. "
                "The XCPU replacement was reviewed against "
                f"{registration.source_version}; manually review the upstream "
                "change, update the implementation and differential tests, then "
                "update only this kernel's hashes and source_version in "
                "vllm_xcpu_plugin.fake_triton.vllm_kernels._KERNELS."
            )

        kernel_kwargs = {}
        metadata = {}
        for name, value in kwargs.items():
            if name not in kernel.parameter_names and name in _TRITON_LAUNCH_METADATA:
                metadata[name] = value
            else:
                kernel_kwargs[name] = value
        unsupported = set(metadata) - registration.allowed_metadata
        if unsupported:
            raise InvalidLaunchError(
                f"{kernel.qualname}: unsupported launch metadata {sorted(unsupported)}"
            )

        try:
            arguments = kernel.bind_arguments(args, kernel_kwargs)
        except TypeError as exc:
            raise InvalidLaunchError(f"{kernel.qualname}: {exc}") from exc
        resolved_grid = _resolve_grid(grid, arguments)
        if len(resolved_grid) not in registration.allowed_grid_dims:
            raise InvalidLaunchError(
                f"{kernel.qualname}: {len(resolved_grid)}D grid is not allowed"
            )

        launch = KernelLaunch(
            kernel=kernel,
            grid=resolved_grid,
            arguments=arguments,
            metadata=metadata,
        )
        result = registration.dispatcher(launch)
        self._launch_counts[kernel.qualname] += 1
        return result

    def launch_counts(self) -> dict[str, int]:
        return dict(self._launch_counts)

    def registrations(self) -> dict[str, KernelRegistration]:
        return dict(self._registrations)


def _resolve_grid(grid: Any, arguments: Mapping[str, Any]) -> tuple[int, ...]:
    if callable(grid):
        grid = grid(dict(arguments))
    if isinstance(grid, int):
        grid = (grid,)
    if not isinstance(grid, (tuple, list)) or not 1 <= len(grid) <= 3:
        raise InvalidLaunchError("grid must be an int or a 1D-3D tuple/list")
    for value in grid:
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidLaunchError("grid dimensions must be integers")
        if value < 0:
            raise InvalidLaunchError("grid dimensions must be non-negative")
    return tuple(grid)


class _KernelLauncher:
    def __init__(self, kernel: FakeJITFunction, grid: Any) -> None:
        self._kernel = kernel
        self._grid = grid

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._kernel.registry.dispatch(self._kernel, self._grid, args, kwargs)


class FakeJITFunction:
    """Import-compatible Triton JIT function with registry-backed launch."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        registry: KernelRegistry | None = None,
        jit_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.fn = fn
        self.registry = registry or get_registry()
        self.jit_options = dict(jit_options or {})
        self.signature = inspect.signature(fn)
        self._argument_binder = _ArgumentBinder(self.signature)
        self.parameter_names = self._argument_binder.parameter_names
        self.source_hash = _source_fingerprint(fn)
        self.signature_hash = _signature_fingerprint(self.signature)
        self.constexpr_names = tuple(
            parameter.name
            for parameter in self.signature.parameters.values()
            if _is_constexpr(parameter.annotation)
        )
        self.qualname = f"{fn.__module__}.{fn.__qualname__}"
        update_wrapper(self, fn)

    def bind_arguments(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self._argument_binder(args, kwargs)

    def __getitem__(self, grid: Any) -> _KernelLauncher:
        return _KernelLauncher(self, grid)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise InvalidLaunchError(
            f"{self.qualname}: Fake Triton kernels must be launched with "
            "kernel[grid](...)"
        )

    def run(self, *args: Any, grid: Any, **kwargs: Any) -> Any:
        return self[grid](*args, **kwargs)

    def warmup(self, *args: Any, grid: Any, **kwargs: Any) -> Any:
        return self[grid](*args, **kwargs)


_REGISTRY = KernelRegistry()


def get_registry() -> KernelRegistry:
    return _REGISTRY


def jit(
    fn: Callable[..., Any] | None = None,
    **options: Any,
) -> FakeJITFunction | Callable[[Callable[..., Any]], FakeJITFunction]:
    def decorate(function: Callable[..., Any]) -> FakeJITFunction:
        if isinstance(function, FakeJITFunction):
            return function
        return FakeJITFunction(function, jit_options=options)

    if fn is not None:
        return decorate(fn)
    return decorate


def passthrough_decorator(*decorator_args: Any, **decorator_kwargs: Any) -> Any:
    del decorator_kwargs
    if decorator_args and callable(decorator_args[0]):
        return decorator_args[0]

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        return function

    return decorate
