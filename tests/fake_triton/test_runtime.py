# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_xcpu_plugin.fake_triton.runtime import (
    FakeJITFunction,
    InvalidLaunchError,
    KernelRegistry,
    KernelVersionError,
    UnknownKernelError,
    jit,
)


class _Constexpr:
    name = "constexpr"


def _kernel(output, value, BLOCK_SIZE: _Constexpr = 16):
    del output, value, BLOCK_SIZE


@pytest.fixture
def kernel_and_registry():
    registry = KernelRegistry()
    return FakeJITFunction(_kernel, registry=registry), registry


def test_dispatch_binds_arguments_and_callable_grid(kernel_and_registry):
    kernel, registry = kernel_and_registry
    launches = []
    registry.register(
        kernel,
        launches.append,
        expected_source_hash=kernel.source_hash,
        allowed_grid_dims=(1,),
    )

    output = object()
    kernel[lambda meta: (meta["BLOCK_SIZE"] // 4,)](output, 7, BLOCK_SIZE=20)

    assert len(launches) == 1
    assert launches[0].grid == (5,)
    assert launches[0].arguments == {
        "output": output,
        "value": 7,
        "BLOCK_SIZE": 20,
    }
    assert registry.launch_counts()[kernel.qualname] == 1


def test_unknown_kernel_fails_closed(kernel_and_registry):
    kernel, _ = kernel_and_registry
    with pytest.raises(UnknownKernelError, match="not registered"):
        kernel[(1,)](object(), 1)


def test_source_version_mismatch_is_rejected(kernel_and_registry):
    kernel, registry = kernel_and_registry
    with pytest.raises(KernelVersionError) as exc_info:
        registry.register(
            kernel,
            lambda launch: None,
            expected_source_hash="wrong",
            source_version="v0.19.0",
        )
    message = str(exc_info.value)
    assert "source fingerprint mismatch" in message
    assert "v0.19.0" in message
    assert "Manually review" in message
    assert "source_version" in message


def test_deferred_version_mismatch_fails_when_launched(kernel_and_registry):
    kernel, registry = kernel_and_registry
    registry.register(
        kernel,
        lambda launch: None,
        expected_source_hash="wrong",
        source_version="v0.24.0",
        defer_version_check=True,
    )

    with pytest.raises(KernelVersionError, match="changed after registration"):
        kernel[(1,)](object(), 1)


def test_launch_metadata_is_explicitly_allowlisted(kernel_and_registry):
    kernel, registry = kernel_and_registry
    registry.register(
        kernel,
        lambda launch: None,
        expected_source_hash=kernel.source_hash,
    )
    with pytest.raises(InvalidLaunchError, match="unsupported launch metadata"):
        kernel[(1,)](object(), 1, num_warps=4)


def test_allowlisted_launch_metadata_reaches_adapter(kernel_and_registry):
    kernel, registry = kernel_and_registry
    launches = []
    registry.register(
        kernel,
        launches.append,
        expected_source_hash=kernel.source_hash,
        allowed_metadata=("num_warps",),
    )
    kernel[(1,)](object(), 1, num_warps=4)
    assert launches[0].metadata == {"num_warps": 4}


def test_conflicting_duplicate_registration_is_rejected(kernel_and_registry):
    kernel, registry = kernel_and_registry
    registry.register(
        kernel,
        lambda launch: None,
        expected_source_hash=kernel.source_hash,
    )
    with pytest.raises(KernelVersionError, match="Conflicting registration"):
        registry.register(
            kernel,
            lambda launch: launch,
            expected_source_hash=kernel.source_hash,
        )


def test_invalid_grid_and_direct_call_fail(kernel_and_registry):
    kernel, registry = kernel_and_registry
    registry.register(
        kernel,
        lambda launch: None,
        expected_source_hash=kernel.source_hash,
    )
    with pytest.raises(InvalidLaunchError, match="non-negative"):
        kernel[(-1,)](object(), 1)
    with pytest.raises(InvalidLaunchError, match="must be launched"):
        kernel(object(), 1)


def test_jit_supports_decorator_with_and_without_parentheses():
    @jit
    def direct(value):
        del value

    @jit(do_not_specialize=["value"])
    def configured(value):
        del value

    assert isinstance(direct, FakeJITFunction)
    assert isinstance(configured, FakeJITFunction)
    assert configured.jit_options == {"do_not_specialize": ["value"]}
