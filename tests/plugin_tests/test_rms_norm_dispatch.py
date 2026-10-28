from __future__ import annotations

import pytest
import torch
import torch_xcpu  # noqa: F401

from vllm_xcpu_plugin.rms_norm_dispatch import _rms_norm, register_rms_norm_kernel


@pytest.fixture(autouse=True, scope="module")
def registered():
    register_rms_norm_kernel()


def _reference(x, weight, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        w = weight.float()
        if w.dim() == 2:
            w = w.view(w.shape[0], *([1] * (x.dim() - 2)), w.shape[1])
        y = y * w
    return y.to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape,grouped",
    [
        ((7, 64), False),
        ((3, 5, 2, 128), True),
        ((3, 5, 2, 128), False),
        ((4, 6, 33), True),
        ((5, 33), False),
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "no_weight", "inplace"])
def test_vllm_rms_norm_matches_reference(dtype, shape, grouped, layout):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    if layout == "strided":
        # Non-unit hidden stride: packed by the kernel.
        x = torch.randn(*shape[:-1], shape[-1] * 2, dtype=dtype)[..., ::2]
    weight = None
    if layout != "no_weight":
        weight = torch.randn(
            (shape[0], shape[-1]) if grouped else (shape[-1],), dtype=dtype
        )
    expected = _reference(x, weight, 1e-6)

    x_dev = x.to("mcpu")
    w_dev = weight.to("mcpu") if weight is not None else None
    out = (
        x_dev if layout == "inplace" else torch.empty(shape, dtype=dtype, device="mcpu")
    )
    torch.ops._C.rms_norm(out, x_dev, w_dev, 1e-6)
    tol = 2e-2 if dtype != torch.float32 else 1e-5
    torch.testing.assert_close(out.cpu(), expected, atol=tol, rtol=tol)

    # The C++ kernel and the Python kernel kept for shape dumps agree bitwise.
    if layout == "inplace":
        x_ref = x.to("mcpu")
        _rms_norm(x_ref, x_ref, w_dev, 1e-6)
        python_out = x_ref
    else:
        python_out = torch.empty(shape, dtype=dtype, device="mcpu")
        _rms_norm(python_out, x_dev, w_dev, 1e-6)
    assert torch.equal(python_out.cpu(), out.cpu())


def test_vllm_rms_norm_rejects_invalid_arguments():
    x = torch.randn(4, 8, device="mcpu")
    with pytest.raises(ValueError, match="rank 2/3/4"):
        torch.ops._C.rms_norm(
            torch.empty(8, device="mcpu"), torch.randn(8, device="mcpu"), None, 1e-6
        )
    with pytest.raises(ValueError, match="shape/dtype"):
        torch.ops._C.rms_norm(torch.empty(4, 9, device="mcpu"), x, None, 1e-6)
    with pytest.raises(ValueError, match=r"\[H\] or \[L, H\]"):
        torch.ops._C.rms_norm(
            torch.empty_like(x), x, torch.randn(7, device="mcpu"), 1e-6
        )
    with pytest.raises(ValueError, match="grouped weight"):
        torch.ops._C.rms_norm(
            torch.empty_like(x), x, torch.randn(3, 8, device="mcpu"), 1e-6
        )
    with pytest.raises(TypeError, match="float32, bfloat16 or float16"):
        xi = torch.ones(4, 8, dtype=torch.int32, device="mcpu")
        torch.ops._C.rms_norm(torch.empty_like(xi), xi, None, 1e-6)


@pytest.mark.parametrize(
    "dump,source", [("0", "vllm_rms_norm.cpp"), ("1", "rms_norm_dispatch.py")]
)
def test_shape_dumps_keep_the_python_kernel(dump, source):
    import os
    import subprocess
    import sys

    # A fresh process: registration is process-wide and happens once.
    code = (
        "import torch, torch_xcpu\n"
        "from vllm_xcpu_plugin.rms_norm_dispatch import register_rms_norm_kernel\n"
        "register_rms_norm_kernel()\n"
        "print([l for l in torch._C._dispatch_dump('_C::rms_norm').splitlines()"
        " if l.startswith('PrivateUse1')])\n"
    )
    env = dict(os.environ, TORCH_XCPU_DUMP_SHAPES=dump)
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert source in out
