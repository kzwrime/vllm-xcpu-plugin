from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

DATA_MODES = {"rand", "data_generate", "data_load"}
DATA_VERSION = 1
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def _pytest_option(pytest_config: pytest.Config | None, name: str) -> str | None:
    if pytest_config is None:
        return None
    try:
        return pytest_config.getoption(name)
    except ValueError:
        return None


def data_mode(pytest_config: pytest.Config | None = None) -> str:
    mode = (
        _pytest_option(pytest_config, "data_mode")
        or os.getenv("DATA_MODE")
        or os.getenv("VLLM_XCPU_DATA_MODE")
        or "rand"
    )
    if mode not in DATA_MODES:
        raise ValueError(
            f"Unsupported data mode {mode!r}. Expected one of {sorted(DATA_MODES)}."
        )
    return mode


def data_dir(pytest_config: pytest.Config | None = None) -> Path:
    path = (
        _pytest_option(pytest_config, "data_dir")
        or os.getenv("DATA_DIR")
        or os.getenv("VLLM_XCPU_DATA_DIR")
    )
    return Path(path) if path else DEFAULT_DATA_DIR


def dtype_id(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def case_id(*parts: object) -> str:
    return "::".join(_case_part(part) for part in parts)


def run_data_mode_case(
    *,
    op_name: str,
    case_name: str,
    build_fn: Callable[[], Mapping[str, Any]],
    run_fn: Callable[[Mapping[str, Any]], Any],
    check_fn: Callable[[Any, Mapping[str, Any]], None],
    pytest_config: pytest.Config | None = None,
) -> None:
    mode = data_mode(pytest_config)
    path = _data_file_path(op_name, case_name, pytest_config)

    if mode == "data_load":
        case = _load_case(path, op_name)
    else:
        case = dict(build_fn())
        if mode == "data_generate":
            _save_case(path, op_name, case_name, case)
            return

    actual = run_fn(case)
    check_fn(actual, case)


def _case_part(part: object) -> str:
    if isinstance(part, torch.dtype):
        return dtype_id(part)
    if isinstance(part, (list, tuple)):
        return "x".join(_case_part(item) for item in part)
    if isinstance(part, dict):
        return "_".join(
            f"{key}={_case_part(value)}" for key, value in sorted(part.items())
        )
    return str(part)


def _safe_filename_part(text: str, max_len: int = 140) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.=+-]+", "_", text).strip("_")
    return safe[:max_len].rstrip("_") or "case"


def _data_file_path(
    op_name: str,
    case_name: str,
    pytest_config: pytest.Config | None,
) -> Path:
    digest = hashlib.sha256(case_name.encode()).hexdigest()[:12]
    filename = f"{_safe_filename_part(case_name)}_{digest}.pt"
    return data_dir(pytest_config) / op_name / filename


def _save_case(
    path: Path, op_name: str, case_name: str, case: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": DATA_VERSION,
        "op": op_name,
        "case_name": case_name,
        "case": _to_cpu(case),
    }
    torch.save(payload, path)


def _load_case(path: Path, op_name: str) -> Mapping[str, Any]:
    if not path.exists():
        pytest.fail(
            f"Missing {op_name} test data: {path}. "
            "Run pytest with --data-mode=data_generate first."
        )
    payload = torch.load(path, map_location="cpu")
    if payload.get("version") != DATA_VERSION:
        pytest.fail(
            f"Unsupported test data version in {path}: {payload.get('version')!r}"
        )
    if payload.get("op") != op_name:
        pytest.fail(f"Unexpected op in test data {path}: {payload.get('op')!r}")
    return payload["case"]


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value
