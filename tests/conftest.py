import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--data-mode",
        action="store",
        choices=("rand", "data_generate", "data_load"),
        default=None,
        help="kernel test data mode. Defaults to DATA_MODE or rand.",
    )
    parser.addoption(
        "--data-dir",
        action="store",
        default=None,
        help=(
            "directory used by data_generate/data_load modes. "
            "Defaults to test/ops/data."
        ),
    )


@pytest.fixture
def default_vllm_config():
    """Set a default VllmConfig for tests that directly test CustomOps or pathways
    that use get_current_vllm_config() outside of a full engine context.
    """
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield
