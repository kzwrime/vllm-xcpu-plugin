from setuptools import setup

setup(
    name='vllm_xcpu_plugin',
    version='0.1',
    packages=['vllm_xcpu_plugin'],
    entry_points={
        # 'vllm.platform_plugins': [
        #     "xcpu_platform_plugin = vllm_xcpu_plugin:xcpu_platform_plugin"  # noqa
        # ],
        "vllm.general_plugins":
        ["xcpu_custom_ops = vllm_xcpu_plugin:register_ops"],
    })
