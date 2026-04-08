# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

# Reference default values of atol and rtol are from
# https://github.com/pytorch/pytorch/blob/6d96beb6bec24d73ee3f080bac54d2104068f675/test/test_transformers.py#L67
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-2, torch.float: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float: 1.3e-6}
default_dice_tol = 5e-6


def get_default_atol(output) -> float:
    return default_atol[output.dtype]


def get_default_rtol(output) -> float:
    return default_rtol[output.dtype]


# https://github.com/deepseek-ai/DeepGEMM/blob/d30fc36c8f229f4f873b90a492f6e19e6e610923/deep_gemm/testing/numeric.py#L5
def calc_diff(x: torch.Tensor, y: torch.Tensor):
    # 将输入张量转换为双精度浮点型以提高计算精度
    x, y = x.double(), y.double()

    # 计算分母：x 的平方和加上 y 的平方和
    denominator = (x * x + y * y).sum()

    # 如果分母为 0，意味着 x 和 y 中的所有元素均为 0
    if denominator == 0:
        return 0.0

    # 计算相似度 (Sørensen–Dice Coefficient)
    # 公式：2 * (x · y) / (||x||² + ||y||²)
    sim = 2 * (x * y).sum() / denominator

    # 返回差异度 (1 - 相似度)
    return 1 - sim
