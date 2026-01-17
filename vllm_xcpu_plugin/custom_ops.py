import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    import torch_xcpu.ops as ops

    out = torch.empty_like(x)
    ops.rms_norm(
        out,
        x,
        weight,
        variance_epsilon,
    )
    return out


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_xcpu.ops as ops

    ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return x, residual


def rotary_embedding_xcpu(
    x,
    cos,
    sin,
    interleaved: bool = False,
):
    """
    C++ 算子入口。
    x: [num_tokens, num_heads, rotary_dim] - 已经是 3D 且只包含 rotary 部分
    cos: [num_tokens, rotary_dim // 2]
    sin: [num_tokens, rotary_dim // 2]
    """
    return torch.ops.torch_xcpu.rotary_embedding(
        x,
        cos,
        sin,
        interleaved,
    )


@RMSNorm.register_oot
class XcpuRMSNorm(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)

        if current_platform.is_cpu():
            self._forward_method = self.forward_cpu

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        add_residual = residual is not None
        if add_residual:
            assert residual is not None
            assert self.weight.data is not None
            return fused_add_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )
        else:
            return rms_norm(x, self.weight.data, self.variance_epsilon)


@RotaryEmbedding.register_oot
class XcpuRotaryEmbedding(RotaryEmbedding):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if current_platform.is_cpu():
            self._forward_method = self.forward_cpu

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_interleaved: bool,
    ) -> torch.Tensor:
        """
        对 3D tensor [num_tokens, num_heads, rotary_dim] 应用旋转嵌入。
        返回旋转后的 tensor（相同形状）。
        """
        orig_dtype = x.dtype
        
        # 强制转为 float32 且连续
        x_f32 = x.to(dtype=torch.float32).contiguous()
        
        # 调用 C++ 算子
        x_out_f32 = rotary_embedding_xcpu(x_f32, cos, sin, is_interleaved)
        
        # 转回原始类型
        if orig_dtype != torch.float32:
            return x_out_f32.to(dtype=orig_dtype)
        return x_out_f32

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        CPU 前向传播，参考 vLLM 原始的 forward_static 实现。
        """
        # 1. 准备 positions
        positions_flat = positions.flatten()
        num_tokens = positions_flat.shape[0]
        
        # 2. 从 cos_sin_cache 中提取对应的 cos 和 sin
        cos_sin_selected = self.cos_sin_cache.index_select(0, positions_flat)
        cos, sin = cos_sin_selected.chunk(2, dim=-1)
        
        # 强制转为 float32 且连续
        cos = cos.to(dtype=torch.float32).contiguous()
        sin = sin.to(dtype=torch.float32).contiguous()
        
        is_interleaved = not self.is_neox_style
        
        # 3. 保存原始形状，reshape 成 3D [num_tokens, num_heads, head_size]
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        
        # 4. 只对 rotary_dim 部分应用旋转，剩余部分 pass-through
        query_rot = query[..., :self.rotary_dim].contiguous()
        query_pass = query[..., self.rotary_dim:]
        
        # 5. 应用旋转嵌入
        query_rot = self._apply_rotary(query_rot, cos, sin, is_interleaved)
        
        # 6. 拼接并恢复原始形状
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        
        # 7. 对 key 做同样的处理
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., :self.rotary_dim].contiguous()
            key_pass = key[..., self.rotary_dim:]
            key_rot = self._apply_rotary(key_rot, cos, sin, is_interleaved)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        
        return query, key