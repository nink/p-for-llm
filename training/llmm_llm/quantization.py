"""Native PLE-MoE-W1.58A8 training components."""

from __future__ import annotations

import hashlib
import pickle

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchao.prototype.quantized_training import (
    BitNetTrainingLinearWeight,
    bitnet_training,
)
from torchao.prototype.quantized_training.bitnet import (
    quantize_bitnet_weight,
)
from torchao.prototype.quantized_training.int8 import quantize_int8_rowwise
from torchao.quantization import quantize_

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only development environments.
    triton = None


QUANTIZATION_FORMAT = "PLE-MoE-W1.58A8"


def _bitnet_stable_hash_for_caching(weight: BitNetTrainingLinearWeight) -> str:
    """Provide the missing TorchAO hook used by AOTAutograd's disk cache."""
    from torch._inductor.codecache import extract_tensor_metadata_for_cache_key

    inner_tensor_names, subclass_metadata = weight.__tensor_flatten__()
    inner_metadata = tuple(
        (
            name,
            extract_tensor_metadata_for_cache_key(getattr(weight, name)),
        )
        for name in inner_tensor_names
    )
    payload = (
        f"{type(weight).__module__}.{type(weight).__qualname__}",
        weight.shape,
        weight.requires_grad,
        subclass_metadata,
        inner_metadata,
    )
    return hashlib.blake2b(pickle.dumps(payload), digest_size=16).hexdigest()


if not hasattr(BitNetTrainingLinearWeight, "_stable_hash_for_caching"):
    BitNetTrainingLinearWeight._stable_hash_for_caching = (
        _bitnet_stable_hash_for_caching
    )


if triton is not None:

    @triton.jit
    def _grouped_scaled_int8_mm_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        row_scale_ptr,
        weight_scale_ptr,
        group_expert_ptr,
        rows,
        output_features,
        input_features,
        input_group_stride,
        input_row_stride,
        input_feature_stride,
        weight_expert_stride,
        weight_input_stride,
        weight_output_stride,
        output_group_stride,
        output_row_stride,
        output_feature_stride,
        row_scale_group_stride,
        row_scale_row_stride,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        tile = tl.program_id(0)
        group = tl.program_id(1)
        grid_n = tl.cdiv(output_features, BLOCK_N)
        tile_m = tile // grid_n
        tile_n = tile % grid_n
        row_offsets = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        output_offsets = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        input_offsets = tl.arange(0, BLOCK_K)
        expert = tl.load(group_expert_ptr + group)

        input_block = (
            input_ptr
            + group * input_group_stride
            + row_offsets[:, None] * input_row_stride
            + input_offsets[None, :] * input_feature_stride
        )
        weight_block = (
            weight_ptr
            + expert * weight_expert_stride
            + input_offsets[:, None] * weight_input_stride
            + output_offsets[None, :] * weight_output_stride
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for remaining in range(input_features, 0, -BLOCK_K):
            input_mask = (row_offsets[:, None] < rows) & (
                input_offsets[None, :] < remaining
            )
            weight_mask = (
                input_offsets[:, None] < remaining
            ) & (output_offsets[None, :] < output_features)
            input_values = tl.load(input_block, mask=input_mask, other=0)
            weight_values = tl.load(weight_block, mask=weight_mask, other=0)
            accumulator += tl.dot(input_values, weight_values)
            input_block += BLOCK_K * input_feature_stride
            weight_block += BLOCK_K * weight_input_stride

        row_scale = tl.load(
            row_scale_ptr
            + group * row_scale_group_stride
            + row_offsets * row_scale_row_stride,
            mask=row_offsets < rows,
            other=0.0,
        ).to(tl.float32)
        weight_scale = tl.load(weight_scale_ptr + expert).to(tl.float32)
        output = accumulator.to(tl.float32) * row_scale[:, None] * weight_scale
        output_block = (
            output_ptr
            + group * output_group_stride
            + row_offsets[:, None] * output_row_stride
            + output_offsets[None, :] * output_feature_stride
        )
        output_mask = (row_offsets[:, None] < rows) & (
            output_offsets[None, :] < output_features
        )
        tl.store(output_block, output, mask=output_mask)


    @triton.jit
    def _grouped_scaled_int8_mm_grad_input_kernel(
        grad_output_ptr,
        weight_ptr,
        grad_input_ptr,
        weight_scale_ptr,
        group_expert_ptr,
        rows,
        output_features,
        input_features,
        grad_output_group_stride,
        grad_output_row_stride,
        grad_output_feature_stride,
        weight_expert_stride,
        weight_output_stride,
        weight_input_stride,
        grad_input_group_stride,
        grad_input_row_stride,
        grad_input_feature_stride,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        tile = tl.program_id(0)
        group = tl.program_id(1)
        grid_n = tl.cdiv(input_features, BLOCK_N)
        tile_m = tile // grid_n
        tile_n = tile % grid_n
        row_offsets = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        input_offsets = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        output_offsets = tl.arange(0, BLOCK_K)
        expert = tl.load(group_expert_ptr + group)

        grad_output_block = (
            grad_output_ptr
            + group * grad_output_group_stride
            + row_offsets[:, None] * grad_output_row_stride
            + output_offsets[None, :] * grad_output_feature_stride
        )
        weight_block = (
            weight_ptr
            + expert * weight_expert_stride
            + output_offsets[:, None] * weight_output_stride
            + input_offsets[None, :] * weight_input_stride
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for remaining in range(output_features, 0, -BLOCK_K):
            grad_output_mask = (row_offsets[:, None] < rows) & (
                output_offsets[None, :] < remaining
            )
            weight_mask = (
                output_offsets[:, None] < remaining
            ) & (input_offsets[None, :] < input_features)
            grad_output_values = tl.load(
                grad_output_block,
                mask=grad_output_mask,
                other=0.0,
            ).to(tl.bfloat16)
            weight_values = tl.load(
                weight_block,
                mask=weight_mask,
                other=0,
            ).to(tl.bfloat16)
            accumulator += tl.dot(grad_output_values, weight_values)
            grad_output_block += BLOCK_K * grad_output_feature_stride
            weight_block += BLOCK_K * weight_output_stride

        weight_scale = tl.load(weight_scale_ptr + expert).to(tl.float32)
        grad_input = accumulator * weight_scale
        grad_input_block = (
            grad_input_ptr
            + group * grad_input_group_stride
            + row_offsets[:, None] * grad_input_row_stride
            + input_offsets[None, :] * grad_input_feature_stride
        )
        grad_input_mask = (row_offsets[:, None] < rows) & (
            input_offsets[None, :] < input_features
        )
        tl.store(grad_input_block, grad_input, mask=grad_input_mask)


def _grouped_scaled_int8_mm(
    input_i8: Tensor,
    weight_i8: Tensor,
    row_scales: Tensor,
    weight_scales: Tensor,
    group_experts: Tensor,
) -> Tensor:
    """Run independent expert INT8 GEMMs in one two-dimensional Triton launch."""
    if triton is None:
        raise RuntimeError("CUDA grouped BitNet training requires Triton")
    if input_i8.ndim != 3 or weight_i8.ndim != 3:
        raise ValueError("grouped INT8 GEMM expects rank-3 input and weight tensors")
    groups, rows, input_features = input_i8.shape
    experts, weight_input_features, output_features = weight_i8.shape
    if (
        input_i8.dtype is not torch.int8
        or weight_i8.dtype is not torch.int8
        or input_features != weight_input_features
        or row_scales.shape != (groups, rows)
        or weight_scales.shape != (experts,)
        or group_experts.shape != (groups,)
    ):
        raise ValueError("invalid grouped INT8 GEMM tensor shapes or dtypes")

    output = torch.empty(
        (groups, rows, output_features),
        device=input_i8.device,
        dtype=row_scales.dtype,
    )
    block_k, num_stages = (
        (64, 3)
        if (input_features, output_features) == (192, 512)
        else (32, 4)
    )
    grid = (triton.cdiv(rows, 128) * triton.cdiv(output_features, 128), groups)
    _grouped_scaled_int8_mm_kernel[grid](
        input_i8,
        weight_i8,
        output,
        row_scales,
        weight_scales,
        group_experts,
        rows,
        output_features,
        input_features,
        *input_i8.stride(),
        *weight_i8.stride(),
        *output.stride(),
        *row_scales.stride(),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=num_stages,
    )
    return output


def _grouped_scaled_int8_mm_grad_input(
    grad_output: Tensor,
    weight_i8: Tensor,
    weight_scales: Tensor,
    group_experts: Tensor,
) -> Tensor:
    """Backpropagate grouped BitNet inputs without expanding selected weights."""
    if triton is None:
        raise RuntimeError("CUDA grouped BitNet training requires Triton")
    groups, rows, output_features = grad_output.shape
    experts, weight_output_features, input_features = weight_i8.shape
    if (
        grad_output.dtype is not torch.bfloat16
        or weight_i8.dtype is not torch.int8
        or output_features != weight_output_features
        or weight_scales.shape != (experts,)
        or group_experts.shape != (groups,)
    ):
        raise ValueError("invalid grouped BitNet input-gradient tensor shapes or dtypes")

    grad_input = torch.empty(
        (groups, rows, input_features),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    block_k, num_stages = (
        (64, 3)
        if (output_features, input_features) == (512, 192)
        else (32, 4)
    )
    grid = (
        triton.cdiv(rows, 128) * triton.cdiv(input_features, 128),
        groups,
    )
    _grouped_scaled_int8_mm_grad_input_kernel[grid](
        grad_output,
        weight_i8,
        grad_input,
        weight_scales,
        group_experts,
        rows,
        output_features,
        input_features,
        *grad_output.stride(),
        *weight_i8.stride(),
        *grad_input.stride(),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=num_stages,
    )
    return grad_input


class _GroupedBitNetTrainingLinear(torch.autograd.Function):
    """Batched W1.58A8 projection with one contiguous expert-weight bank."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        group_experts: Tensor,
        weights: Tensor,
    ) -> Tensor:
        if weights.ndim != 3:
            raise ValueError("grouped BitNet linear expects [experts, out, in] weights")
        groups, rows, input_features = input.shape
        input_i8, row_scales = quantize_int8_rowwise(
            input.reshape(-1, input_features),
            eps=1e-5,
        )
        input_i8 = input_i8.reshape(groups, rows, input_features).contiguous()
        row_scales = row_scales.reshape(groups, rows).contiguous()

        master_weights = weights.to(dtype=input.dtype)
        weight_scales_fp32 = master_weights.float().abs().mean(dim=(1, 2))
        weight_i8 = quantize_bitnet_weight(
            master_weights,
            weight_scales_fp32[:, None, None],
        ).contiguous()
        weight_scales = weight_scales_fp32.to(dtype=input.dtype).contiguous()
        output = _grouped_scaled_int8_mm(
            input_i8,
            weight_i8.transpose(1, 2).contiguous(),
            row_scales,
            weight_scales,
            group_experts.contiguous(),
        )
        ctx.save_for_backward(
            input_i8,
            row_scales,
            weight_i8,
            weight_scales,
            group_experts,
        )
        ctx.weight_dtype = weights.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            input_i8,
            row_scales,
            weight_i8,
            weight_scales,
            group_experts,
        ) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = None
        if ctx.needs_input_grad[0]:
            if grad_output.dtype is torch.bfloat16:
                grad_input = _grouped_scaled_int8_mm_grad_input(
                    grad_output,
                    weight_i8,
                    weight_scales,
                    group_experts,
                )
            else:
                selected_weights = weight_i8.index_select(0, group_experts).to(
                    grad_output.dtype
                )
                selected_scales = weight_scales.index_select(0, group_experts)
                grad_input = torch.bmm(grad_output, selected_weights)
                grad_input = grad_input * selected_scales[:, None, None]

        weight_grad_needed = ctx.needs_input_grad[2]
        if weight_grad_needed:
            dequantized_input = input_i8.to(row_scales.dtype) * row_scales.unsqueeze(
                -1
            )
            group_weight_grads = torch.bmm(
                grad_output.transpose(1, 2),
                dequantized_input,
            )
            weight_grads = torch.zeros(
                weight_i8.shape,
                device=grad_output.device,
                dtype=group_weight_grads.dtype,
            )
            weight_grads.index_add_(0, group_experts, group_weight_grads)
            returned_weight_grad = weight_grads.to(dtype=ctx.weight_dtype)
        else:
            returned_weight_grad = None
        return grad_input, None, returned_weight_grad


def grouped_bitlinear(
    input: Tensor,
    group_experts: Tensor,
    weights: Tensor,
) -> Tensor:
    """Apply one BitNet projection to many selected experts at once on CUDA."""
    if input.device.type != "cuda":
        raise ValueError("grouped BitNet linear is CUDA-only")
    if torch.is_autocast_enabled("cuda"):
        autocast_dtype = torch.get_autocast_gpu_dtype()
        input = input.to(dtype=autocast_dtype)
    return _GroupedBitNetTrainingLinear.apply(input, group_experts, weights)


def ternary_ste_linear(input: Tensor, weight: Tensor) -> Tensor:
    """Reference W1.58A8 linear path for one packed expert on CPU."""
    scale = weight.float().abs().mean()
    codes = quantize_bitnet_weight(weight, scale)
    quantized = codes.to(dtype=weight.dtype) * scale.to(dtype=weight.dtype)
    ste_weight = weight + (quantized - weight).detach()
    return F.linear(input, ste_weight)


class MixedDtypeRMSNorm(nn.RMSNorm):
    """Use a fused norm dtype without lowering the FP32 master parameter."""

    def normalize(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.normalized_shape, None, self.eps)

    def apply_weight(self, normalized: Tensor) -> Tensor:
        weight = self.weight
        if weight is not None and weight.dtype != normalized.dtype:
            weight = weight.to(dtype=normalized.dtype)
        return normalized if weight is None else normalized * weight

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        if weight is not None and weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)


class BitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.input_norm = MixedDtypeRMSNorm(in_features, eps=rms_norm_eps)

    def _linear(self, x: Tensor) -> Tensor:
        input_shape = x.shape
        flat_x = x.reshape(-1, self.in_features)
        if flat_x.size(0) == 1:
            # TorchAO's INT8 kernel requires at least two activation rows.
            flat_x = torch.cat((flat_x, torch.zeros_like(flat_x)), dim=0)
            output = F.linear(flat_x, self.weight, self.bias)[:1]
        else:
            output = F.linear(flat_x, self.weight, self.bias)
        return output.reshape(*input_shape[:-1], self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self._linear(self.input_norm(x))

    def normalize_input(self, x: Tensor) -> Tensor:
        return self.input_norm.normalize(x)

    def forward_normalized(self, normalized: Tensor) -> Tensor:
        return self._linear(self.input_norm.apply_weight(normalized))


class TernaryEmbedding(nn.Embedding):
    """Embedding with a high-precision master weight and ternary STE forward."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["sparse"] = True
        super().__init__(*args, **kwargs)
        self.register_buffer("weight_abs_sum", self.weight.new_zeros(()))
        self.register_buffer(
            "row_abs_sums",
            self.weight.new_zeros(self.num_embeddings),
        )

    @torch.no_grad()
    def reset_absmean_cache(self) -> None:
        self.row_abs_sums.copy_(self.weight.float().abs().sum(dim=1))
        self.weight_abs_sum.copy_(self.row_abs_sums.sum())

    def absmean_scale(self) -> Tensor:
        return self.weight_abs_sum / self.weight.numel()

    @torch.no_grad()
    def prepare_absmean_update(self) -> Tensor | None:
        gradient = self.weight.grad
        if gradient is None:
            return None
        if not gradient.is_sparse:
            raise RuntimeError("PLE table gradient must remain sparse")
        gradient = gradient.coalesce()
        self.weight.grad = gradient
        return gradient.indices()[0]

    @torch.no_grad()
    def finish_absmean_update(
        self,
        rows: Tensor | None,
    ) -> None:
        if rows is None:
            return
        old_row_sums = self.row_abs_sums.index_select(0, rows)
        new_row_sums = (
            self.weight.index_select(0, rows).float().abs().sum(dim=1)
        )
        self.weight_abs_sum.add_((new_row_sums - old_row_sums).sum())
        self.row_abs_sums.index_copy_(0, rows, new_row_sums)

    def quantized_weight(self, token_ids: Tensor | None = None) -> Tensor:
        weight = self.weight
        if token_ids is not None:
            weight = F.embedding(
                token_ids,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
        scale = self.absmean_scale()
        codes = quantize_bitnet_weight(weight, scale)
        quantized = codes.to(weight.dtype) * scale.to(weight.dtype)
        return weight + (quantized - weight).detach()

    @torch.compiler.disable
    def forward(self, token_ids: Tensor) -> Tensor:
        return self.quantized_weight(token_ids)


def enable_bitnet_training(module: nn.Module) -> None:
    quantize_(
        module,
        bitnet_training(),
        filter_fn=lambda candidate, _name: isinstance(candidate, BitLinear),
    )
