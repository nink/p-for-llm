"""CUDA fused output projection and causal cross entropy for LLMM training."""

from __future__ import annotations

import torch
import triton
from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import amp_custom_bwd, amp_custom_fwd, element_mul_kernel
from liger_kernel.ops.utils import is_hip
from torch import Tensor


_MAX_FUSED_SIZE = 65536 // 2
DEFAULT_LOSS_CHUNK_SIZE = 4096


class _FixedChunkFusedLinearCrossEntropy(torch.autograd.Function):
    """Compute exact causal loss with bounded full-vocabulary chunks."""

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        hidden: Tensor,
        weight: Tensor,
        target: Tensor,
        ignore_index: int,
        non_ignore_count: int,
        chunk_size: int,
    ) -> Tensor:
        token_count, hidden_size = hidden.shape
        vocab_size = weight.shape[0]
        chunk_size = min(chunk_size, token_count)
        block_size = min(_MAX_FUSED_SIZE, triton.next_power_of_2(vocab_size))

        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight)
        # Liger leaves ignore_index loss slots untouched when its kernel returns.
        losses = torch.zeros(token_count, dtype=torch.float32, device=hidden.device)

        for start in range(0, token_count, chunk_size):
            end = min(start + chunk_size, token_count)
            hidden_chunk = hidden[start:end]
            target_chunk = target[start:end].contiguous()
            grad_logits = (hidden_chunk @ weight.t()).contiguous()
            loss_chunk = losses[start:end]

            liger_cross_entropy_kernel[(end - start,)](
                X_ptr=grad_logits,
                X_stride=grad_logits.stride(-2),
                Y_ptr=target_chunk,
                Y_stride=target_chunk.stride(-1),
                weight_ptr=None,
                loss_ptr=loss_chunk,
                z_loss_ptr=None,
                loss_stride=loss_chunk.stride(-1),
                token_accuracy_ptr=None,
                token_accuracy_stride=0,
                predicted_tokens_ptr=None,
                predicted_tokens_stride=0,
                n_cols=vocab_size,
                n_non_ignore=non_ignore_count,
                sum_non_ignore_weight=non_ignore_count,
                weight_sum=0.0,
                ignore_index=ignore_index,
                lse_square_scale=0.0,
                label_smoothing=0.0,
                reduction="mean",
                softcap=None,
                RETURN_Z_LOSS=False,
                RETURN_TOKEN_ACCURACY=False,
                RETURN_PREDICTED_TOKENS=False,
                HAS_WEIGHT=False,
                HAS_SOFTCAPPING=False,
                HAS_GRADIENTS=True,
                BLOCK_SIZE=block_size,
                num_warps=32 if not is_hip() else 16,
            )

            grad_hidden[start:end] = grad_logits @ weight
            grad_logits_t = grad_logits.t()
            accumulation_input = hidden_chunk
            if accumulation_input.dtype != grad_logits_t.dtype:
                accumulation_input = accumulation_input.to(grad_logits_t.dtype)
            torch.addmm(
                grad_weight,
                grad_logits_t,
                accumulation_input,
                out_dtype=torch.float32,
                out=grad_weight,
            )

        ctx.save_for_backward(grad_hidden.detach(), grad_weight.detach())
        return losses.sum()

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        grad_hidden, grad_weight = ctx.saved_tensors
        hidden_size = grad_hidden.shape[-1]
        hidden_block_size = min(
            _MAX_FUSED_SIZE,
            triton.next_power_of_2(hidden_size),
        )
        element_mul_kernel[(grad_hidden.shape[0],)](
            grad_hidden,
            grad_hidden.stride(-2),
            grad_output,
            hidden_size,
            BLOCK_SIZE=hidden_block_size,
            num_warps=32 if not is_hip() else 16,
        )
        element_mul_kernel[(grad_weight.shape[0],)](
            grad_weight,
            grad_weight.stride(-2),
            grad_output,
            hidden_size,
            BLOCK_SIZE=hidden_block_size,
            num_warps=32 if not is_hip() else 16,
        )
        return grad_hidden, grad_weight, None, None, None, None


def fixed_chunk_fused_linear_cross_entropy(
    hidden: Tensor,
    weight: Tensor,
    target: Tensor,
    *,
    non_ignore_count: int,
    chunk_size: int = DEFAULT_LOSS_CHUNK_SIZE,
    ignore_index: int = -100,
) -> Tensor:
    if hidden.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("fused loss expects [tokens, hidden], [vocab, hidden], [tokens]")
    if hidden.shape[0] != target.shape[0] or hidden.shape[1] != weight.shape[1]:
        raise ValueError("fused loss tensor shapes do not match")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 < non_ignore_count <= target.numel():
        raise ValueError("non_ignore_count must be in [1, target.numel()]")
    return _FixedChunkFusedLinearCrossEntropy.apply(
        hidden,
        weight,
        target,
        ignore_index,
        non_ignore_count,
        chunk_size,
    )
