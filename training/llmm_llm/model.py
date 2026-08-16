import math
from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchao.prototype.quantized_training.bitnet import quantize_bitnet_weight

from .config import ModelConfig
from .fused_loss import (
    DEFAULT_LOSS_CHUNK_SIZE,
    fixed_chunk_fused_linear_cross_entropy,
)
from .quantization import (
    BitLinear,
    MixedDtypeRMSNorm,
    TernaryEmbedding,
    enable_bitnet_training,
    grouped_bitlinear,
    ternary_ste_linear,
)


@dataclass(frozen=True, slots=True)
class LayerKVCache:
    keys: Tensor
    values: Tensor
    length: int


ModelKVCache: TypeAlias = tuple[LayerKVCache, ...]


_CUDA_MOE_DISPATCH_ROWS = 128


def _materialize_ternary_weight(weight: Tensor, dimensions: tuple[int, ...]) -> Tensor:
    master = getattr(weight, "original_weight_tensor", weight)
    scale = master.float().abs().mean(dim=dimensions, keepdim=True)
    codes = quantize_bitnet_weight(master, scale)
    return codes.to(dtype=master.dtype) * scale.to(dtype=master.dtype)


def _build_rope(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    position_offset: int = 0,
) -> tuple[Tensor, Tensor]:
    frequencies = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(
        position_offset,
        position_offset + seq_len,
        device=device,
    ).float()
    angles = torch.outer(positions, frequencies).repeat_interleave(2, dim=-1)
    return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def _rotate_half(x: Tensor) -> Tensor:
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1)
    return rotated.flatten(-2)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + _rotate_half(x) * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.q_proj = BitLinear(
            config.d_model,
            config.n_heads * self.head_dim,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.k_proj = BitLinear(
            config.d_model,
            config.n_kv_heads * self.head_dim,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.v_proj = BitLinear(
            config.d_model,
            config.n_kv_heads * self.head_dim,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.output = BitLinear(
            config.d_model,
            config.d_model,
            rms_norm_eps=config.rms_norm_eps,
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        q, k, v = self._project_qkv(x, cos, sin)
        return self._attend(q, k, v, is_causal=True)

    def _project_qkv(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape
        normalized = self.q_proj.normalize_input(x)
        q = self.q_proj.forward_normalized(normalized).view(
            batch_size, seq_len, self.n_heads, self.head_dim
        )
        k = self.k_proj.forward_normalized(normalized).view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        )
        v = self.v_proj.forward_normalized(normalized).view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        )
        return (
            _apply_rope(q.transpose(1, 2), cos, sin),
            _apply_rope(k.transpose(1, 2), cos, sin),
            v.transpose(1, 2),
        )

    def _attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        is_causal: bool,
    ) -> Tensor:
        batch_size, _, seq_len, _ = q.shape
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            enable_gqa=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.head_dim
        )
        return self.output(attended)

    def forward_cached(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: LayerKVCache | None = None,
    ) -> tuple[Tensor, LayerKVCache]:
        q, k, v = self._project_qkv(x, cos, sin)

        if cache is None:
            cache = LayerKVCache(
                keys=k.new_empty(
                    k.size(0),
                    self.n_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                ),
                values=v.new_empty(
                    v.size(0),
                    self.n_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                ),
                length=0,
            )
        end = cache.length + k.size(2)
        cache.keys[:, :, cache.length:end].copy_(k)
        cache.values[:, :, cache.length:end].copy_(v)
        new_cache = LayerKVCache(cache.keys, cache.values, end)
        attended = self._attend(
            q,
            new_cache.keys[:, :, :end],
            new_cache.values[:, :, :end],
            is_causal=cache.length == 0,
        )
        return attended, new_cache


class PackedSwiGLU(nn.Module):
    """One layer's independent SwiGLU experts in contiguous parameter banks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.rms_norm_eps = config.rms_norm_eps
        self.gate_weight = nn.Parameter(
            torch.empty(config.n_experts, config.ffn_hidden, config.d_model)
        )
        self.up_weight = nn.Parameter(
            torch.empty(config.n_experts, config.ffn_hidden, config.d_model)
        )
        self.down_weight = nn.Parameter(
            torch.empty(config.n_experts, config.d_model, config.ffn_hidden)
        )
        self.gate_norm = nn.Parameter(torch.ones(config.n_experts, config.d_model))
        self.up_norm = nn.Parameter(torch.ones(config.n_experts, config.d_model))
        self.down_norm = nn.Parameter(torch.ones(config.n_experts, config.ffn_hidden))
        self._cpu_inference_prepared = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in (self.gate_weight, self.up_weight, self.down_weight):
            nn.init.normal_(weight, mean=0.0, std=0.02)

    def initialize_residual(self, std: float) -> None:
        nn.init.normal_(self.down_weight, mean=0.0, std=std)

    def _normalize(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), None, self.rms_norm_eps)

    @torch.no_grad()
    def prepare_cpu_inference_(self) -> None:
        for weight in (self.gate_weight, self.up_weight, self.down_weight):
            weight.copy_(_materialize_ternary_weight(weight, (1, 2)))
            weight.requires_grad_(False)
        self._cpu_inference_prepared = True

    def grouped_forward(self, x: Tensor, group_experts: Tensor) -> Tensor:
        normalized = self._normalize(x)
        gate_input = normalized * self.gate_norm.index_select(
            0, group_experts
        ).unsqueeze(1).to(dtype=normalized.dtype)
        up_input = normalized * self.up_norm.index_select(
            0, group_experts
        ).unsqueeze(1).to(dtype=normalized.dtype)
        gate = grouped_bitlinear(gate_input, group_experts, self.gate_weight)
        up = grouped_bitlinear(up_input, group_experts, self.up_weight)
        down_normalized = self._normalize(F.silu(gate) * up)
        down_input = down_normalized * self.down_norm.index_select(
            0, group_experts
        ).unsqueeze(1).to(dtype=down_normalized.dtype)
        return grouped_bitlinear(down_input, group_experts, self.down_weight)

    def forward_one(self, x: Tensor, expert_index: int) -> Tensor:
        normalized = self._normalize(x)
        linear = F.linear if self._cpu_inference_prepared else ternary_ste_linear
        gate = linear(
            normalized * self.gate_norm[expert_index].to(dtype=normalized.dtype),
            self.gate_weight[expert_index],
        )
        up = linear(
            normalized * self.up_norm[expert_index].to(dtype=normalized.dtype),
            self.up_weight[expert_index],
        )
        down_normalized = self._normalize(F.silu(gate) * up)
        return linear(
            down_normalized * self.down_norm[expert_index].to(
                dtype=down_normalized.dtype
            ),
            self.down_weight[expert_index],
        )


class Top1MoE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.router_top_k
        self.router = BitLinear(
            config.d_model,
            config.n_experts,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.experts = PackedSwiGLU(config)

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        router_logits = self.router(x).float()
        router_probabilities = F.softmax(router_logits, dim=-1)
        if self.top_k == 1:
            selected_probabilities, selected_experts = router_probabilities.max(dim=-1)
            selected_probabilities = selected_probabilities.unsqueeze(-1)
            selected_experts = selected_experts.unsqueeze(-1)
        else:
            selected_probabilities, selected_experts = router_probabilities.topk(
                self.top_k, dim=-1
            )
            selected_probabilities = selected_probabilities / selected_probabilities.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
        return (
            router_logits,
            router_probabilities,
            selected_probabilities,
            selected_experts,
        )

    def _expert_counts(
        self,
        selected_experts: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        flat_experts = selected_experts.flatten()
        if valid_mask is None:
            return torch.zeros(
                self.n_experts,
                device=flat_experts.device,
                dtype=torch.long,
            ).scatter_add(0, flat_experts, torch.ones_like(flat_experts))
        weights = torch.stack(
            (
                torch.ones_like(flat_experts),
                valid_mask.flatten().to(dtype=torch.long),
            )
        )
        indices = flat_experts.unsqueeze(0).expand_as(weights)
        return torch.zeros(
            (2, self.n_experts),
            device=flat_experts.device,
            dtype=torch.long,
        ).scatter_add(1, indices, weights)

    def _grouped_expert_forward(
        self,
        expert_input: Tensor,
        group_experts: Tensor,
    ) -> Tensor:
        """Evaluate packed expert groups through three batched W1.58A8 projections."""
        return self.experts.grouped_forward(expert_input, group_experts)

    def _dispatch_cuda_grouped(
        self,
        x: Tensor,
        selected_probabilities: Tensor,
        selected_experts: Tensor,
        padded_rows: int,
        expert_counts: Tensor,
    ) -> Tensor:
        """Pack routed tokens without synchronizing route metadata to the CPU."""
        flat_x = x.flatten(0, 1)
        flat_probabilities = selected_probabilities.flatten()
        flat_experts = selected_experts.flatten()

        group_counts = torch.div(
            expert_counts + padded_rows - 1,
            padded_rows,
            rounding_mode="floor",
        )
        expert_starts = torch.cumsum(expert_counts, dim=0) - expert_counts
        group_starts = torch.cumsum(group_counts, dim=0) - group_counts
        max_groups = (
            (flat_experts.numel() + padded_rows - 1) // padded_rows
            + self.n_experts
            - 1
        )
        group_indices_all = torch.arange(max_groups, device=x.device)
        group_experts = torch.searchsorted(
            group_starts,
            group_indices_all,
            right=True,
        ).sub(1).clamp_(0, self.n_experts - 1)

        sorted_experts, sorted_token_indices = torch.sort(flat_experts)
        sorted_positions = torch.arange(flat_experts.numel(), device=x.device)
        relative_positions = sorted_positions - expert_starts.index_select(
            0, sorted_experts
        )
        group_indices = group_starts.index_select(
            0, sorted_experts
        ) + torch.div(relative_positions, padded_rows, rounding_mode="floor")
        row_indices = torch.remainder(relative_positions, padded_rows)

        packed_input = torch.zeros(
            (max_groups, padded_rows, x.size(-1)),
            device=x.device,
            dtype=x.dtype,
        )
        packed_input[group_indices, row_indices] = flat_x.index_select(
            0, sorted_token_indices
        )
        packed_output = self._grouped_expert_forward(
            packed_input,
            group_experts,
        )
        sorted_output = packed_output[group_indices, row_indices]
        sorted_scales = flat_probabilities.index_select(0, sorted_token_indices)
        output = torch.zeros_like(flat_x)
        output.index_copy_(
            0,
            sorted_token_indices,
            (sorted_output * sorted_scales.unsqueeze(-1)).to(dtype=output.dtype),
        )
        return output.view_as(x)

    def _dispatch(
        self,
        x: Tensor,
        selected_probabilities: Tensor,
        selected_experts: Tensor,
        padded_rows: int | None = None,
        expert_counts: Tensor | None = None,
    ) -> Tensor:
        if padded_rows is not None and x.device.type == "cuda":
            if expert_counts is None:
                expert_counts = self._expert_counts(selected_experts)
            return self._dispatch_cuda_grouped(
                x,
                selected_probabilities,
                selected_experts,
                padded_rows,
                expert_counts,
            )
        flat_x = x.flatten(0, 1)
        flat_probabilities = selected_probabilities.flatten()
        flat_experts = selected_experts.flatten()
        output = torch.zeros_like(flat_x)
        for expert_index in range(self.n_experts):
            token_indices = torch.nonzero(
                flat_experts == expert_index,
                as_tuple=False,
            ).flatten()
            if token_indices.numel() == 0:
                continue
            chunks = (
                token_indices.split(padded_rows)
                if padded_rows is not None
                else (token_indices,)
            )
            for chunk_indices in chunks:
                row_count = chunk_indices.numel()
                expert_input = flat_x.index_select(0, chunk_indices)
                if padded_rows is not None and row_count < padded_rows:
                    expert_input = F.pad(
                        expert_input,
                        (0, 0, 0, padded_rows - row_count),
                    )
                expert_output = self.experts.forward_one(
                    expert_input, expert_index
                )[:row_count]
                expert_scale = flat_probabilities.index_select(0, chunk_indices)
                expert_output = expert_output * expert_scale.to(
                    dtype=expert_output.dtype
                ).unsqueeze(-1)
                output.index_copy_(
                    0,
                    chunk_indices,
                    expert_output.to(dtype=output.dtype),
                )
        return output.view_as(x)

    def forward(
        self,
        x: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if valid_mask is not None and valid_mask.shape != x.shape[:2]:
            raise ValueError("MoE valid mask must match token shape")
        (
            router_logits,
            router_probabilities,
            selected_probabilities,
            selected_experts,
        ) = self._route(x)
        padded_rows = _CUDA_MOE_DISPATCH_ROWS if x.device.type == "cuda" else None
        output = torch.zeros_like(x)
        expert_counts = torch.zeros(
            self.n_experts, device=x.device, dtype=torch.long
        )
        for slot in range(self.top_k):
            slot_experts = selected_experts[..., slot]
            slot_probabilities = selected_probabilities[..., slot]
            counts = self._expert_counts(slot_experts, valid_mask)
            if valid_mask is None:
                slot_dispatch = slot_expert = counts
            else:
                slot_dispatch, slot_expert = counts.unbind(0)
            output = output + self._dispatch(
                x,
                slot_probabilities,
                slot_experts,
                padded_rows=padded_rows,
                expert_counts=slot_dispatch,
            )
            expert_counts = expert_counts + slot_expert

        flat_probabilities = router_probabilities.flatten(0, 1)
        token_valid = (
            torch.ones(x.shape[:2], device=x.device, dtype=flat_probabilities.dtype)
            if valid_mask is None
            else valid_mask.to(dtype=flat_probabilities.dtype)
        )
        flat_valid = token_valid.flatten()
        valid_token_count = flat_valid.sum().clamp_min(1.0)
        token_fraction = expert_counts.to(dtype=flat_probabilities.dtype)
        token_fraction = token_fraction / (valid_token_count * self.top_k)
        mean_router_probability = (
            flat_probabilities * flat_valid.unsqueeze(-1)
        ).sum(dim=0) / valid_token_count
        balance_loss = self.n_experts * torch.sum(
            token_fraction * mean_router_probability
        )
        z_values = torch.logsumexp(router_logits, dim=-1).square().flatten()
        z_loss = (z_values * flat_valid).sum() / valid_token_count
        entropy_values = -torch.xlogy(
            router_probabilities,
            router_probabilities,
        ).sum(dim=-1).flatten()
        router_entropy = (entropy_values * flat_valid).sum() / valid_token_count
        return output, balance_loss, z_loss, expert_counts, router_entropy

    def forward_inference(self, x: Tensor) -> Tensor:
        _, _, selected_probabilities, selected_experts = self._route(x)
        if x.device.type == "cpu" and x.shape[:2] == (1, 1):
            output = torch.zeros_like(x)
            for slot in range(self.top_k):
                expert_index = int(selected_experts[0, 0, slot].item())
                expert_output = self.experts.forward_one(
                    x.flatten(0, 1),
                    expert_index,
                ).view_as(x)
                scale = selected_probabilities[..., slot].to(dtype=expert_output.dtype)
                output = output + expert_output * scale.unsqueeze(-1)
            return output
        output = torch.zeros_like(x)
        for slot in range(self.top_k):
            output = output + self._dispatch(
                x,
                selected_probabilities[..., slot],
                selected_experts[..., slot],
            )
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.moe = Top1MoE(config)
        self.ple_gate = BitLinear(
            config.d_model,
            config.ple_dim,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.ple_projection = BitLinear(
            config.ple_dim,
            config.d_model,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.ple_norm = MixedDtypeRMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        x: Tensor,
        ple: Tensor,
        cos: Tensor,
        sin: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = x + self.attention(x, cos, sin)
        moe, balance_loss, z_loss, expert_counts, router_entropy = self.moe(
            x, valid_mask
        )
        x = x + moe
        gated_ple = F.gelu(self.ple_gate(x)) * ple
        x = x + self.ple_norm(self.ple_projection(gated_ple))
        return x, balance_loss, z_loss, expert_counts, router_entropy

    def forward_cached(
        self,
        x: Tensor,
        ple: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: LayerKVCache | None = None,
    ) -> tuple[Tensor, LayerKVCache]:
        attention, new_cache = self.attention.forward_cached(
            x,
            cos,
            sin,
            cache,
        )
        x = x + attention
        x = x + self.moe.forward_inference(x)
        gated_ple = F.gelu(self.ple_gate(x)) * ple
        x = x + self.ple_norm(self.ple_projection(gated_ple))
        return x, new_cache


class LLMM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.ple_model_projection = BitLinear(
            config.d_model,
            config.n_layers * config.ple_dim,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.ple_projection_norm = MixedDtypeRMSNorm(
            config.ple_dim, eps=config.rms_norm_eps
        )
        self.ple_table = TernaryEmbedding(
            config.vocab_size, config.n_layers * config.ple_dim
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.output_norm = MixedDtypeRMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        rope_cos, rope_sin = _build_rope(
            config.max_seq_len,
            config.head_dim,
            config.rope_theta,
            torch.device("cpu"),
        )
        self.register_buffer("_rope_cos_fp32", rope_cos, persistent=False)
        self.register_buffer("_rope_sin_fp32", rope_sin, persistent=False)
        self.register_buffer(
            "_rope_cos_bf16",
            rope_cos.to(dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "_rope_sin_bf16",
            rope_sin.to(dtype=torch.bfloat16),
            persistent=False,
        )

        self.apply(self._initialize)
        self.ple_table.reset_absmean_cache()
        self.lm_head.weight = self.token_embedding.weight
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, mean=0.0, std=residual_std)
            block.moe.experts.initialize_residual(residual_std)
            nn.init.normal_(block.ple_projection.weight, mean=0.0, std=residual_std)
            nn.init.zeros_(block.ple_norm.weight)
        enable_bitnet_training(self)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _per_layer_embeddings(self, x: Tensor, table: Tensor) -> Tensor:
        batch_size, seq_len = x.shape[:2]
        projected = self.ple_model_projection(x) * (self.config.d_model**-0.5)
        projected = projected.view(
            batch_size, seq_len, self.config.n_layers, self.config.ple_dim
        )
        projected = self.ple_projection_norm(projected)
        table = table.view(
            batch_size, seq_len, self.config.n_layers, self.config.ple_dim
        )
        table = table * math.sqrt(self.config.ple_dim)
        return (projected + table) * (2.0**-0.5)

    @torch.no_grad()
    def prepare_cpu_inference_(self) -> None:
        if next(self.parameters()).device.type != "cpu":
            raise ValueError("CPU inference preparation requires a CPU model")
        for module in self.modules():
            if isinstance(module, BitLinear):
                materialized = _materialize_ternary_weight(module.weight, (0, 1))
                module.weight = nn.Parameter(materialized, requires_grad=False)
        for block in self.blocks:
            block.moe.experts.prepare_cpu_inference_()

    def _rope(
        self,
        seq_len: int,
        position_offset: int,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if dtype == torch.bfloat16:
            cos = self._rope_cos_bf16
            sin = self._rope_sin_bf16
        else:
            cos = self._rope_cos_fp32
            sin = self._rope_sin_fp32
            if dtype != torch.float32:
                cos = cos.to(dtype=dtype)
                sin = sin.to(dtype=dtype)
        end = position_offset + seq_len
        return (
            cos[:, :, position_offset:end],
            sin[:, :, position_offset:end],
        )

    @staticmethod
    def _execution_dtype(x: Tensor) -> torch.dtype:
        if torch.is_autocast_enabled(x.device.type):
            return torch.get_autocast_dtype(x.device.type)
        return x.dtype

    def _hidden_states(
        self,
        token_ids: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self._hidden_states_from_ple(
            token_ids,
            self.ple_table(token_ids),
            valid_mask,
        )

    def _hidden_states_from_ple(
        self,
        token_ids: Tensor,
        ple_table: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.size(1) > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        if valid_mask is not None and valid_mask.shape != token_ids.shape:
            raise ValueError("valid mask must match token IDs")

        x = self.token_embedding(token_ids)
        ple = self._per_layer_embeddings(x, ple_table)
        cos, sin = self._rope(
            token_ids.size(1),
            0,
            self._execution_dtype(x),
        )
        balance_losses: list[Tensor] = []
        z_losses: list[Tensor] = []
        expert_counts: list[Tensor] = []
        router_entropies: list[Tensor] = []
        for layer_index, block in enumerate(self.blocks):
            x, balance_loss, z_loss, counts, entropy = block(
                x,
                ple[:, :, layer_index],
                cos,
                sin,
                valid_mask,
            )
            balance_losses.append(balance_loss)
            z_losses.append(z_loss)
            expert_counts.append(counts)
            router_entropies.append(entropy)
        return (
            self.output_norm(x),
            torch.stack(balance_losses).mean(),
            torch.stack(z_losses).mean(),
            torch.stack(expert_counts),
            torch.stack(router_entropies),
        )

    @staticmethod
    def _shifted_labels(labels: Tensor) -> Tensor:
        return F.pad(labels[:, 1:], (0, 1), value=-100)

    def training_loss(self, token_ids: Tensor, labels: Tensor) -> Tensor:
        if labels.shape != token_ids.shape:
            raise ValueError("labels must match token_ids shape")
        return self.training_loss_components(token_ids, labels)[0]

    def _causal_loss_from_hidden_states(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        *,
        chunk_size: int = DEFAULT_LOSS_CHUNK_SIZE,
        non_ignore_count: int | None = None,
    ) -> Tensor:
        if hidden_states.shape[:2] != labels.shape:
            raise ValueError("hidden states and labels must have matching batches")
        shifted_labels = self._shifted_labels(labels)
        flattened_hidden = hidden_states.reshape(-1, self.config.d_model)
        flattened_labels = shifted_labels.reshape(-1)
        if hidden_states.device.type == "cuda":
            if non_ignore_count is None:
                non_ignore_count = int(
                    flattened_labels.ne(-100).sum().item()
                )
            return fixed_chunk_fused_linear_cross_entropy(
                flattened_hidden,
                self.lm_head.weight,
                flattened_labels,
                non_ignore_count=non_ignore_count,
                chunk_size=chunk_size,
                ignore_index=-100,
            )
        logits = F.linear(flattened_hidden, self.lm_head.weight)
        return F.cross_entropy(logits, flattened_labels, ignore_index=-100)

    def _training_losses_from_hidden_states(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        balance_loss: Tensor,
        z_loss: Tensor,
        *,
        loss_chunk_size: int = DEFAULT_LOSS_CHUNK_SIZE,
        non_ignore_count: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        causal_loss = self._causal_loss_from_hidden_states(
            hidden_states,
            labels,
            chunk_size=loss_chunk_size,
            non_ignore_count=non_ignore_count,
        )
        total_loss = (
            causal_loss
            + self.config.router_balance_loss_coefficient * balance_loss
            + self.config.router_z_loss_coefficient * z_loss
        )
        return total_loss, causal_loss, balance_loss, z_loss

    def training_loss_components(
        self,
        token_ids: Tensor,
        labels: Tensor,
        *,
        valid_mask: Tensor | None = None,
        non_ignore_count: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if labels.shape != token_ids.shape:
            raise ValueError("labels must match token_ids shape")
        hidden_states, balance_loss, z_loss, counts, entropy = self._hidden_states(
            token_ids,
            valid_mask,
        )
        total_loss, causal_loss, balance_loss, z_loss = (
            self._training_losses_from_hidden_states(
                hidden_states,
                labels,
                balance_loss,
                z_loss,
                non_ignore_count=non_ignore_count,
            )
        )
        return total_loss, causal_loss, balance_loss, z_loss, counts, entropy

    def forward(
        self,
        token_ids: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        hidden_states, balance_loss, z_loss, _, _ = self._hidden_states(token_ids)

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            if labels.shape != token_ids.shape:
                raise ValueError("labels must match token_ids shape")
            shifted_labels = self._shifted_labels(labels)
            causal_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                shifted_labels.reshape(-1),
                ignore_index=-100,
            )
            loss = (
                causal_loss
                + self.config.router_balance_loss_coefficient * balance_loss
                + self.config.router_z_loss_coefficient * z_loss
            )
        return logits, loss

    def forward_cached(
        self,
        token_ids: Tensor,
        cache: ModelKVCache | None = None,
    ) -> tuple[Tensor, ModelKVCache]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")

        cached_length = 0
        if cache is not None:
            if len(cache) != self.config.n_layers:
                raise ValueError("cache must contain one entry per layer")
            if token_ids.size(1) != 1:
                raise ValueError("cached decoding accepts one new token at a time")
            cached_length = cache[0].length
        if cached_length + token_ids.size(1) > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        x = self.token_embedding(token_ids)
        ple = self._per_layer_embeddings(x, self.ple_table(token_ids))
        cos, sin = self._rope(
            token_ids.size(1),
            cached_length,
            self._execution_dtype(x),
        )
        new_cache: list[LayerKVCache] = []
        for layer_index, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[layer_index]
            x, updated_cache = block.forward_cached(
                x,
                ple[:, :, layer_index],
                cos,
                sin,
                layer_cache,
            )
            new_cache.append(updated_cache)

        logits = self.lm_head(self.output_norm(x))
        return logits, tuple(new_cache)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
