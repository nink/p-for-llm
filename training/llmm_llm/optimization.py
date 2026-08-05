"""Optimizer schedule primitives for LLMM pretraining."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only development environments.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fused_sparse_adam_kernel(
        parameter_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        gradient_ptr,
        row_indices_ptr,
        row_width,
        parameter_row_stride,
        parameter_column_stride,
        gradient_row_stride,
        gradient_column_stride,
        beta1: tl.constexpr,
        beta2: tl.constexpr,
        eps: tl.constexpr,
        step_size,
        maximize: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        gradient_row = tl.program_id(0)
        column_block = tl.program_id(1)
        columns = column_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = columns < row_width
        parameter_row = tl.load(row_indices_ptr + gradient_row)
        parameter_offsets = (
            parameter_row * parameter_row_stride
            + columns * parameter_column_stride
        )
        gradient_offsets = (
            gradient_row * gradient_row_stride
            + columns * gradient_column_stride
        )

        gradient = tl.load(gradient_ptr + gradient_offsets, mask=mask).to(tl.float32)
        if maximize:
            gradient = -gradient
        exp_avg = tl.load(exp_avg_ptr + parameter_offsets, mask=mask).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + parameter_offsets, mask=mask).to(tl.float32)
        parameter = tl.load(parameter_ptr + parameter_offsets, mask=mask).to(tl.float32)

        exp_avg = beta1 * exp_avg + (1.0 - beta1) * gradient
        exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * gradient * gradient
        parameter -= step_size * exp_avg / (tl.sqrt(exp_avg_sq) + eps)

        tl.store(parameter_ptr + parameter_offsets, parameter, mask=mask)
        tl.store(exp_avg_ptr + parameter_offsets, exp_avg, mask=mask)
        tl.store(exp_avg_sq_ptr + parameter_offsets, exp_avg_sq, mask=mask)


    @triton.jit
    def _fused_masked_adamw_kernel(
        parameter_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        gradient_ptr,
        active_ptr,
        step_ptr,
        elements_per_expert,
        learning_rate_ptr,
        grad_scale_ptr,
        beta1: tl.constexpr,
        beta2: tl.constexpr,
        weight_decay: tl.constexpr,
        eps: tl.constexpr,
        maximize: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        element_block = tl.program_id(0)
        expert = tl.program_id(1)
        offsets = element_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_expert
        active = tl.load(active_ptr + expert).to(tl.int1)
        parameter_offsets = expert * elements_per_expert + offsets
        parameter = tl.load(parameter_ptr + parameter_offsets, mask=mask).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + parameter_offsets, mask=mask).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + parameter_offsets, mask=mask).to(tl.float32)
        gradient = tl.load(gradient_ptr + parameter_offsets, mask=mask).to(tl.float32)
        learning_rate = tl.load(learning_rate_ptr).to(tl.float32)
        grad_scale = tl.load(grad_scale_ptr).to(tl.float32)
        gradient = gradient / grad_scale
        if maximize:
            gradient = -gradient
        step = tl.load(step_ptr + expert).to(tl.float32)
        beta1_power = tl.exp(tl.log(beta1) * step)
        beta2_power = tl.exp(tl.log(beta2) * step)
        step_size = learning_rate * tl.sqrt(1.0 - beta2_power) / (1.0 - beta1_power)
        updated_exp_avg = beta1 * exp_avg + (1.0 - beta1) * gradient
        updated_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * gradient * gradient
        updated_parameter = parameter * (1.0 - learning_rate * weight_decay)
        updated_parameter -= step_size * updated_exp_avg / (tl.sqrt(updated_exp_avg_sq) + eps)
        tl.store(
            parameter_ptr + parameter_offsets,
            tl.where(active, updated_parameter, parameter),
            mask=mask,
        )
        tl.store(
            exp_avg_ptr + parameter_offsets,
            tl.where(active, updated_exp_avg, exp_avg),
            mask=mask,
        )
        tl.store(
            exp_avg_sq_ptr + parameter_offsets,
            tl.where(active, updated_exp_avg_sq, exp_avg_sq),
            mask=mask,
        )


@dataclass(slots=True)
class _FusedAdamWGroup:
    parameters: list[nn.Parameter]
    expert_coordinates: list[tuple[int, int] | None]
    exp_avgs: list[Tensor]
    exp_avg_sqs: list[Tensor]
    state_steps: list[Tensor]
    step_values: Tensor
    learning_rate: Tensor
    beta1: float
    beta2: float
    weight_decay: float
    eps: float
    maximize: bool


@dataclass(slots=True)
class _ExpertBankAdamWState:
    parameter: nn.Parameter
    exp_avg: Tensor
    exp_avg_sq: Tensor
    steps: Tensor
    layer_index: int
    learning_rate: Tensor
    beta1: float
    beta2: float
    weight_decay: float
    eps: float
    maximize: bool


class CachedFusedAdamW:
    """Execute a CUDA AdamW optimizer without rebuilding tensor groups each step."""

    def __init__(
        self,
        optimizer: torch.optim.AdamW,
        expert_coordinates: dict[nn.Parameter, int] | None = None,
    ) -> None:
        self.optimizer = optimizer
        coordinates = expert_coordinates or {}
        self.expert_banks: list[_ExpertBankAdamWState] = []
        self.groups = [
            self._prepare_group(group, coordinates)
            for group in optimizer.param_groups
        ]

    def _prepare_group(
        self,
        group: dict,
        expert_coordinates: dict[nn.Parameter, int],
    ) -> _FusedAdamWGroup:
        parameters = list(group["params"])
        if not parameters or any(parameter.device.type != "cuda" for parameter in parameters):
            raise ValueError("cached fused AdamW requires non-empty CUDA parameter groups")
        if len({(parameter.device, parameter.dtype) for parameter in parameters}) != 1:
            raise ValueError("cached fused AdamW requires one device and dtype per group")
        if group["amsgrad"] or group["differentiable"]:
            raise ValueError("cached fused AdamW does not support AMSGrad or differentiable mode")

        device = parameters[0].device
        learning_rate = group["lr"]
        if isinstance(learning_rate, Tensor):
            learning_rate = learning_rate.detach().to(device=device)
        else:
            learning_rate = torch.tensor(float(learning_rate), device=device)
        group["lr"] = learning_rate
        group["capturable"] = True
        group["fused"] = True

        regular_parameters: list[nn.Parameter] = []
        exp_avgs: list[Tensor] = []
        exp_avg_sqs: list[Tensor] = []
        state_steps: list[Tensor] = []
        for parameter in parameters:
            state = self.optimizer.state[parameter]
            expert_layer = expert_coordinates.get(parameter)
            if not state:
                state["step"] = torch.zeros(
                    parameter.size(0) if expert_layer is not None else (),
                    dtype=torch.float32,
                    device=device,
                )
                state["exp_avg"] = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
            elif not isinstance(state["step"], Tensor):
                state["step"] = torch.tensor(
                    float(state["step"]),
                    dtype=torch.float32,
                    device=device,
                )
            elif state["step"].device != device:
                state["step"] = state["step"].to(device=device)
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
            if expert_layer is not None:
                if state["step"].shape != (parameter.size(0),):
                    raise ValueError("packed expert bank has incompatible AdamW step state")
                self.expert_banks.append(
                    _ExpertBankAdamWState(
                        parameter=parameter,
                        exp_avg=state["exp_avg"],
                        exp_avg_sq=state["exp_avg_sq"],
                        steps=state["step"],
                        layer_index=expert_layer,
                        learning_rate=learning_rate,
                        beta1=group["betas"][0],
                        beta2=group["betas"][1],
                        weight_decay=group["weight_decay"],
                        eps=group["eps"],
                        maximize=group["maximize"],
                    )
                )
                continue
            regular_parameters.append(parameter)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])

        beta1, beta2 = group["betas"]
        if isinstance(beta1, Tensor) or isinstance(beta2, Tensor):
            raise ValueError("cached fused AdamW expects scalar beta values")
        step_values = torch.stack(state_steps).contiguous()
        state_steps = [step_values[index] for index in range(len(regular_parameters))]
        for parameter, state_step in zip(
            regular_parameters, state_steps, strict=True
        ):
            self.optimizer.state[parameter]["step"] = state_step
        return _FusedAdamWGroup(
            parameters=regular_parameters,
            expert_coordinates=[None] * len(regular_parameters),
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            state_steps=state_steps,
            step_values=step_values,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
        )

    @torch.no_grad()
    def zero_grad(self) -> None:
        gradients = self.gradients()
        if gradients:
            torch._foreach_zero_(gradients)

    def gradients(self) -> list[Tensor]:
        regular_gradients = [
            parameter.grad
            for group in self.groups
            for parameter in group.parameters
            if parameter.grad is not None
        ]
        return regular_gradients + [
            bank.parameter.grad
            for bank in self.expert_banks
            if bank.parameter.grad is not None
        ]

    @torch.no_grad()
    def step(
        self,
        *,
        grad_scale: Tensor,
        active_experts: Tensor | None = None,
    ) -> None:
        for group in self.groups:
            if not group.parameters:
                continue
            parameters = group.parameters
            gradients = [parameter.grad for parameter in parameters]
            exp_avgs = group.exp_avgs
            exp_avg_sqs = group.exp_avg_sqs
            state_steps = group.state_steps
            group.step_values.add_(1)
            torch._fused_adamw_(
                parameters,
                gradients,
                exp_avgs,
                exp_avg_sqs,
                [],
                state_steps,
                amsgrad=False,
                lr=group.learning_rate,
                beta1=group.beta1,
                beta2=group.beta2,
                weight_decay=group.weight_decay,
                eps=group.eps,
                maximize=group.maximize,
                grad_scale=grad_scale,
                found_inf=None,
            )
        if self.expert_banks:
            if triton is None:
                raise RuntimeError("packed expert banks require Triton")
            active_device = (
                None
                if active_experts is None
                else active_experts.to(
                    device=self.expert_banks[0].parameter.device,
                    non_blocking=True,
                )
            )
            for bank in self.expert_banks:
                active = (
                    torch.ones_like(bank.steps, dtype=torch.bool)
                    if active_device is None
                    else active_device[bank.layer_index]
                )
                bank.steps.add_(active.to(dtype=bank.steps.dtype))
                elements_per_expert = bank.parameter.numel() // bank.parameter.size(0)
                _fused_masked_adamw_kernel[
                    (triton.cdiv(elements_per_expert, 256), bank.parameter.size(0))
                ](
                    bank.parameter,
                    bank.exp_avg,
                    bank.exp_avg_sq,
                    bank.parameter.grad,
                    active,
                    bank.steps,
                    elements_per_expert,
                    bank.learning_rate,
                    grad_scale,
                    beta1=bank.beta1,
                    beta2=bank.beta2,
                    weight_decay=bank.weight_decay,
                    eps=bank.eps,
                    maximize=bank.maximize,
                    BLOCK_SIZE=256,
                    num_warps=4,
                )


class FusedSparseAdam:
    """Update the coalesced PLE rows in one CUDA kernel."""

    def __init__(self, optimizer: torch.optim.SparseAdam, parameter: nn.Parameter) -> None:
        if triton is None or parameter.device.type != "cuda":
            raise ValueError("fused SparseAdam requires Triton and a CUDA parameter")
        optimizer_parameters = (
            optimizer.param_groups[0]["params"]
            if len(optimizer.param_groups) == 1
            else ()
        )
        if len(optimizer_parameters) != 1 or optimizer_parameters[0] is not parameter:
            raise ValueError("fused SparseAdam expects exactly one parameter")
        self.optimizer = optimizer
        self.parameter = parameter
        self.group = optimizer.param_groups[0]
        state = optimizer.state[parameter]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(
                parameter,
                memory_format=torch.preserve_format,
            )
            state["exp_avg_sq"] = torch.zeros_like(
                parameter,
                memory_format=torch.preserve_format,
            )
        self.state = state

    def zero_grad(self) -> None:
        self.parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        gradient = self.parameter.grad
        if gradient is None:
            return
        if not gradient.is_sparse or not gradient.is_coalesced():
            raise RuntimeError("fused SparseAdam requires a coalesced sparse gradient")
        indices = gradient.indices()[0]
        values = gradient.values()
        if indices.numel() == 0:
            return
        if values.ndim != 2 or self.parameter.ndim != 2:
            raise RuntimeError("fused SparseAdam expects row-sparse rank-2 tensors")

        self.state["step"] += 1
        beta1, beta2 = self.group["betas"]
        learning_rate = self.group["lr"]
        if isinstance(learning_rate, Tensor):
            raise ValueError("fused SparseAdam expects a scalar learning rate")
        step = self.state["step"]
        step_size = (
            learning_rate
            * math.sqrt(1.0 - beta2**step)
            / (1.0 - beta1**step)
        )
        block_size = 256
        grid = (indices.numel(), triton.cdiv(values.shape[1], block_size))
        _fused_sparse_adam_kernel[grid](
            self.parameter,
            self.state["exp_avg"],
            self.state["exp_avg_sq"],
            values,
            indices,
            values.shape[1],
            *self.parameter.stride(),
            *values.stride(),
            beta1=beta1,
            beta2=beta2,
            eps=self.group["eps"],
            step_size=step_size,
            maximize=self.group.get("maximize", False),
            BLOCK_SIZE=block_size,
            num_warps=8,
        )


def learning_rate_for_step(
    step: int,
    total_steps: int,
    peak_learning_rate: float,
    warmup_ratio: float,
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step <= 0 or step > total_steps:
        raise ValueError("step must be in [1, total_steps]")
    if peak_learning_rate <= 0:
        raise ValueError("peak_learning_rate must be positive")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0, 1]")

    warmup_steps = (
        min(total_steps, max(1, math.ceil(total_steps * warmup_ratio)))
        if warmup_ratio > 0.0
        else 0
    )
    if step <= warmup_steps:
        return peak_learning_rate * step / warmup_steps
    if total_steps == warmup_steps:
        return peak_learning_rate

    decay_progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return peak_learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_progress))


def set_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    for parameter_group in optimizer.param_groups:
        current = parameter_group["lr"]
        if isinstance(current, Tensor):
            current.fill_(learning_rate)
        else:
            parameter_group["lr"] = learning_rate


@torch.no_grad()
def fused_adamw_clip_scale_(
    dense_gradients: list[Tensor],
    sparse_parameter: nn.Parameter,
    max_norm: float,
) -> tuple[Tensor, Tensor]:
    """Compute one global norm and return the fused-Adam unscale factor."""
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    sparse_gradient = sparse_parameter.grad
    if sparse_gradient is None or not sparse_gradient.is_sparse:
        raise RuntimeError("expected one sparse PLE gradient")
    sparse_gradient = sparse_gradient.coalesce()
    sparse_parameter.grad = sparse_gradient

    dense_norms = torch._foreach_norm(dense_gradients, 2.0)
    sparse_norm = torch.linalg.vector_norm(sparse_gradient.values(), ord=2.0)
    total_norm = torch.linalg.vector_norm(
        torch.stack((*dense_norms, sparse_norm)),
        ord=2.0,
    )
    clip_coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    sparse_gradient.values().mul_(clip_coefficient)
    return total_norm, clip_coefficient.reciprocal()


@torch.no_grad()
def clip_grad_norm_sparse_(
    parameters: Iterable[nn.Parameter],
    max_norm: float,
) -> Tensor:
    """Clip one global L2 norm without densifying sparse gradients."""
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")

    dense_parameters: list[nn.Parameter] = []
    sparse_gradients: list[Tensor] = []
    norm_tensors: list[Tensor] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            gradient = gradient.coalesce()
            parameter.grad = gradient
            sparse_gradients.append(gradient)
            norm_tensors.append(gradient.values())
        else:
            dense_parameters.append(parameter)
            norm_tensors.append(gradient)

    total_norm = torch.nn.utils.get_total_norm(norm_tensors, norm_type=2.0)
    if dense_parameters:
        torch.nn.utils.clip_grads_with_norm_(
            dense_parameters,
            max_norm,
            total_norm,
        )
    clip_coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for gradient in sparse_gradients:
        gradient.values().mul_(clip_coefficient.to(gradient.device))
    return total_norm
