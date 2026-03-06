import math

from typing import Callable
from typing import List
from typing import Optional

import torch

from liger_kernel.ops.utils import ensure_contiguous


class LigerTiledMLPFunction(torch.autograd.Function):
    """
    Based on DeepSpeed's TiledMLP:
    https://github.com/deepspeedai/DeepSpeed/blob/v0.18.2/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L838

    Perform a tiled MLP computation to massively reduce memory usage needed to compute MLP
    when using very long sequence lengths.

    This module re-computes `forward` in the `backward`. So the `forward` occurs twice each iteration.
    And if you're using activation checkpointing it then occurs thrice.

    Args:
        fn: the function to call on sharded inputs (e.g., mlp.forward)
        mlp_module: the MLP nn.Module object
        x: the input to MLP.forward (hidden_states)
        shards: how many shards to use
        compute_params: a list of weights engaged in the compute

    Returns:
        the computed hidden_states
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        fn: Callable,
        mlp_module: torch.nn.Module,
        x: torch.Tensor,
        shards: int,
        compute_params: Optional[List[torch.nn.Parameter]] = None,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.mlp_module = mlp_module
        ctx.shards = shards
        ctx.save_for_backward(x)

        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size] (moe experts)
        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        with torch.no_grad():
            output_shards = [fn(mlp_module, x_shard) for x_shard in x_shards]
        output_unsharded = torch.cat(output_shards, dim=-2)

        return output_unsharded

    @staticmethod
    @ensure_contiguous
    def backward(ctx, *grads) -> tuple:
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        mlp_module = ctx.mlp_module
        shards = ctx.shards

        x_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad_(x_requires_grad)

        hidden_size = x.shape[-1]
        x_shape_orig = x.shape

        x = x.view(-1, hidden_size)
        incoming_grad = grads[0].view(-1, hidden_size)
        x_grad = torch.zeros_like(x)

        # initialize param grad accumulators
        param_grads = {p: None for p in mlp_module.parameters()}

        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        for i, x_shard in enumerate(x_shards):
            x_shard = x_shard.detach().requires_grad_(x_requires_grad)

            shard_step = x_shard.shape[0]
            shard_offset = i * x_shards[0].shape[0]
            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)

            with torch.enable_grad():
                output = fn(mlp_module, x_shard)

                # compute grads locally — never touches .grad, never triggers FSDP hooks
                inputs = [x_shard] + list(mlp_module.parameters())
                local_grads = torch.autograd.grad(
                    outputs=output,
                    inputs=inputs,
                    grad_outputs=incoming_grad_shard,
                )

            # accumulate x grad
            x_shard_grad = local_grads[0]
            x_grad.narrow(0, shard_offset, shard_step).copy_(x_shard_grad)

            # accumulate param grads manually
            for p, g in zip(mlp_module.parameters(), local_grads[1:]):
                if param_grads[p] is None:
                    param_grads[p] = g
                else:
                    param_grads[p] += g

        # write accumulated grads into .grad ONCE — FSDP sees exactly one grad event
        for p, g in param_grads.items():
            if p.grad is None:
                p.grad = g
            else:
                p.grad += g

        x_grad = x_grad.view(x_shape_orig)

        return (None, None, x_grad, None, None)




def apply_tiled_mlp(
    fn: Callable,
    mlp_module: torch.nn.Module,
    x: torch.Tensor,
    num_shards: Optional[int] = None,
    compute_params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """
    Apply tiled MLP computation for memory efficiency.

    Args:
        fn: the function to call on sharded inputs (e.g., lambda module, x: module(x))
        mlp_module: the MLP nn.Module object
        x: the input tensor with shape [bs, seqlen, hidden_size] or [seqlen, hidden_size]
        num_shards: number of shards to use. If None, automatically calculated as ceil(seqlen / hidden_size)
        compute_params: list of parameters for DeepSpeed ZeRO optimization

    Returns:
        output tensor with the same shape as input
    """
    if num_shards is None:
        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size]
        hidden_size = x.shape[-1]
        seqlen = x.shape[-2]
        num_shards = math.ceil(seqlen / hidden_size)

    # Ensure num_shards is at least 1
    num_shards = max(1, num_shards)

    return LigerTiledMLPFunction.apply(
        fn,
        mlp_module,
        x,
        num_shards,
        compute_params,
    )
