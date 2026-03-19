"""
Benchmark: test_fsdp branch backward vs main branch backward for LigerTiledMLPFunction.

Both backward implementations are inlined here so we can benchmark without switching branches.
"""

import gc
import time

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# A minimal SwiGLU-style MLP (same as LigerTiledSwiGLUMLP internals)
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def mlp_fn(module, x):
    return module(x)


# ---------------------------------------------------------------------------
# Branch backward  (test_fsdp: torch.autograd.grad per shard, manual accum)
# ---------------------------------------------------------------------------

def backward_branch(x, mlp, shards, incoming_grad):
    """Returns x_grad. Writes into mlp param .grad."""
    hidden_size = x.shape[-1]
    x_shape_orig = x.shape
    x_req = x.requires_grad

    xf = x.detach().view(-1, hidden_size)
    xf.requires_grad_(x_req)
    gf = incoming_grad.view(-1, hidden_size)
    x_grad = torch.zeros_like(xf)

    param_grads = {p: None for p in mlp.parameters()}
    x_shards = list(torch.chunk(xf, chunks=shards, dim=0))

    for i, x_shard in enumerate(x_shards):
        x_shard = x_shard.detach().requires_grad_(x_req)
        shard_step = x_shard.shape[0]
        shard_offset = i * x_shards[0].shape[0]
        g_shard = gf.narrow(0, shard_offset, shard_step).view_as(x_shard)

        with torch.enable_grad():
            out = mlp_fn(mlp, x_shard)
            inputs = [x_shard] + list(mlp.parameters())
            local_grads = torch.autograd.grad(
                outputs=out,
                inputs=inputs,
                grad_outputs=g_shard,
            )

        x_grad.narrow(0, shard_offset, shard_step).copy_(local_grads[0])
        for p, g in zip(mlp.parameters(), local_grads[1:]):
            if param_grads[p] is None:
                param_grads[p] = g
            else:
                param_grads[p] += g

    for p, g in param_grads.items():
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g

    return x_grad.view(x_shape_orig)


# ---------------------------------------------------------------------------
# Main backward  (main branch: .grad pre-set views, one torch.autograd.backward)
# ---------------------------------------------------------------------------

def backward_main(x, mlp, shards, incoming_grad):
    """Returns x_grad. Writes into mlp param .grad."""
    hidden_size = x.shape[-1]
    x_shape_orig = x.shape
    x_req = x.requires_grad

    xf = x.detach().view(-1, hidden_size)
    xf.requires_grad_(x_req)
    gf = incoming_grad.view(-1, hidden_size)
    x_grad = torch.zeros_like(xf)

    x_shards = list(torch.chunk(xf, chunks=shards, dim=0))

    all_outputs = []
    all_incoming_grads = []

    for i, x_shard in enumerate(x_shards):
        x_shard.requires_grad_(x_req)
        shard_step = x_shards[i].shape[0]
        shard_offset = i * x_shards[0].shape[0]

        x_shard.grad = x_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)

        with torch.enable_grad():
            all_outputs.append(mlp_fn(mlp, x_shard))
            all_incoming_grads.append(
                gf.narrow(0, shard_offset, shard_step).view_as(x_shard)
            )

    torch.autograd.backward(all_outputs, all_incoming_grads)

    return x_grad.view(x_shape_orig)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def reset_grads(mlp):
    for p in mlp.parameters():
        p.grad = None


def measure(fn, warmup=2, repeats=5):
    """Return (mean_ms, peak_mem_MB) averaged over repeats."""
    device = next(iter(torch.cuda.current_device.__self__ if False else [None]))
    # warm-up
    for _ in range(warmup):
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()

    times_ms = []
    mems_mb  = []
    for _ in range(repeats):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)
        mems_mb.append(torch.cuda.max_memory_allocated() / 1024**2)

    return sum(times_ms) / repeats, sum(mems_mb) / repeats


def run_case(hidden_size, seqlen, batch_size, shards, intermediate_size, dtype):
    device = "cuda"
    mlp_branch = SimpleMLP(hidden_size, intermediate_size).to(device, dtype)
    mlp_main   = SimpleMLP(hidden_size, intermediate_size).to(device, dtype)

    # same weights
    with torch.no_grad():
        for pb, pm in zip(mlp_branch.parameters(), mlp_main.parameters()):
            pm.copy_(pb)

    x_base = torch.randn(batch_size, seqlen, hidden_size, device=device, dtype=dtype, requires_grad=True)
    incoming_grad = torch.randn(batch_size, seqlen, hidden_size, device=device, dtype=dtype)

    def run_branch():
        reset_grads(mlp_branch)
        x = x_base.detach().requires_grad_(True)
        backward_branch(x, mlp_branch, shards, incoming_grad)

    def run_main():
        reset_grads(mlp_main)
        x = x_base.detach().requires_grad_(True)
        backward_main(x, mlp_main, shards, incoming_grad)

    t_branch, m_branch = measure(run_branch)
    t_main,   m_main   = measure(run_main)

    return t_branch, m_branch, t_main, m_main


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

configs = [
    # (hidden_size, seqlen, batch_size, shards, intermediate_size)
    (1024,  2048,  1, 1,  4096),
    (1024,  2048,  1, 4,  4096),
    (1024,  2048,  1, 8,  4096),
    (2048,  4096,  1, 1,  8192),
    (2048,  4096,  1, 4,  8192),
    (2048,  4096,  1, 8,  8192),
    (2048,  8192,  1, 4,  8192),
    (2048,  8192,  1, 8,  8192),
    (2048,  8192,  1, 16, 8192),
    (4096,  8192,  1, 4, 14336),
    (4096,  8192,  1, 8, 14336),
    (4096,  8192,  1, 16,14336),
    (4096, 16384,  1, 8, 14336),
    (4096, 16384,  1, 16,14336),
    (4096, 16384,  1, 32,14336),
]

dtype = torch.bfloat16

header = (
    f"{'hidden':>8} {'seqlen':>7} {'bs':>3} {'shards':>7} {'interm':>7} "
    f"{'br_ms':>8} {'br_MB':>8} "
    f"{'main_ms':>8} {'main_MB':>8} "
    f"{'Δms':>8} {'ΔMB':>8}"
)
sep = "-" * len(header)

print(f"\nBackward benchmark: branch (test_fsdp) vs main   dtype={dtype}   device=cuda\n")
print(header)
print(sep)

for hidden_size, seqlen, batch_size, shards, intermediate_size in configs:
    try:
        t_br, m_br, t_main, m_main = run_case(
            hidden_size, seqlen, batch_size, shards, intermediate_size, dtype
        )
        delta_ms = t_br - t_main
        delta_mb = m_br - m_main
        print(
            f"{hidden_size:>8} {seqlen:>7} {batch_size:>3} {shards:>7} {intermediate_size:>7} "
            f"{t_br:>8.2f} {m_br:>8.1f} "
            f"{t_main:>8.2f} {m_main:>8.1f} "
            f"{delta_ms:>+8.2f} {delta_mb:>+8.1f}"
        )
    except torch.cuda.OutOfMemoryError:
        print(
            f"{hidden_size:>8} {seqlen:>7} {batch_size:>3} {shards:>7} {intermediate_size:>7} "
            f"{'OOM':>8} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8}"
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()

print(sep)
print("Δms = branch_ms - main_ms   (positive = branch slower)")
print("ΔMB = branch_MB - main_MB   (positive = branch uses more peak memory)")
