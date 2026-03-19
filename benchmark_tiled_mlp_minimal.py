"""
Minimal benchmark: test_fsdp branch backward vs main branch backward.
Run with: python benchmark_tiled_mlp_minimal.py
"""
import gc, time
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate = nn.Linear(h, i, bias=False)
        self.up   = nn.Linear(h, i, bias=False)
        self.down = nn.Linear(i, h, bias=False)
    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


def backward_branch(x, mlp, shards, grad_out):
    """test_fsdp: torch.autograd.grad per shard, manual param grad accumulation."""
    h = x.shape[-1]; orig = x.shape
    xf = x.detach().view(-1, h).requires_grad_(True)
    gf = grad_out.view(-1, h)
    x_grad = torch.zeros_like(xf)
    param_grads = {p: None for p in mlp.parameters()}
    chunks = list(torch.chunk(xf, shards, dim=0))
    for i, shard in enumerate(chunks):
        shard = shard.detach().requires_grad_(True)
        off, step = i * chunks[0].shape[0], shard.shape[0]
        gs = gf.narrow(0, off, step)
        with torch.enable_grad():
            out = mlp(shard)
            grads = torch.autograd.grad(out, [shard] + list(mlp.parameters()), gs)
        x_grad.narrow(0, off, step).copy_(grads[0])
        for p, g in zip(mlp.parameters(), grads[1:]):
            param_grads[p] = g if param_grads[p] is None else param_grads[p] + g
    for p, g in param_grads.items():
        p.grad = g if p.grad is None else p.grad + g
    return x_grad.view(orig)


def backward_main(x, mlp, shards, grad_out):
    """main: pre-set .grad views, single torch.autograd.backward over all shards."""
    h = x.shape[-1]; orig = x.shape
    xf = x.detach().view(-1, h).requires_grad_(True)
    gf = grad_out.view(-1, h)
    x_grad = torch.zeros_like(xf)
    chunks = list(torch.chunk(xf, shards, dim=0))
    all_out, all_g = [], []
    for i, shard in enumerate(chunks):
        shard.requires_grad_(True)
        off, step = i * chunks[0].shape[0], shard.shape[0]
        shard.grad = x_grad.narrow(0, off, step).view_as(shard)
        with torch.enable_grad():
            all_out.append(mlp(shard))
            all_g.append(gf.narrow(0, off, step).view_as(shard))
    torch.autograd.backward(all_out, all_g)
    return x_grad.view(orig)


def bench(fn, warmup=2, reps=5):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    times, mems = [], []
    for _ in range(reps):
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        mems.append(torch.cuda.max_memory_allocated() / 1024**2)
    return sum(times)/reps, sum(mems)/reps


CONFIGS = [
    # hidden, seqlen, shards, intermediate
    (1024,  2048,  4,  4096),
    (2048,  4096,  4,  8192),
    (2048,  8192,  8,  8192),
    (4096,  8192,  8, 14336),
    (4096, 16384, 16, 14336),
]

print(f"\n{'hidden':>8} {'seqlen':>7} {'shards':>7}   {'branch_ms':>10} {'branch_MB':>10}   {'main_ms':>8} {'main_MB':>8}   {'Δms':>7} {'ΔMB':>7}")
print("-" * 90)

for h, s, shards, inter in CONFIGS:
    dev, dt = "cuda", torch.bfloat16
    mlp_b = MLP(h, inter).to(dev, dt)
    mlp_m = MLP(h, inter).to(dev, dt)
    with torch.no_grad():
        for pb, pm in zip(mlp_b.parameters(), mlp_m.parameters()):
            pm.copy_(pb)
    x    = torch.randn(1, s, h, device=dev, dtype=dt)
    g    = torch.randn_like(x)

    def run_b(): [p.__setattr__('grad', None) for p in mlp_b.parameters()]; backward_branch(x, mlp_b, shards, g)
    def run_m(): [p.__setattr__('grad', None) for p in mlp_m.parameters()]; backward_main(x, mlp_m, shards, g)

    tb, mb = bench(run_b)
    tm, mm = bench(run_m)
    print(f"{h:>8} {s:>7} {shards:>7}   {tb:>10.2f} {mb:>10.1f}   {tm:>8.2f} {mm:>8.1f}   {tb-tm:>+7.2f} {mb-mm:>+7.1f}")

print("-" * 90)
print("Δ = branch − main  (negative ΔMB = branch uses less memory)")
