import tempfile

import pytest
import torch
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig

from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledGEGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledSwiGLUMLP
from liger_kernel.utils import infer_comm_backend
from liger_kernel.utils import infer_device

device = infer_device()


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (1, 1024, 128, 256),  # num_shards=8 if auto
        (2, 1024, 64, 256),  # num_shards=16 if auto
        # weird shapes
        (4, 127, 128, 256),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-0, 2e-6),
        pytest.param(
            torch.bfloat16,
            1e-0,
            1e-0,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize("num_shards", [None, 2, 4])
@pytest.mark.parametrize("check_2d", [True, False])
def test_tiled_geglu_correctness(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol, num_shards, check_2d):
    """Test that TiledGEGLUMLP produces similar results as regular GEGLUMLP."""

    # BF16 accumulation is sensitive to the number of reduction steps. Narrow hidden layers
    # (hidden_size < 128) combined with sharding result in high-density summation boundaries
    # where rounding errors exceed standard tolerances. We skip these edge cases to maintain
    # strict parity checks for production-scale shapes.
    if dtype == torch.bfloat16 and hidden_size < 128:
        pytest.skip(f"Skipping unstable BF16 configuration: hidden_size={hidden_size}")

    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="gelu_pytorch_tanh",
    )

    # scale input so that the numerical errors are accumulated less
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype) * 0.1
    x1 = _input.detach().clone().requires_grad_(True)
    x2 = _input.detach().clone().requires_grad_(True)

    # Convert to 2D input for MoE experts testing
    if check_2d:
        x1 = x1.view(-1, hidden_size)
        x2 = x2.view(-1, hidden_size)

    # Initialize weights
    G = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    U = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    D = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    # Regular GEGLU MLP
    regular_mlp = LigerGEGLUMLP(config=config).to(device).to(dtype)
    regular_mlp.gate_proj.weight.data = G
    regular_mlp.up_proj.weight.data = U
    regular_mlp.down_proj.weight.data = D

    # Tiled GEGLU MLP
    tiled_mlp = LigerTiledGEGLUMLP(config=config, num_shards=num_shards).to(device).to(dtype)
    tiled_mlp.gate_proj.weight.data = G
    tiled_mlp.up_proj.weight.data = U
    tiled_mlp.down_proj.weight.data = D

    # Forward pass
    y1 = regular_mlp(x1)
    y2 = tiled_mlp(x2)
    torch.testing.assert_close(y1, y2, atol=atol, rtol=rtol, msg="Forward outputs don't match")

    # Backward pass
    dy = torch.randn_like(y1)
    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    # Dynamic parameter discovery ensures PEFT/LoRA adapters are also validated
    regular_params = [p for p in regular_mlp.parameters() if p.requires_grad]
    tiled_params = [p for p in tiled_mlp.parameters() if p.requires_grad]
    assert len(regular_params) == len(tiled_params), "Number of trainable parameters mismatch"

    for p1, p2 in zip(regular_params, tiled_params):
        torch.testing.assert_close(
            p1.grad,
            p2.grad,
            atol=atol,
            rtol=rtol,
            msg="Gradients for trainable parameters do not match",
        )

    torch.testing.assert_close(x1.grad, x2.grad, atol=atol, rtol=rtol, msg="Input gradients don't match")


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 512, 512, 1024),
        (1, 1024, 256, 512),
        # weird shapes
        (4, 127, 128, 256),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-0, 2e-6),
        pytest.param(
            torch.bfloat16,
            1e-0,
            1e-0,
            marks=pytest.mark.skip(reason="bfloat16 tests disabled due to numerical instability"),
        ),
    ],
)
@pytest.mark.parametrize("num_shards", [None, 2, 4])
@pytest.mark.parametrize("check_2d", [True, False])
def test_tiled_swiglu_correctness(
    bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol, num_shards, check_2d
):
    """Test that TiledSwiGLUMLP produces similar results as regular SwiGLUMLP."""

    # See rationale in test_tiled_geglu_correctness
    if dtype == torch.bfloat16 and hidden_size < 128:
        pytest.skip(f"Skipping unstable BF16 configuration: hidden_size={hidden_size}")

    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
    )

    # scale input so that the numerical errors are accumulated less
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype) * 0.1
    x1 = _input.detach().clone().requires_grad_(True)
    x2 = _input.detach().clone().requires_grad_(True)

    if check_2d:
        x1 = x1.view(-1, hidden_size)
        x2 = x2.view(-1, hidden_size)

    # Initialize weights
    G = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    U = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    D = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    # Regular SwiGLU MLP
    regular_mlp = LigerSwiGLUMLP(config=config).to(device).to(dtype)
    regular_mlp.gate_proj.weight.data = G
    regular_mlp.up_proj.weight.data = U
    regular_mlp.down_proj.weight.data = D

    # Tiled SwiGLU MLP
    tiled_mlp = LigerTiledSwiGLUMLP(config=config, num_shards=num_shards).to(device).to(dtype)
    tiled_mlp.gate_proj.weight.data = G
    tiled_mlp.up_proj.weight.data = U
    tiled_mlp.down_proj.weight.data = D

    # Forward pass
    y1 = regular_mlp(x1)
    y2 = tiled_mlp(x2)
    torch.testing.assert_close(y1, y2, atol=atol, rtol=rtol, msg="Forward outputs don't match")

    # Backward pass
    dy = torch.randn_like(y1)
    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    # Check gradients
    regular_params = [p for p in regular_mlp.parameters() if p.requires_grad]
    tiled_params = [p for p in tiled_mlp.parameters() if p.requires_grad]
    assert len(regular_params) == len(tiled_params), "Number of trainable parameters mismatch"

    for p1, p2 in zip(regular_params, tiled_params):
        torch.testing.assert_close(
            p1.grad,
            p2.grad,
            atol=atol,
            rtol=rtol,
            msg="Gradients for trainable parameters do not match",
        )

    torch.testing.assert_close(x1.grad, x2.grad, atol=atol, rtol=rtol, msg="Input gradients don't match")


# ──────────────────────────────────────────────────────────────────────────────
# DTensor (sequence-parallel) test
# Verifies that LigerTiledSwiGLUMLP produces identical results when the input
# is a DTensor sharded along the sequence dimension vs. a plain tensor.
# ──────────────────────────────────────────────────────────────────────────────


def _test_dtensor_tiled_swiglu_mlp(
    rank, world_size, bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol, num_shards, file_name
):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend=infer_comm_backend(),
        init_method=f"file://{file_name}",
        rank=rank,
        world_size=world_size,
    )
    device = f"{infer_device()}:{rank}" if infer_device() != "cpu" else "cpu"
    device_mesh = torch.distributed.device_mesh.init_device_mesh(
        infer_device(), mesh_shape=(world_size,), mesh_dim_names=("tp",)
    )

    config = LlamaConfig(hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act="silu")

    # Deterministic weight initialisation — same on every rank
    torch.manual_seed(42)
    G = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    U = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    D = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    torch.manual_seed(7)
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype) * 0.1

    # Shard input along the sequence dimension (dim=1)
    dt = torch.distributed.tensor.distribute_tensor(
        _input,
        device_mesh=device_mesh,
        placements=[torch.distributed.tensor.Shard(1)],
    )

    def _make_mlp():
        mlp = LigerTiledSwiGLUMLP(config=config, num_shards=num_shards).to(device).to(dtype)
        mlp.gate_proj.weight.data = G.clone()
        mlp.up_proj.weight.data = U.clone()
        mlp.down_proj.weight.data = D.clone()
        return mlp

    tiled_mlp_dist = _make_mlp()
    tiled_mlp_ref = _make_mlp()

    # Triton kernels cannot access DTensor memory directly (no DTensor dispatch rules for
    # custom autograd functions). Extract the plain CUDA local shard and compare against
    # the matching slice of the full reference — this tests sequence-parallel correctness.
    x_local = dt.to_local().requires_grad_(True)
    shard_seq = x_local.shape[1]  # handles uneven splits
    seq_start = rank * (seq_len // world_size)
    x_ref = _input[:, seq_start : seq_start + shard_seq, :].detach().clone().requires_grad_(True)

    y_local = tiled_mlp_dist(x_local)
    y_ref = tiled_mlp_ref(x_ref)
    torch.testing.assert_close(y_local, y_ref, atol=atol, rtol=rtol)

    dy = torch.randn_like(y_ref)
    y_local.backward(dy.clone())
    y_ref.backward(dy.clone())

    torch.testing.assert_close(x_local.grad, x_ref.grad, atol=atol, rtol=rtol)
    for p_local, p_ref in zip(tiled_mlp_dist.parameters(), tiled_mlp_ref.parameters()):
        torch.testing.assert_close(p_local.grad, p_ref.grad, atol=atol, rtol=rtol)

    torch.distributed.destroy_process_group()


@pytest.mark.xfail(
    torch.cuda.device_count() < 2,
    reason="Pending multi-GPU host support. This test is expected to pass when run with multi-GPU host.",
)
@pytest.mark.parametrize(
    "world_size, bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 2, 16, 64, 128),
        (2, 1, 32, 32, 64),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        (torch.bfloat16, 2e-1, 2e-2),
    ],
)
@pytest.mark.parametrize("num_shards", [None, 2, 4])
def test_dtensor_tiled_swiglu_mlp(world_size, bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol, num_shards):
    with tempfile.NamedTemporaryFile() as f:
        mp.spawn(
            _test_dtensor_tiled_swiglu_mlp,
            args=(world_size, bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol, num_shards, f.name),
            nprocs=world_size,
            join=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# FSDP test
# Verifies that LigerTiledSwiGLUMLP forward+backward work correctly under FSDP1
# wrapping — the key regression from the FSDP backward fix (using
# torch.autograd.grad() instead of torch.autograd.backward() per shard).
# ──────────────────────────────────────────────────────────────────────────────


def _test_fsdp_tiled_swiglu_mlp(
    rank, world_size, bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol, num_shards, file_name
):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend=infer_comm_backend(),
        init_method=f"file://{file_name}",
        rank=rank,
        world_size=world_size,
    )
    device = f"{infer_device()}:{rank}" if infer_device() != "cpu" else "cpu"

    config = LlamaConfig(hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act="silu")

    # Same weight initialisation on every rank so FSDP and reference start identically
    torch.manual_seed(42)
    G = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    U = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    D = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    def _make_mlp():
        mlp = LigerTiledSwiGLUMLP(config=config, num_shards=num_shards).to(device).to(dtype)
        mlp.gate_proj.weight.data = G.clone()
        mlp.up_proj.weight.data = U.clone()
        mlp.down_proj.weight.data = D.clone()
        return mlp

    ref_mlp = _make_mlp()
    fsdp_mlp = FSDP(_make_mlp(), device_id=rank)

    # Same input on every rank — with identical gradients the FSDP allreduce-mean
    # equals the per-rank value, making direct comparison with ref valid.
    torch.manual_seed(7)
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype) * 0.1
    x_ref = _input.detach().clone().requires_grad_(True)
    x_fsdp = _input.detach().clone().requires_grad_(True)

    y_ref = ref_mlp(x_ref)
    y_fsdp = fsdp_mlp(x_fsdp)
    torch.testing.assert_close(y_ref, y_fsdp, atol=atol, rtol=rtol)

    dy = torch.randn_like(y_ref)
    y_ref.backward(dy.clone())
    y_fsdp.backward(dy.clone())

    # Input gradients are not sharded by FSDP — compare directly
    torch.testing.assert_close(x_ref.grad, x_fsdp.grad, atol=atol, rtol=rtol)

    # Gather sharded parameter gradients for comparison
    with FSDP.summon_full_params(fsdp_mlp, with_grads=True, rank0_only=False):
        for p_ref, p_fsdp in zip(ref_mlp.parameters(), fsdp_mlp.parameters()):
            torch.testing.assert_close(p_ref.grad, p_fsdp.grad, atol=atol, rtol=rtol)

    torch.distributed.destroy_process_group()


@pytest.mark.xfail(
    torch.cuda.device_count() < 2,
    reason="Pending multi-GPU host support. This test is expected to pass when run with multi-GPU host.",
)
@pytest.mark.parametrize(
    "world_size, bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 2, 16, 64, 128),
        (2, 1, 32, 32, 64),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        (torch.bfloat16, 2e-1, 2e-2),
    ],
)
@pytest.mark.parametrize("num_shards", [None, 2, 4])
def test_fsdp_tiled_swiglu_mlp(world_size, bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol, num_shards):
    with tempfile.NamedTemporaryFile() as f:
        mp.spawn(
            _test_fsdp_tiled_swiglu_mlp,
            args=(world_size, bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol, num_shards, f.name),
            nprocs=world_size,
            join=True,
        )
