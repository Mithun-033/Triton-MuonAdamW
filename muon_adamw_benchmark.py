import copy

import numpy as np
import torch
from torch import nn

from MuonAdamW import AdamConfig, MuonAdamW, MuonConfig, PyTorchMuonAdamW

model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.LayerNorm(2048),
    nn.ReLU(),
    nn.Linear(2048, 4096),
    nn.LayerNorm(4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    nn.LayerNorm(4096),
    nn.ReLU(),
    nn.Linear(4096, 1024),
).cuda()

X = torch.randn(1000, 1024, device="cuda")
Y = torch.randn(1000, 1024, device="cuda")

model_triton = model
model_pytorch = copy.deepcopy(model)

adam_config = AdamConfig(lr=1e-3, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.01)
muon_config = MuonConfig(
    lr=1e-3,
    beta=0.95,
    eps=1e-7,
    weight_decay=0.1,
    coeffs=(3.445, -4.775, 2.0315),
    nesterov=True,
    ns_steps=5,
    adjust_lr_fn="original",
)


def split_parameters(model):
    adam_params = []
    muon_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            muon_params.append(param)
        else:
            adam_params.append(param)
    return {"adam": adam_params, "muon": muon_params}


triton_optimizer = MuonAdamW(
    split_parameters(model_triton),
    muon_config=muon_config,
    adam_config=adam_config,
    parameter_split="explicit",
)
pytorch_optimizer = PyTorchMuonAdamW(
    split_parameters(model_pytorch),
    muon_config=muon_config,
    adam_config=adam_config,
    parameter_split="explicit",
)
compiled_pytorch_step = torch.compile(pytorch_optimizer.step)


def check_close(name, a, b, atol=1e-6, rtol=1e-5):
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    close = np.allclose(a_np, b_np, atol=atol, rtol=rtol)
    max_abs_diff = np.max(np.abs(a_np - b_np))
    max_rel_diff = np.max(np.abs(a_np - b_np) / (np.abs(b_np) + 1e-8))
    print(
        f"  {name}: {'OK' if close else 'FAIL'}  max_abs={max_abs_diff:.2e}  max_rel={max_rel_diff:.2e}"
    )
    return close


def get_grads(model):
    model.zero_grad()
    output = model(X)
    loss = torch.nn.functional.mse_loss(output, Y)
    loss.backward()


get_grads(model_triton)
get_grads(model_pytorch)

print("Gradient check:")
for (name_triton, param_triton), (name_pytorch, param_pytorch) in zip(
    model_triton.named_parameters(), model_pytorch.named_parameters()
):
    if param_triton.grad is not None:
        check_close(name_triton, param_triton.grad, param_pytorch.grad)


def benchmark_optimizer(step_fn, warmup=50, iterations=100):
    for _ in range(warmup):
        step_fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        step_fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations


iterations = 5
triton_ms = 0
pytorch_ms = 0

for _ in range(iterations):
    triton_ms += benchmark_optimizer(triton_optimizer.step)
    pytorch_ms += benchmark_optimizer(compiled_pytorch_step)

triton_ms /= iterations
pytorch_ms /= iterations

print("\nFinal parameter check:")
all_close = True
for (name_triton, param_triton), (name_pytorch, param_pytorch) in zip(
    model_triton.named_parameters(), model_pytorch.named_parameters()
):
    all_close &= check_close(name_triton, param_triton, param_pytorch, atol=1e-2, rtol=1e-2)
print(f"All parameters match: {'OK' if all_close else 'FAIL'}")

num_params = sum(param.numel() for param in model_triton.parameters())

print(f"Parameters:       {num_params:,}")
print()
print("Triton MuonAdamW:")
print(f"  Time:           {triton_ms:.4f} ms")
print()
print("PyTorch MuonAdamW:")
print(f"  Time:           {pytorch_ms:.4f} ms")
print()
print(f"Speedup:          {pytorch_ms / triton_ms:.2f}x")

torch.cuda.synchronize()
print(f"Memory Triton:    {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
torch.cuda.reset_peak_memory_stats()
print(f"Memory PyTorch:   {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
