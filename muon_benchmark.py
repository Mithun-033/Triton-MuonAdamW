import argparse
import copy

import numpy as np
import torch
import torch.nn as nn

from triton_muon import TritonMuon

parser = argparse.ArgumentParser(description="Benchmark Triton AdamW vs PyTorch AdamW")
parser.add_argument("--fused", action="store_true", help="Use fused AdamW in PyTorch")
args = parser.parse_args()

model = nn.Sequential(
    nn.Linear(1024, 2048, bias = False),
    nn.ReLU(),
    nn.Linear(2048, 4096, bias = False),
    nn.ReLU(),
    nn.Linear(4096,4096, bias = False),
    nn.ReLU(),
    nn.Linear(4096, 4096, bias = False),
    nn.ReLU(),
    nn.Linear(4096, 1024, bias = False),
).cuda()

X = torch.randn(1000, 1024).to("cuda")
Y = torch.randn(1000, 1024).to("cuda")

model_triton = model
model_pytorch = copy.deepcopy(model)

triton_optimizer = TritonMuon(model_triton, coeffs=(3.4445, -4.775, 2.0315), beta=0.95, lr=1e-3, weight_decay=0.1, steps=5)
pytorch_optimizer = torch.optim.Muon(model_pytorch.parameters(), lr = 0.001, momentum = 0.95, nesterov = True, ns_steps = 5)
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
for (n1, p1), (n2, p2) in zip(
    model_triton.named_parameters(), model_pytorch.named_parameters()
):
    if p1.grad is not None:
        check_close(n1, p1.grad, p2.grad, atol=1e-6, rtol=1e-5)


def benchmark_optimizer(step_fn, warmup=50, iterations=100):
    # Warmup
    for _ in range(warmup):
        step_fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for i in range(iterations):
        step_fn()

    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    ms_per_step = total_ms / iterations

    return ms_per_step


iterations = 5
triton_ms = 0
pytorch_ms = 0

for _ in range(iterations):
    triton_ms += benchmark_optimizer(triton_optimizer.step)
    pytorch_ms += benchmark_optimizer(compiled_pytorch_step)

triton_ms /= iterations
pytorch_ms /= iterations

# Final parameter validation
print("\nFinal parameter check:")
all_close = True
for (n1, p1), (n2, p2) in zip(model_triton.named_parameters(), model_pytorch.named_parameters()):
    all_close &= check_close(n1, p1, p2, atol=1e-2, rtol=1e-2)
print(f"All parameters match: {'OK' if all_close else 'FAIL'}")

num_params = sum(p.numel() for p in model_triton.parameters())

print(f"Parameters:       {num_params:,}")
print()
print(f"Triton Muon:")
print(f"  Time:           {triton_ms:.4f} ms")
print()
print(f"PyTorch Muon:")
print(f"  Time:           {pytorch_ms:.4f} ms")
print()
print(f"Speedup:          {pytorch_ms / triton_ms:.2f}x")

# Memory stats
torch.cuda.synchronize()
print(f"Memory Triton:    {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
torch.cuda.reset_peak_memory_stats()
# Note: PyTorch memory already tracked from its benchmark runs
print(f"Memory PyTorch:   {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")