"""Test triton on Windows."""
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')
out = torch.zeros_like(x)
grid = lambda meta: (triton.cdiv(1024, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, out, 1024, BLOCK_SIZE=256)
diff = (out - x - y).abs().max().item()
print(f"Triton OK: diff={diff:.8f}")
