import torch
import cuda_add  # Import the compiled CUDA extension

# Generate random tensors on CUDA
size = 1024
a = torch.randn(size, device='cuda', dtype=torch.float32)
b = torch.randn(size, device='cuda', dtype=torch.float32)
c = torch.empty(size, device='cuda', dtype=torch.float32)

# Call the CUDA kernel
cuda_add.add(a, b, c)

# Validate results
torch.testing.assert_close(c, a + b, rtol=1e-4, atol=1e-5)
print("Test passed! CUDA kernel works correctly.")
