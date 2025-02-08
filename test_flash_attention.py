import torch
import flash_attention_cuda  # Import the compiled CUDA extension
import triton
import triton.language as tl
import time


from triton_flash_attn import triton_attention
#from flash_attn.flash_attn_interface import flash_attn_func

if torch.cuda.is_available():
    print(f"Available GPU(s): {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available. Running on CPU.")


# Create dummy tensors
B, H, S, D = 64, 128, 2048, 128  # Batch, Heads, Sequence, Head Dim

Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)

# Allocate output tensor
O = torch.zeros_like(Q)

# Benchmark Custom FlashAttention
start = time.time()
flash_attention_cuda.flash_attention(Q, K, V, O)
torch.cuda.synchronize()
custom_time = time.time() - start
custom_output = O.clone()

# Benchmark PyTorch Attention
start = time.time()
torch_attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
torch.cuda.synchronize()
torch_time = time.time() - start

"""# Benchmark Official FlashAttention v2
start = time.time()
flash_v2_output = flash_attn_func(Q, K, V, dropout_p=0.0, causal=False)
torch.cuda.synchronize()
flash_v2_time = time.time() - start"""

# Print Results
print(f"Custom FlashAttention CUDA Time: {custom_time:.8f}s")
print(f"PyTorch Attention Time: {torch_time:.8f}s")


# ✅ Fix: Ensure correct tensor strides and shape extraction
#grid = (batch_size, head_dim, seq_len)


#start = time.time()
#triton_attention[(batch_size, head_dim, seq_len)](Q, K, V, O, BLOCK=128)
#torch.cuda.synchronize()
#triton_time = time.time() - start


#print(f"Triton Attention Time: {triton_time:.8f}s")

# Validate correctness
print("custom", custom_output.shape)
print("torch", torch_attn.shape)
print("custom", custom_output[0])
print("torch", torch_attn[0])
torch.testing.assert_close(custom_output, torch_attn, rtol=1e-4, atol=1e-5)



#torch.testing.assert_close(flash_v2_output, attn, rtol=1e-4, atol=1e-5)
