
import triton
import triton.language as tl

@triton.jit
def triton_attention(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qz, stride_qs, stride_qh,
    stride_kz, stride_ks, stride_kh,
    stride_vz, stride_vs, stride_vh,
    stride_oz, stride_os, stride_oh,
    seq_len: tl.constexpr, head_dim: tl.constexpr
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.program_id(2)

    if seq >= seq_len:
        return

    # Load Q and K for softmax
    q = tl.load(Q_ptr + batch * stride_qz + seq * stride_qs + head * stride_qh)
    k = tl.load(K_ptr + batch * stride_kz + head * stride_kh)  # No sequence indexing
    v = tl.load(V_ptr + batch * stride_vz + head * stride_vh)

    # Compute dot product (scaled)
    dot_product = q * k / (head_dim ** 0.5)

    # Apply softmax
    max_val = tl.max(dot_product)
    exp_score = tl.exp(dot_product - max_val)
    sum_exp = tl.sum(exp_score)
    softmax_out = exp_score / sum_exp

    # Compute weighted sum
    out = softmax_out * v

    # Store result
    tl.store(O_ptr + batch * stride_oz + seq * stride_os + head * stride_oh, out)
