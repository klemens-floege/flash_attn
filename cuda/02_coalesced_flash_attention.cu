#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

//Custom FlashAttention CUDA Time: 0.00203109s
//PyTorch Attention Time: 0.04485536s

//B, H, S, D = 2, 4, 128, 64  # Batch, Heads, Sequence, Head Dim

__global__ void flash_attention_multihead_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int B, int H, int S, int D)
{
    // b,h come from gridDim
    int b = blockIdx.x;  // batch index
    int h = blockIdx.y;  // head index
    //int q = threadIdx.x; // query index

    // -- Transpose-like re-indexing of threadIdx.x into q_coalesced --
    //    blockDim.x assumed to be 32*32 = 1024 (for typical flash-attention sizes).
    //    This re-maps the single dimension of threads into a 32×32 tile.
    int tid = threadIdx.x;            // 0..1023
    int warp_row = tid / 32;         // 0..31  ("row" in the tile), which warp a thread belongs to.
    int warp_col = tid % 32;         // 0..31  ("column" in the tile), which thread inside the warp it is.
    int q_coalesced = warp_col * 32 + warp_row; 

    //If S>1024, this simple version won’t cover all queries in a single block 
    //and you’d need a tiling strategy.
    // If S < 1024, then some threads won't have a valid q.
    if (q_coalesced >= S || b >= B || h >= H) {
        return;
    }

    float scores[1024];   // hold up to S=1024
    float score_sum = 0.0f;
    float scale = 1.0f / sqrtf((float)D);

    // Base pointers to Q, K, V for (b,h)
    // Each is sized [B, H, S, D] in row-major:
    //   offset = ((b*H + h)*S + some_q)*D + d
    const float* Q_ptr = Q + ((b * H + h) * S + q_coalesced) * D;

    // -----------------------------
    // 1) Compute all attention scores for this q_coalesced
    // -----------------------------
    for (int k = 0; k < S; k++) {
        float score = 0.0f;

        const float* K_ptr = K + ((b * H + h) * S + k) * D;
        // Dot product over D
        for (int d = 0; d < D; d++) {
            score += Q_ptr[d] * K_ptr[d];
        }
        score *= scale;
        scores[k] = expf(score);
        score_sum += scores[k];
    }
    
    // -----------------------------
    // 2) Compute the output by weighting V
    // -----------------------------
    float* out_ptr = output + ((b * H + h) * S + q_coalesced) * D;
    for (int d = 0; d < D; d++) {
        float out_val = 0.0f;
        for (int k = 0; k < S; k++) {
            const float* V_ptr = V + ((b * H + h) * S + k) * D;
            float softmax_val = scores[k] / score_sum;
            out_val += softmax_val * V_ptr[d];
        }
        out_ptr[d] = out_val;
    }
    
}

void launch_flash_attention_multihead(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O)
{
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);
    
    // Typically we want block = 1024 (32*32) for the "transposed" indexing to work well
    dim3 grid(B, H);
    dim3 block(1024);  // up to S=1024
    
    flash_attention_multihead_kernel<<<grid, block>>>(
        Q.data_ptr<float>(), 
        K.data_ptr<float>(), 
        V.data_ptr<float>(), 
        O.data_ptr<float>(), 
        B, H, S, D
    );
}

PYBIND11_MODULE(flash_attention_cuda, m) {
    m.def("flash_attention", &launch_flash_attention_multihead, "Simple Multi-Head Attention (CUDA)");
}


