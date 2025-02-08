#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

//B, H, S, D = 64, 128, 4096, 128

#define TILE_SIZE 128  // tile over the "key" dimension
#define D_MAX     128  // dimension per head

__global__ void flash_attention_multihead_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int B, int H, int S, int D)
{
    extern __shared__ float shared_mem[];
    
    int b = blockIdx.y;  // batch
    int h = blockIdx.z;  // head

    // Query index = blockIdx.x * blockDim.x + threadIdx.x
    int q = blockIdx.x * blockDim.x + threadIdx.x;

    // If q is out of range, do nothing
    if (b >= B || h >= H || q >= S) {
        return;
    }

    // Each thread handles exactly one query (q) for this (b,h).
    // We'll tile over the "key" positions (S) using shared memory.
    __shared__ float K_tile[TILE_SIZE * D_MAX];  // shape = (TILE_SIZE, D)
    __shared__ float V_tile[TILE_SIZE * D_MAX];  // shape = (TILE_SIZE, D)

    // Scale factor for dot-products
    float scale = 1.0f / sqrtf((float)D);

    // Offsets into Q, K, V for (b, h)
    const float* Q_row = Q + (b * H * S * D) + (h * S * D) + (q * D);
    const float* K_head = K + (b * H * S * D) + (h * S * D);
    const float* V_head = V + (b * H * S * D) + (h * S * D);

    // Output offset
    float* O_row = output + (b * H * S * D) + (h * S * D) + (q * D);

    // First pass: find the maximum logit (for numerical stability)
    float max_logit = -std::numeric_limits<float>::infinity();

    for (int s_tile = 0; s_tile < S; s_tile += TILE_SIZE) {
        int local_s = s_tile + threadIdx.x;
        // Each thread loads 1 row of K, V into shared memory if in range
        if (local_s < S) {
            for (int d = 0; d < D; d++) {
                K_tile[threadIdx.x * D + d] = K_head[local_s * D + d];
                V_tile[threadIdx.x * D + d] = V_head[local_s * D + d];
            }
        }
        __syncthreads();

        // Compute dot-products for all positions in [s_tile, s_tile + TILE_SIZE)
        // The local index 's' in the tile is 0..(TILE_SIZE-1)
        // but we must check if (s_tile+s) < S
        for (int s = 0; s < TILE_SIZE && (s_tile + s) < S; s++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += Q_row[d] * K_tile[s * D + d];
            }
            dot *= scale;
            if (dot > max_logit) {
                max_logit = dot;
            }
        }
        __syncthreads();
    }

    // Second pass: compute the sum of exp(...) and the weighted values
    float sum_exp = 0.0f;
    float out_acc[D_MAX];
    for (int d = 0; d < D; d++) {
        out_acc[d] = 0.0f;
    }

    for (int s_tile = 0; s_tile < S; s_tile += TILE_SIZE) {
        int local_s = s_tile + threadIdx.x;
        // Reload tile from global memory
        if (local_s < S) {
            for (int d = 0; d < D; d++) {
                K_tile[threadIdx.x * D + d] = K_head[local_s * D + d];
                V_tile[threadIdx.x * D + d] = V_head[local_s * D + d];
            }
        }
        __syncthreads();

        // Accumulate softmax contributions
        for (int s = 0; s < TILE_SIZE && (s_tile + s) < S; s++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += Q_row[d] * K_tile[s * D + d];
            }
            dot *= scale;
            float w = expf(dot - max_logit);
            sum_exp += w;
            // Weighted accumulation of V
            for (int d = 0; d < D; d++) {
                out_acc[d] += w * V_tile[s * D + d];
            }
        }
        __syncthreads();
    }

    // Write final normalized output
    for (int d = 0; d < D; d++) {
        O_row[d] = out_acc[d] / sum_exp;
    }
    
}

void launch_flash_attention_multihead(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O)
{
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);
    
    dim3 block(std::min(S, TILE_SIZE));  // Threads per block (128 max)
    dim3 grid(B, H);  // One block per batch/head

    size_t shared_memory_size = 2 * TILE_SIZE * 128 * sizeof(float);  // Shared memory for K/V

    flash_attention_multihead_kernel<<<grid, block, shared_memory_size>>>(
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
