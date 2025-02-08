#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <cfloat>  // Defines FLT_MAX


//B, H, S, D = 64, 128, 4096, 128

//Tile SIze is GPU dependend (H100 GPU defaults to 48 KB of shared memory per block)
#define TILE_SIZE 32  // tile over the "key" dimension
#define D_MAX     128  // dimension per head

__global__ void flash_attention_multihead_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int B, int H, int S, int D)
{
    extern __shared__ float shared_mem[];

    float* K_tile = shared_mem;                      // First half for K
    float* V_tile = shared_mem + (TILE_SIZE * D);    // Second half for V
    
    int q = blockIdx.x * blockDim.x + threadIdx.x;  // 0..S-1
    int b = blockIdx.y;                             // batch
    int h = blockIdx.z;                             // head

    if (q >= S || b >= B || h >= H) return;

    // Scale factor for dot-products
    float scale = 1.0f / sqrtf((float)D);

    // Offsets into Q, K, V for (b, h)
    const float* Q_row = Q + (b * H * S * D) + (h * S * D) + (q * D);
    const float* K_head = K + (b * H * S * D) + (h * S * D);
    const float* V_head = V + (b * H * S * D) + (h * S * D);

    // Output offset
    float* O_row = output + (b * H * S * D) + (h * S * D) + (q * D);

    // First pass: find the maximum logit (for numerical stability)
    //First Pass (Load tiles, compute scaled QK^T, find max_logit).
    //float max_logit = -std::numeric_limits<float>::infinity();
    float max_logit = -FLT_MAX;

    for (int s_tile = 0; s_tile < S; s_tile += TILE_SIZE) {
        int s_local =  threadIdx.x;
        int global_s = s_tile + s_local;  // Global key index in S

        // Load K and V tiles
        if (global_s < S) {
            for (int d = 0; d < D; d++) {
                K_tile[s_local * D + d] = K_head[global_s * D + d];
                V_tile[s_local * D + d] = V_head[global_s * D + d];
            }
        } else {
            // Zero-pad if outside valid range (prevents undefined reads)
            for (int d = 0; d < D; d++) {
                K_tile[s_local * D + d] = 0.0f;
                V_tile[s_local * D + d] = 0.0f;
            }
        }
        __syncthreads();

        for (int s = 0; s < TILE_SIZE && (s_tile + s) < S; s++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += Q_row[d] * K_tile[s * D + d];
            }
            dot *= scale;
            max_logit = fmaxf(max_logit, dot);
        }
        __syncthreads(); // Ensure max_logit is stable before next tile load
    }


    // Second pass: compute the sum of exp(...) and the weighted values
    //Second Pass (Recompute QK^T, apply softmax, compute output).
    float sum_exp = 0.0f;
    float out_acc[D_MAX];
    for (int d = 0; d < D; d++) {
        out_acc[d] = 0.0f;
    }

    for (int s_tile = 0; s_tile < S; s_tile += TILE_SIZE) {
        int s_local = threadIdx.x; 
        //int s_local = threadIdx.x % TILE_SIZE;  // ✅ Use this if blockDim.x > TILE_SIZE (more threads per block)
        int global_s = s_tile + s_local;

        // Load K and V tiles
        if (global_s < S) {
            for (int d = 0; d < D; d++) {
                K_tile[s_local * D + d] = K_head[global_s * D + d];
                V_tile[s_local * D + d] = V_head[global_s * D + d];
            }
        } else {
            for (int d = 0; d < D; d++) {
                K_tile[s_local * D + d] = 0.0f;
                V_tile[s_local * D + d] = 0.0f;
            }
        }
        __syncthreads();

        for (int s = 0; s < TILE_SIZE && (s_tile + s) < S; s++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += Q_row[d] * K_tile[s * D + d];
            }
            dot *= scale;
            float w = expf(dot - max_logit);
            sum_exp += w;

            for (int d = 0; d < D; d++) {
                out_acc[d] += w * V_tile[s * D + d];
            }
        }
        __syncthreads();
    }

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
    
    size_t shared_memory_size = 2 * TILE_SIZE * D * sizeof(float);  // K_tile + V_tile

    // Number of threads per block in x-dim:
    int block_x = std::min(S, TILE_SIZE);  // e.g. 128
    // Number of blocks needed in x-dim:
    int grid_x  = (S + block_x - 1) / block_x;

    // 3D grid
    dim3 block(TILE_SIZE);
    dim3 grid((S + TILE_SIZE - 1) / TILE_SIZE, B, H);


    std::cout << "Grid: (" << grid_x << ", " << B << ", " << H << ") "
          << " Block: (" << block_x << ", 1, 1) " << std::endl;

    flash_attention_multihead_kernel<<<grid, block, shared_memory_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, H, S, D
    );
 

    // CUDA error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize(); 
}

PYBIND11_MODULE(flash_attention_cuda, m) {
    m.def("flash_attention", &launch_flash_attention_multihead, "Simple Multi-Head Attention (CUDA)");
}
