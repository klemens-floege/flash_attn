#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

//Custom FlashAttention CUDA Time: 0.00157285s
//PyTorch Attention Time: 0.00096726s

__global__ void flash_attention_multihead_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int B, int H, int S, int D)
{
    int b = blockIdx.x;  // batch index
    int h = blockIdx.y;  // head index
    int q = threadIdx.x; // query index
    if (b >= B || h >= H || q >= S) return;
    
    float score_sum = 0.0f;
    float scores[1024];  // Assume S <= 1024 for simplicity
    float scale = 1.0f / sqrtf((float)D);
    
    for (int k = 0; k < S; k++) {
        float score = 0.0f;
        for (int d = 0; d < D; d++) {
            score += Q[b * H * S * D + h * S * D + q * D + d] * K[b * H * S * D + h * S * D + k * D + d];
        }
        score *= scale;
        scores[k] = expf(score);
        score_sum += scores[k];
    }
    
    for (int d = 0; d < D; d++) {
        float out_val = 0.0f;
        for (int k = 0; k < S; k++) {
            out_val += (scores[k] / score_sum) * V[b * H * S * D + h * S * D + k * D + d];
        }
        output[b * H * S * D + h * S * D + q * D + d] = out_val;
    }
}

void launch_flash_attention_multihead(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O)
{
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);
    
    dim3 grid(B, H);
    dim3 block(S);
    
    flash_attention_multihead_kernel<<<grid, block>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), B, H, S, D);
}

PYBIND11_MODULE(flash_attention_cuda, m) {
    m.def("flash_attention", &launch_flash_attention_multihead, "Simple Multi-Head Attention (CUDA)");
}


