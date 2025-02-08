#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_add(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int size = a.numel();
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    add_kernel<<<blocks_per_grid, threads_per_block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        size
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &launch_add, "Element-wise addition (CUDA)");
}
