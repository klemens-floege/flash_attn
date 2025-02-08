from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_add_extension',
    ext_modules=[
        CUDAExtension(
            name='cuda_add',
            sources=['add_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)