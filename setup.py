from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



setup(
    name='flash_attention_cuda',  # Name should match the PyBind11 module
    ext_modules=[
        CUDAExtension(
            name='flash_attention_cuda',  # Match PyBind11 module
            sources=['flash_attention_cuda.cu'],  # Ensure file exists
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
