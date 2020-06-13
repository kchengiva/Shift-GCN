from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='shift_cuda_linear_cpp',
    ext_modules=[
        CUDAExtension('shift_cuda', [
            'shift_cuda.cpp',
            'shift_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
