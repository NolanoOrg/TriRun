from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='trirun',
    version='0.1.0',
    author='Francis Couture-Harpin',
    author_email='git@compilade.net',
    description='Highly optimized FP16xINT2 CUDA matmul kernel.',
    install_requires=['numpy', 'torch'],
    packages=['trirun'],
    ext_modules=[cpp_extension.CUDAExtension(
        'trirun_cuda', ['trirun/trirun_cuda.cpp', 'trirun/trirun_cuda_kernel.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
