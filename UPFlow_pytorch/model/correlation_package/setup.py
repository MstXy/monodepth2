#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14']

nvcc_args = [
    # '-gencode', 'arch=compute_50,code=sm_50',
    # '-gencode', 'arch=compute_52,code=sm_52',
    # '-gencode', 'arch=compute_60,code=sm_60',
    # '-gencode', 'arch=compute_61,code=sm_61',
    # '-gencode', 'arch=compute_61,code=compute_61',
    '-gencode', 'arch=compute_86,code=sm_86',
    '-ccbin', '/usr/bin/gcc-9'
    # '-ccbin', '/usr/bin/gcc-4.9'
]


setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension(
            name='correlation_cuda',
            sources=['correlation_cuda.cc','correlation_cuda_kernel.cu'],
            extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args, 'cuda-path': ['/usr/local/cuda']},
            # extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
            include_dirs=['/home/liu/anaconda3/envs/upflow/lib/python3.8/site-packages/pybind11/include/'])
],
    cmdclass={
        'build_ext': BuildExtension
    })
