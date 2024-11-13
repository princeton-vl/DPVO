import os.path as osp
from setuptools import setup, find_packages

ROOT = osp.dirname(osp.abspath(__file__))

def get_extensions():
    from torch.utils.cpp_extension import CUDAExtension
    return [
        CUDAExtension('cuda_corr',
            sources=['dpvo/altcorr/correlation.cpp', 'dpvo/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': ['-O3'],
            }),
        CUDAExtension('cuda_ba',
            sources=['dpvo/fastba/ba.cpp', 'dpvo/fastba/ba_cuda.cu', 'dpvo/fastba/block_e.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': ['-O3'],
            }),
    ]
    # return []

def get_build_extension():
    # return None
    from torch.utils.cpp_extension import BuildExtension
    return BuildExtension

setup(
    name='dpvo',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio',
    ],
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': get_build_extension()
    },
    )


