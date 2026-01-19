"""
CUDA Extension Build Script for KernelPyTorch

This script handles building custom CUDA kernels only.
Package metadata is defined in pyproject.toml (PEP 621).

Version: 0.4.3 (synced with pyproject.toml)

Usage:
    pip install -e .                  # Editable install (uses pyproject.toml)
    python setup.py build_ext --inplace  # Build CUDA extensions only
"""

__version__ = "0.4.4"  # Keep in sync with pyproject.toml

from setuptools import setup
import os
import sys

# Only import CUDA-related modules if we're building extensions
def get_cuda_extension():
    """Create CUDA extension if CUDA is available."""
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        if not torch.cuda.is_available():
            return None, {}

    except ImportError:
        return None, {}

    # Get CUDA toolkit path
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        try:
            import subprocess
            nvcc_path = subprocess.check_output(['which', 'nvcc']).decode().strip()
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        except Exception:
            cuda_home = '/usr/local/cuda'

    cuda_sources = [
        'src/kernel_pytorch/cuda_kernels/fused_ops.cu',
        'src/kernel_pytorch/cuda_kernels/flash_attention_v3.cu',
        'src/kernel_pytorch/cuda_kernels/fused_linear_activation.cu',
        'src/kernel_pytorch/cuda_kernels/cuda_interface.cpp',
    ]

    # Check which source files exist
    existing_sources = [s for s in cuda_sources if os.path.exists(s)]
    if not existing_sources:
        print("No CUDA source files found, skipping CUDA extension build")
        return None, {}

    nvcc_flags = [
        '-O3',
        '--use_fast_math',
        '-Xptxas=-O3',
        '--expt-relaxed-constexpr',
        '-gencode=arch=compute_70,code=sm_70',  # V100
        '-gencode=arch=compute_75,code=sm_75',  # T4, RTX 20xx
        '-gencode=arch=compute_80,code=sm_80',  # A100, RTX 30xx
        '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
        '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
        '-gencode=arch=compute_90,code=sm_90',  # H100
        '-DENABLE_FP8',
        '-DENABLE_FLASH_ATTENTION_V3',
        '-DENABLE_FUSED_KERNELS',
        '-lineinfo',
    ]

    cxx_flags = ['-O3', '-std=c++17']

    include_dirs = [
        'src/kernel_pytorch/cuda_kernels',
        'src/kernel_pytorch/hardware/kernels',
        f'{cuda_home}/include',
    ]

    ext = CUDAExtension(
        name='kernel_pytorch_cuda',
        sources=existing_sources,
        include_dirs=include_dirs,
        extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags},
        libraries=['cudart', 'cublas', 'curand'],
        library_dirs=[f'{cuda_home}/lib64'],
    )

    return ext, {'build_ext': BuildExtension}


# Only build CUDA extensions when explicitly requested
if 'build_ext' in sys.argv or 'bdist_wheel' in sys.argv or 'install' in sys.argv:
    ext_module, cmdclass = get_cuda_extension()
    ext_modules = [ext_module] if ext_module else []
else:
    ext_modules = []
    cmdclass = {}

# Minimal setup - metadata comes from pyproject.toml
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
