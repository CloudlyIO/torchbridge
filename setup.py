"""
Setup script for building custom CUDA extensions.

This script enables compilation of custom CUDA kernels alongside PyTorch,
demonstrating how to integrate low-level GPU programming with high-level ML frameworks.
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Get CUDA toolkit path
def get_cuda_toolkit_path():
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Try to find CUDA installation
        import subprocess
        try:
            nvcc_path = subprocess.check_output(['which', 'nvcc']).decode().strip()
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        except:
            cuda_home = '/usr/local/cuda'
    return cuda_home

# CUDA extension for custom kernels
def create_cuda_extension():
    cuda_sources = [
        'src/kernel_pytorch/cuda_kernels/fused_ops.cu',
        'src/kernel_pytorch/cuda_kernels/cuda_interface.cpp'
    ]

    # CUDA compilation flags
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
        '-lineinfo',
    ]

    # C++ compilation flags
    cxx_flags = [
        '-O3',
        '-std=c++17',
    ]

    cuda_home = get_cuda_toolkit_path()
    include_dirs = [
        'src/kernel_pytorch/cuda_kernels',
        f'{cuda_home}/include',
        f'{cuda_home}/include/cub',  # For CUB library
    ]

    return CUDAExtension(
        name='kernel_pytorch_cuda',
        sources=cuda_sources,
        include_dirs=include_dirs,
        extra_compile_args={
            'cxx': cxx_flags,
            'nvcc': nvcc_flags
        },
        libraries=['cudart', 'cublas', 'curand'],
        library_dirs=[f'{cuda_home}/lib64'],
    )

# Check if we can build CUDA extensions
def cuda_is_available():
    try:
        import torch
        return torch.cuda.is_available() and torch.version.cuda is not None
    except:
        return False

# Setup configuration
ext_modules = []
cmdclass = {}

if cuda_is_available():
    print("CUDA detected, building CUDA extensions...")
    ext_modules.append(create_cuda_extension())
    cmdclass['build_ext'] = BuildExtension
else:
    print("CUDA not available, skipping CUDA extensions...")

setup(
    name='kernel-pytorch',
    version='1.0.0',
    author='KernelPyTorch Team',
    description='Kernel-optimized PyTorch components for ML education',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=[
        'kernel_pytorch',
        'kernel_pytorch.advanced_attention',
        'kernel_pytorch.advanced_memory',
        'kernel_pytorch.attention',
        'kernel_pytorch.compiler_integration',
        'kernel_pytorch.compiler_optimized',
        'kernel_pytorch.components',
        'kernel_pytorch.distributed_scale',
        'kernel_pytorch.gpu_integration',
        'kernel_pytorch.hardware_abstraction',
        'kernel_pytorch.mixture_of_experts',
        'kernel_pytorch.next_gen_optimizations',
        'kernel_pytorch.optimization_patterns',
        'kernel_pytorch.precision',
        'kernel_pytorch.testing_framework',
        'kernel_pytorch.utils',
    ],
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'triton>=2.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'jupyter>=1.0.0',
            'tensorboard>=2.9.0',
        ],
        'benchmark': [
            'memory-profiler',
            'py-spy',
            'torch-tb-profiler',
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False,
)

# Additional build instructions
if __name__ == '__main__':
    print("\n" + "="*60)
    print("KERNEL PYTORCH SETUP")
    print("="*60)

    if cuda_is_available():
        print("✓ CUDA toolkit detected")
        print("✓ Will build custom CUDA kernels")
        print("\nBuild commands:")
        print("  python setup.py build_ext --inplace  # Development build")
        print("  pip install -e .                      # Editable install")
        print("  pip install .                         # Production install")
    else:
        print("⚠ CUDA not available")
        print("  Only CPU-based components will be available")
        print("  Install CUDA toolkit and PyTorch with CUDA for full functionality")

    print("\nOptimization levels available:")
    print("  Level 1: PyTorch native (always available)")
    print("  Level 2: TorchScript JIT (always available)")
    print("  Level 3: torch.compile (PyTorch 2.0+)")
    if cuda_is_available():
        print("  Level 4: Triton kernels (CUDA required)")
        print("  Level 5: Custom CUDA kernels (CUDA required)")

    print("\nAfter installation, run:")
    print("  python -c 'import kernel_pytorch; print(\"Installation successful!\")'")
    print("="*60)