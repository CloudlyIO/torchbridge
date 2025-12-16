# External References and Links

This document maintains a comprehensive list of external resources, documentation, and links for all technologies, concepts, and methodologies used in this project.

## Core Technologies

### PyTorch Ecosystem
- **PyTorch Official**: https://pytorch.org/
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **torch.compile Guide**: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- **TorchScript Documentation**: https://pytorch.org/docs/stable/jit.html
- **TorchInductor**: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747

### CUDA and GPU Programming
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **cuDNN Deep Learning Library**: https://developer.nvidia.com/cudnn
- **cuBLAS**: https://docs.nvidia.com/cuda/cublas/
- **NVIDIA GPU Architecture**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation

### Triton and Kernel Development
- **Triton Language**: https://triton-lang.org/
- **Triton Documentation**: https://triton-lang.org/main/index.html
- **Triton Tutorial**: https://triton-lang.org/main/getting-started/tutorials/index.html
- **OpenAI Triton GitHub**: https://github.com/openai/triton
- **Triton Python API**: https://triton-lang.org/main/python-api/index.html

## Optimization Technologies

### Flash Attention
- **Flash Attention Paper**: https://arxiv.org/abs/2205.14135
- **Flash Attention v2**: https://arxiv.org/abs/2307.08691
- **Flash Attention GitHub**: https://github.com/Dao-AILab/flash-attention
- **xFormers Flash Attention**: https://github.com/facebookresearch/xformers

### FlexAttention (PyTorch)
- **FlexAttention Documentation**: https://pytorch.org/blog/flexattention/
- **FlexAttention Tutorial**: https://pytorch.org/tutorials/intermediate/flex_attention_tutorial.html
- **FlexAttention API**: https://pytorch.org/docs/main/generated/torch.nn.attention.flex_attention.html

### Memory Optimization
- **Gradient Checkpointing**: https://pytorch.org/docs/stable/checkpoint.html
- **Mixed Precision Training**: https://pytorch.org/docs/stable/amp.html
- **Memory Format Optimization**: https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html

## Distributed Training

### FSDP (Fully Sharded Data Parallel)
- **FSDP Documentation**: https://pytorch.org/docs/stable/fsdp.html
- **FSDP Tutorial**: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- **FSDP Advanced Features**: https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html

### DDP (Distributed Data Parallel)
- **DDP Documentation**: https://pytorch.org/docs/stable/distributed.html
- **DDP Tutorial**: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- **Multi-GPU Training**: https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html

### Communication Backends
- **NCCL**: https://developer.nvidia.com/nccl
- **Gloo**: https://github.com/facebookincubator/gloo
- **MPI Integration**: https://pytorch.org/docs/stable/distributed.html#mpi-backend

## Hardware Platforms

### NVIDIA GPUs
- **CUDA Compute Capabilities**: https://developer.nvidia.com/cuda-gpus
- **Tensor Core Programming**: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/
- **Ampere Architecture**: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
- **Hopper Architecture**: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

### AMD GPUs
- **ROCm Documentation**: https://rocmdocs.amd.com/
- **HIP Programming Guide**: https://rocmdocs.amd.com/projects/HIP/en/latest/
- **PyTorch ROCm**: https://pytorch.org/get-started/locally/#linux-via-conda
- **AMD MI Series**: https://www.amd.com/en/graphics/instinct-server-accelerators

### Intel Hardware
- **Intel Extension for PyTorch**: https://intel.github.io/intel-extension-for-pytorch/
- **Intel oneAPI**: https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html
- **Intel XPU**: https://intel.github.io/intel-extension-for-pytorch/xpu/latest/

## Cloud Platforms

### AWS
- **EC2 GPU Instances**: https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing
- **SageMaker**: https://aws.amazon.com/sagemaker/
- **EKS with GPU**: https://docs.aws.amazon.com/eks/latest/userguide/gpu-ami.html
- **AWS Deep Learning AMI**: https://aws.amazon.com/machine-learning/amis/

### Google Cloud Platform
- **GCP GPUs**: https://cloud.google.com/compute/docs/gpus
- **TPU Documentation**: https://cloud.google.com/tpu/docs
- **Vertex AI**: https://cloud.google.com/vertex-ai
- **AI Platform**: https://cloud.google.com/ai-platform

### Azure
- **Azure GPU VMs**: https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-gpu
- **Azure Machine Learning**: https://azure.microsoft.com/en-us/services/machine-learning/
- **Azure Kubernetes Service**: https://docs.microsoft.com/en-us/azure/aks/

## Benchmarking and Profiling

### Performance Tools
- **PyTorch Profiler**: https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
- **NVIDIA Nsight**: https://developer.nvidia.com/nsight-systems
- **AMD ROCProfiler**: https://rocmdocs.amd.com/projects/rocprofiler/en/latest/
- **Intel VTune**: https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html

### Benchmarking Frameworks
- **TorchBench**: https://github.com/pytorch/benchmark
- **MLPerf**: https://mlcommons.org/en/training/
- **DeepSpeed**: https://www.deepspeed.ai/
- **FairScale**: https://github.com/facebookresearch/fairscale

## Research Papers

### Attention Mechanisms
- **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
- **Flash Attention**: https://arxiv.org/abs/2205.14135
- **Flash Attention v2**: https://arxiv.org/abs/2307.08691
- **Ring Attention**: https://arxiv.org/abs/2310.01889
- **Mamba**: https://arxiv.org/abs/2312.00752

### Optimization Techniques
- **Gradient Checkpointing**: https://arxiv.org/abs/1604.06174
- **Mixed Precision Training**: https://arxiv.org/abs/1710.03740
- **ZeRO**: https://arxiv.org/abs/1910.02054
- **FP8 Training**: https://arxiv.org/abs/2209.05433

### Hardware Optimization
- **Tensor Core Utilization**: https://arxiv.org/abs/1805.09934
- **Memory-Efficient Attention**: https://arxiv.org/abs/2112.05682
- **Structured Sparsity**: https://arxiv.org/abs/2203.08757

## Development Tools

### Code Quality
- **Black Formatter**: https://black.readthedocs.io/
- **flake8**: https://flake8.pycqa.org/
- **mypy**: https://mypy.readthedocs.io/
- **pytest**: https://docs.pytest.org/

### Documentation
- **Sphinx**: https://www.sphinx-doc.org/
- **MkDocs**: https://www.mkdocs.org/
- **Jupyter Book**: https://jupyterbook.org/

### Version Control
- **Git Best Practices**: https://git-scm.com/doc
- **GitHub Actions**: https://docs.github.com/en/actions
- **Pre-commit Hooks**: https://pre-commit.com/

## Standards and Specifications

### IEEE Standards
- **IEEE 754 (Floating Point)**: https://ieeexplore.ieee.org/document/8766229
- **IEEE 1394 (Performance)**: https://ieeexplore.ieee.org/document/8945331

### CUDA Standards
- **CUDA Runtime API**: https://docs.nvidia.com/cuda/cuda-runtime-api/
- **CUDA Driver API**: https://docs.nvidia.com/cuda/cuda-driver-api/
- **PTX Instruction Set**: https://docs.nvidia.com/cuda/parallel-thread-execution/

### OpenCL
- **OpenCL Specification**: https://www.khronos.org/opencl/
- **PyOpenCL**: https://documen.tician.de/pyopencl/

## Community Resources

### Forums and Discussion
- **PyTorch Forums**: https://discuss.pytorch.org/
- **NVIDIA Developer Forums**: https://forums.developer.nvidia.com/
- **AMD Developer Community**: https://community.amd.com/
- **Stack Overflow**: https://stackoverflow.com/questions/tagged/pytorch

### Conferences
- **NeurIPS**: https://neurips.cc/
- **ICML**: https://icml.cc/
- **ICLR**: https://iclr.cc/
- **MLSys**: https://mlsys.org/

### Blogs and News
- **PyTorch Blog**: https://pytorch.org/blog/
- **NVIDIA AI Blog**: https://blogs.nvidia.com/blog/category/deep-learning/
- **Google AI Blog**: https://ai.googleblog.com/
- **Towards Data Science**: https://towardsdatascience.com/

---

## Quick Reference Links

### Essential Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton Tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html)

### Setup Guides
- [CUDA Setup](cuda_setup.md)
- [Cloud Testing](cloud_testing_guide.md)
- [Hardware Abstraction](hardware.md)

### Performance Resources
- [PyTorch Profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

**🚀 All essential references for PyTorch GPU optimization development!**