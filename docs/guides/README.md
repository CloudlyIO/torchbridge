# User Guides

> **Version**: v0.4.18 | Comprehensive guides for deploying and optimizing models with KernelPyTorch

## Deployment & Operations

| Guide | Description | Time |
|-------|-------------|------|
| [Deployment Tutorial](deployment_tutorial.md) | End-to-end production pipeline | 30 min |
| [Testing Guide](testing_guide.md) | Testing, benchmarking, validation | 20 min |
| [Docker Guide](docker.md) | Containerization for deployment | 15 min |
| [Backend Selection](backend_selection.md) | Choose the right hardware backend | 10 min |

## Model Optimization Guides

| Guide | Models Covered | Speedup |
|-------|----------------|---------|
| [Small Models](small_model_guide.md) | BERT, GPT-2, DistilBERT (<7B params) | 2-3x |
| [Vision Models](vision_model_guide.md) | ResNet, ViT, Stable Diffusion | 2-5x |

## Quick Example

```python
from kernel_pytorch import optimize_model, get_config

# Configure for your hardware
config = get_config()
config.device = "cuda"

# Optimize any PyTorch model
optimized_model = optimize_model(your_model)
```

## Learning Path

1. **Start here**: [Installation](../getting-started/installation.md) → [Quickstart](../getting-started/quickstart.md)
2. **Choose backend**: [Backend Selection](backend_selection.md)
3. **Optimize models**: Small Models or Vision Models guide
4. **Deploy**: [Deployment Tutorial](deployment_tutorial.md)

---

**See Also**: [Getting Started](../getting-started/README.md) | [Backends](../backends/README.md) | [Cloud Deployment](../cloud-deployment/README.md)
