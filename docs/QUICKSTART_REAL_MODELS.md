# TorchBridge: Real ML Model Workflow

> Complete guide for developing, testing, deploying, and running real ML models with TorchBridge v0.4.34

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Development Workflow](#2-development-workflow)
3. [Testing & Validation](#3-testing--validation)
4. [Production Deployment](#4-production-deployment)
5. [Monitoring & Operations](#5-monitoring--operations)
6. [Examples by Model Type](#6-examples-by-model-type)

---

## 1. Environment Setup

### 1.1 Installation

```bash
# Basic installation
pip install torchbridge

# With all backends (CUDA, ROCm, XPU, TPU)
pip install torchbridge[all]

# For development
pip install torchbridge[dev]
```

### 1.2 Verify Installation

```python
import torchbridge as kpt
from torchbridge.hardware import get_optimal_backend, create_backend

# Check version
print(f"Version: {kpt.__version__}")  # 0.4.34

# Detect hardware
backend_name = get_optimal_backend()
print(f"Optimal backend: {backend_name}")

# Create backend
backend = create_backend(backend_name)
print(f"Device info: {backend.get_device_info()}")
```

### 1.3 Project Structure

```
my_ml_project/
├── models/               # Model definitions
│   ├── __init__.py
│   └── transformer.py
├── training/             # Training scripts
│   ├── train.py
│   └── config.yaml
├── inference/            # Inference code
│   ├── serve.py
│   └── batch_predict.py
├── tests/                # Tests
│   ├── test_model.py
│   └── test_inference.py
├── configs/              # Configurations
│   ├── dev.yaml
│   └── prod.yaml
└── requirements.txt
```

---

## 2. Development Workflow

### 2.1 Define Your Model

```python
# models/transformer.py
import torch
import torch.nn as nn
from torchbridge.attention import UnifiedAttentionFusion
from torchbridge.precision import MixedPrecisionConfig

class OptimizedTransformer(nn.Module):
    """Transformer with TorchBridge optimizations."""

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        backend: str = "auto"
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Use optimized attention
        self.layers = nn.ModuleList([
            UnifiedAttentionFusion(
                embed_dim=d_model,
                num_heads=n_heads,
                backend=backend
            )
            for _ in range(n_layers)
        ])

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, x, x)
        return self.output(x)
```

### 2.2 Optimize for Training

```python
# training/train.py
import torch
from torchbridge.hardware import get_optimal_backend, create_backend
from torchbridge.precision import MixedPrecisionTrainer, MixedPrecisionConfig
from torchbridge.memory import MemoryOptimizer
from torchbridge.monitoring import configure_logging, get_logger, create_slo_manager

# Setup logging
configure_logging(json_format=True, level="INFO")
logger = get_logger(__name__)

def train_model(model, train_loader, config):
    """Train with TorchBridge optimizations."""

    # 1. Auto-detect optimal backend
    backend_name = get_optimal_backend()
    backend = create_backend(backend_name)
    device = backend.device
    logger.info(f"Using backend: {backend_name}", extra={"device": str(device)})

    # 2. Setup mixed precision
    precision_config = MixedPrecisionConfig(
        enabled=True,
        dtype=torch.bfloat16 if backend.supports_bfloat16() else torch.float16,
        loss_scale="dynamic"
    )
    trainer = MixedPrecisionTrainer(model, precision_config)

    # 3. Optimize memory
    memory_optimizer = MemoryOptimizer(
        gradient_checkpointing=True,
        activation_offloading=config.get("offload_activations", False)
    )
    model = memory_optimizer.optimize(model)

    # 4. Move to device
    model = model.to(device)

    # 5. Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # 6. Training loop
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward with mixed precision
            with trainer.autocast():
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # Backward with gradient scaling
            trainer.backward(loss, optimizer)
            trainer.step(optimizer)
            optimizer.zero_grad()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}",
                    extra={"loss": loss.item(), "epoch": epoch}
                )

        logger.info(f"Epoch {epoch} complete", extra={"avg_loss": total_loss / len(train_loader)})

    return model
```

### 2.3 Configuration Files

```yaml
# configs/dev.yaml
model:
  vocab_size: 50000
  d_model: 512
  n_heads: 8
  n_layers: 6

training:
  epochs: 10
  lr: 1e-4
  batch_size: 32
  gradient_checkpointing: true
  offload_activations: false

precision:
  enabled: true
  dtype: bfloat16
  loss_scale: dynamic

monitoring:
  log_level: DEBUG
  metrics_port: 9090
  slo_latency_p99_ms: 100
```

```yaml
# configs/prod.yaml
model:
  vocab_size: 50000
  d_model: 1024
  n_heads: 16
  n_layers: 24

training:
  epochs: 100
  lr: 3e-4
  batch_size: 64
  gradient_checkpointing: true
  offload_activations: true

precision:
  enabled: true
  dtype: bfloat16
  loss_scale: dynamic

monitoring:
  log_level: INFO
  metrics_port: 9090
  slo_latency_p99_ms: 50
  alert_on_regression: true
```

---

## 3. Testing & Validation

### 3.1 Unit Tests

```python
# tests/test_model.py
import pytest
import torch
from models.transformer import OptimizedTransformer

class TestOptimizedTransformer:
    """Tests for the optimized transformer."""

    @pytest.fixture
    def model(self):
        return OptimizedTransformer(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2
        )

    def test_forward_pass(self, model):
        """Test basic forward pass."""
        x = torch.randint(0, 1000, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, 1000)

    def test_backward_pass(self, model):
        """Test gradient computation."""
        x = torch.randint(0, 1000, (2, 32))
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check gradients exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_mixed_precision(self, model):
        """Test with mixed precision."""
        from torchbridge.precision import MixedPrecisionConfig, MixedPrecisionTrainer

        config = MixedPrecisionConfig(enabled=True, dtype=torch.float16)
        trainer = MixedPrecisionTrainer(model, config)

        x = torch.randint(0, 1000, (2, 32))
        with trainer.autocast():
            out = model(x)

        assert out.shape == (2, 32, 1000)

# Run tests
# pytest tests/test_model.py -v
```

### 3.2 Performance Tests

```python
# tests/test_performance.py
import pytest
import torch
import time
from torchbridge.monitoring import get_performance_tracker

class TestPerformance:
    """Performance regression tests."""

    @pytest.fixture
    def model_and_input(self):
        from models.transformer import OptimizedTransformer
        model = OptimizedTransformer(d_model=512, n_heads=8, n_layers=6)
        model.eval()
        x = torch.randint(0, 50000, (8, 128))
        return model, x

    def test_inference_latency(self, model_and_input):
        """Test inference meets latency SLO."""
        model, x = model_and_input

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                model(x)

        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                model(x)
            times.append((time.perf_counter() - start) * 1000)

        p99 = sorted(times)[int(len(times) * 0.99)]
        assert p99 < 200, f"P99 latency {p99:.1f}ms exceeds 200ms SLO"

    def test_no_regression(self, model_and_input):
        """Test no performance regression from baseline."""
        model, x = model_and_input
        tracker = get_performance_tracker()

        metrics = tracker.record_performance(model, x, "transformer_v1")
        regression = tracker.detect_regression(model, metrics)

        assert not regression, f"Performance regression detected: {regression}"
```

### 3.3 Cloud Validation (Cost-Optimized)

```bash
# Run free-tier validation
python scripts/validation/cost_optimized_validation.py --tier quick

# This generates:
# - Google Colab notebook
# - Kaggle notebook
# - Intel DevCloud script
# Total cost: $0
```

---

## 4. Production Deployment

### 4.1 Export Model

```python
# inference/export.py
import torch
from torchbridge.deployment import ModelExporter, ExportConfig

def export_model(model, output_dir):
    """Export model for production deployment."""

    exporter = ModelExporter()

    # Sample input for tracing
    sample_input = torch.randint(0, 50000, (1, 128))

    # Export to TorchScript
    ts_path = exporter.export_torchscript(
        model,
        sample_input,
        output_dir / "model.pt"
    )
    print(f"TorchScript: {ts_path}")

    # Export to ONNX
    onnx_path = exporter.export_onnx(
        model,
        sample_input,
        output_dir / "model.onnx",
        opset_version=17
    )
    print(f"ONNX: {onnx_path}")

    # Export to SafeTensors (weights only)
    st_path = exporter.export_safetensors(
        model,
        output_dir / "model.safetensors"
    )
    print(f"SafeTensors: {st_path}")

    return ts_path, onnx_path, st_path
```

### 4.2 Inference Server

```python
# inference/serve.py
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from torchbridge.monitoring import (
    MetricsExporter,
    start_metrics_server,
    create_enhanced_health_monitor,
    configure_logging,
    get_logger,
    correlation_context
)

# Setup
configure_logging(json_format=True)
logger = get_logger(__name__)
app = FastAPI(title="ML Model API")

# Load model
model = torch.jit.load("models/model.pt")
model.eval()

# Setup monitoring
metrics = MetricsExporter(model_name="transformer")
health_monitor = create_enhanced_health_monitor()
start_metrics_server(metrics, port=9090)

class PredictRequest(BaseModel):
    input_ids: list[int]

class PredictResponse(BaseModel):
    predictions: list[float]
    latency_ms: float

@app.get("/health")
async def health():
    """Health check endpoint."""
    status = health_monitor.check_health()
    return {"status": status.value, "checks": health_monitor.get_all_checks()}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Model inference endpoint."""
    import time

    with correlation_context():
        start = time.perf_counter()

        # Prepare input
        input_tensor = torch.tensor([request.input_ids])

        # Inference
        with torch.no_grad():
            output = model(input_tensor)

        latency_ms = (time.perf_counter() - start) * 1000

        # Record metrics
        metrics.record_inference(
            latency_ms=latency_ms,
            batch_size=1
        )

        logger.info(
            "Inference complete",
            extra={"latency_ms": latency_ms}
        )

        return PredictResponse(
            predictions=output[0].tolist(),
            latency_ms=latency_ms
        )

# Run: uvicorn inference.serve:app --host 0.0.0.0 --port 8000
```

### 4.3 Docker Deployment

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and code
COPY models/ models/
COPY inference/ inference/

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "inference.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yaml
version: '3.8'
services:
  model-server:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - LOG_LEVEL=INFO
      - METRICS_PORT=9090
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards
```

---

## 5. Monitoring & Operations

### 5.1 Setup Monitoring

```python
# monitoring/setup.py
from torchbridge.monitoring import (
    configure_logging,
    MetricsExporter,
    start_metrics_server,
    create_slo_manager,
    create_enhanced_health_monitor,
    create_default_alert_rules,
    export_alert_rules_json,
    SLOConfig, SLIType
)

def setup_monitoring(model_name: str, config: dict):
    """Setup complete monitoring stack."""

    # 1. Logging
    configure_logging(
        json_format=True,
        level=config.get("log_level", "INFO")
    )

    # 2. Metrics
    metrics = MetricsExporter(model_name=model_name)
    start_metrics_server(metrics, port=config.get("metrics_port", 9090))

    # 3. Health monitoring
    health = create_enhanced_health_monitor()

    # 4. SLO tracking
    slos = [
        SLOConfig(
            name="latency_p99",
            sli_type=SLIType.LATENCY,
            target=0.99,
            threshold=config.get("slo_latency_p99_ms", 100),
            window_seconds=3600
        ),
        SLOConfig(
            name="availability",
            sli_type=SLIType.AVAILABILITY,
            target=0.999,
            window_seconds=86400
        ),
        SLOConfig(
            name="error_rate",
            sli_type=SLIType.ERROR_RATE,
            target=0.99,
            threshold=0.01,
            window_seconds=3600
        )
    ]
    slo_manager = create_slo_manager(slos)

    # 5. Alert rules
    alert_rules = create_default_alert_rules(
        latency_p99_threshold=config.get("slo_latency_p99_ms", 100)
    )
    export_alert_rules_json(alert_rules, "grafana-alerts.json")

    return {
        "metrics": metrics,
        "health": health,
        "slo_manager": slo_manager
    }
```

### 5.2 Prometheus Config

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'model-server'
    static_configs:
      - targets: ['model-server:9090']
```

### 5.3 Grafana Dashboard

```python
# Generate Grafana dashboard
from torchbridge.monitoring import create_full_dashboard, export_dashboard_json

dashboard = create_full_dashboard(model_name="transformer")
export_dashboard_json(dashboard, "grafana-dashboards/model-dashboard.json")
```

---

## 6. Examples by Model Type

### 6.1 Text Generation (GPT-style)

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchbridge.models.text import TextModelOptimizer, TextModelConfig

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Optimize
config = TextModelConfig(
    optimization_mode="inference",
    use_torch_compile=True
)
optimizer = TextModelOptimizer(config)
optimized_model = optimizer.optimize(model, task="causal-lm")

# Generate
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = optimized_model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 6.2 Image Classification (ResNet)

```python
from torchvision.models import resnet50, ResNet50_Weights
from torchbridge.models.vision import ResNetOptimizer, VisionOptimizationConfig

# Load model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Optimize
config = VisionOptimizationConfig(
    optimization_level="O2",
    channels_last=True,
    compile_model=True
)
optimizer = ResNetOptimizer(config)
optimized_model = optimizer.optimize(model)

# Inference
images = torch.randn(4, 3, 224, 224).to(memory_format=torch.channels_last)
with torch.no_grad():
    predictions = optimized_model(images)
```

### 6.3 Large Language Model (7B+)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchbridge.precision import FP8TrainingEngine
from torchbridge.memory import MemoryOptimizer

# Load with memory optimization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Apply gradient checkpointing
memory_opt = MemoryOptimizer(gradient_checkpointing=True)
model = memory_opt.optimize(model)

# For H100: Use FP8 training
if torch.cuda.get_device_capability()[0] >= 9:
    fp8_engine = FP8TrainingEngine(model)
    # Training loop uses fp8_engine.forward() and fp8_engine.backward()
```

### 6.4 Mixture of Experts

```python
from torchbridge.moe import MixtureOfExperts, MoEConfig

# Create MoE layer
moe_config = MoEConfig(
    num_experts=8,
    top_k=2,
    expert_dim=2048,
    input_dim=512
)
moe_layer = MixtureOfExperts(moe_config)

# Use in model
class MoETransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = UnifiedAttentionFusion(512, 8)
        self.moe = MixtureOfExperts(moe_config)

    def forward(self, x):
        x = self.attention(x, x, x)
        x = self.moe(x)
        return x
```

---

## Quick Reference

### CLI Commands

```bash
# Optimize model
torchbridge optimize model.pt --output optimized.pt --level O2

# Export model
torchbridge export model.pt --format onnx --output model.onnx

# Profile model
torchbridge profile model.pt --input-shape 1,128 --output profile.json

# Benchmark
torchbridge benchmark model.pt --batch-sizes 1,8,32 --output benchmark.json
```

### Environment Variables

```bash
# Backend selection
export KERNELPYTORCH_BACKEND=cuda  # cuda, rocm, xpu, tpu, cpu

# Logging
export KERNELPYTORCH_LOG_LEVEL=INFO
export KERNELPYTORCH_LOG_FORMAT=json

# Precision
export KERNELPYTORCH_DEFAULT_DTYPE=bfloat16
```

---

## Support

- Documentation: https://torchbridge.readthedocs.io
- Issues: https://github.com/torchbridge/torchbridge/issues
- Discussions: https://github.com/torchbridge/torchbridge/discussions
