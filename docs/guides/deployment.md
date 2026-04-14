# Deployment Guide

Serve and containerize your models for production. For model export, use PyTorch's
native APIs directly — TorchBridge does not wrap them.

## Model Export

TorchBridge does not provide export wrappers. Use PyTorch's native APIs:

### TorchScript

```python
# Trace (preferred for most models)
traced = torch.jit.trace(model, sample_input)
traced.save("model.pt")

# Verify
loaded = torch.jit.load("model.pt")
assert torch.allclose(loaded(sample_input), model(sample_input), atol=1e-5)
```

### ONNX

```python
torch.onnx.export(
    model,
    sample_input,
    "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
)
```

### SafeTensors

```python
from safetensors.torch import save_file

save_file(model.state_dict(), "model.safetensors")
```

## Inference Server

> **Note:** TorchBridge does not provide a serving runtime. The patterns below show how to build a serving layer around a model that TorchBridge has validated. For production serving, use vLLM, TGI, or TorchServe directly.

### FastAPI

```python
import torch
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TorchBridge Model API")
model = torch.jit.load("model.pt")
model.eval()

class PredictRequest(BaseModel):
    input_ids: list[int]

class PredictResponse(BaseModel):
    predictions: list[float]
    latency_ms: float

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    import time
    start = time.perf_counter()

    input_tensor = torch.tensor([request.input_ids])
    with torch.no_grad():
        output = model(input_tensor)

    latency_ms = (time.perf_counter() - start) * 1000
    return PredictResponse(
        predictions=output[0].tolist(),
        latency_ms=latency_ms,
    )

# Run: uvicorn serve:app --host 0.0.0.0 --port 8000
```

### TorchServe

```bash
# Archive model
torch-model-archiver --model-name my_model \
    --version 1.0 \
    --serialized-file model.pt \
    --handler handler.py

# Start server
torchserve --start --model-store model_store --models my_model.mar
```

## Docker

### Production Image

```dockerfile
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY models/ models/
COPY inference/ inference/

EXPOSE 8000 9090

HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "inference.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
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
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### AMD GPU Container

```dockerfile
FROM rocm/pytorch:latest

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker run --device=/dev/kfd --device=/dev/dri --group-add video my-model
```

## Cloud Deployment

### AWS ECS

```bash
# Build and push
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URL
docker build -t model-server .
docker tag model-server:latest $ECR_URL/model-server:latest
docker push $ECR_URL/model-server:latest

# Deploy to ECS with GPU
aws ecs create-service --cluster ml-cluster \
    --service-name model-server \
    --task-definition model-server:1 \
    --desired-count 2
```

### GCP Cloud Run

```bash
gcloud run deploy model-server \
    --image gcr.io/$PROJECT/model-server \
    --gpu 1 --gpu-type nvidia-l4 \
    --memory 16Gi --cpu 4 \
    --port 8000
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: model-server
        image: model-server:latest
        ports:
        - containerPort: 8000
        - containerPort: 9090
        resources:
          limits:
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Monitoring

Use Python's standard `logging` module and your preferred metrics stack (Prometheus,
Datadog, etc.) directly — TorchBridge does not provide monitoring wrappers.

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.info("Inference complete", extra={"latency_ms": 15.3})
```

## Production Checklist

Before deploying:

- [ ] Model exported and validated (outputs match original)
- [ ] Health check endpoint working
- [ ] Resource limits set (CPU, memory, GPU)
- [ ] Metrics endpoint exposed
- [ ] Logging configured (JSON format for production)
- [ ] Container tested locally with GPU
- [ ] Load testing completed
- [ ] Rollback plan documented

## Privacy & Data Handling

TorchBridge collects **no telemetry** and makes **no network requests**. All data stays on your machine.

TorchBridge writes one file to your local filesystem during normal operation:

- **`~/.torchbridge/kernel_benchmarks.json`** — kernel latency cache, keyed by hardware fingerprint. Contains only timing data and hardware metadata — no model weights, inputs, outputs, or PII.

| Question | Answer |
|----------|--------|
| Does TorchBridge phone home? | No |
| Telemetry sent anywhere? | No |
| Safe for air-gapped environments? | Yes |
| Contains model data? | No — only kernel latency timings and hardware metadata |

The OTel exporter (`--otel` flag on `tb-validate`) sends validation spans to an OTLP endpoint
**only when explicitly configured**. No endpoint is configured by default.

```bash
# Remove all locally stored data
rm -rf ~/.torchbridge
```

If sharing diagnostic output for debugging, latency numbers, hardware model names, and
pass/fail results are safe to share. Avoid sharing model file paths or proprietary model
configurations.

## See Also

- [CLI Reference](cli.md)
- [Backends Overview](../backends/overview.md)
- [Distributed Training](distributed-training.md)
