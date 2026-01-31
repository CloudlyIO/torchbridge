# Deployment Guide

Export, serve, and containerize TorchBridge models for production.

## Model Export

### TorchScript

```python
from torchbridge.deployment import export_to_torchscript

sample_input = torch.randn(1, 768)
result = export_to_torchscript(model, output_path="model.pt", sample_input=sample_input)
```

### ONNX

```python
from torchbridge.deployment import export_to_onnx

result = export_to_onnx(model, output_path="model.onnx", sample_input=sample_input, opset_version=17)
```

### SafeTensors (weights only)

```python
from torchbridge.deployment import export_to_safetensors

result = export_to_safetensors(model, output_path="model.safetensors")
```

### Export Validation

Always validate exports:

```python
# Load and verify TorchScript
ts_model = torch.jit.load("model.pt")
ts_output = ts_model(sample_input)

# Compare with original
orig_output = model(sample_input)
assert torch.allclose(ts_output, orig_output, atol=1e-5)
```

## Inference Server

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
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

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

### Prometheus Metrics

```python
from torchbridge.monitoring import MetricsExporter, start_metrics_server

metrics = MetricsExporter(model_name="my_model")
start_metrics_server(metrics, port=9090)

# Record inference metrics
metrics.record_inference(latency_ms=15.3, batch_size=8)
```

### Health Checks

```python
from torchbridge.monitoring import create_enhanced_health_monitor

health = create_enhanced_health_monitor()
status = health.check_health()
```

### Logging

```python
from torchbridge.monitoring import configure_logging, get_logger

configure_logging(json_format=True, level="INFO")
logger = get_logger(__name__)
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

## See Also

- [CLI Reference](cli.md)
- [Backends Overview](../backends/overview.md)
- [Distributed Training](distributed-training.md)
