# End-to-End Deployment Tutorial

**KernelPyTorch v0.4.3** - Complete guide from model optimization to production deployment

This tutorial walks through deploying an optimized PyTorch model from development to production, covering all stages of the deployment pipeline.

---

## Overview

This tutorial covers:
1. Model Optimization with KernelPyTorch
2. Model Export (ONNX, TorchScript)
3. Inference Server Setup (FastAPI, TorchServe)
4. Containerization with Docker
5. Cloud Deployment (AWS, GCP)
6. Monitoring and Observability

**Prerequisites**:
- Python 3.8+
- PyTorch 2.0+
- KernelPyTorch installed (`pip install -e .`)
- Docker (for containerization)

---

## Step 1: Model Optimization

### 1.1 Basic Model Optimization

```python
import torch
import torch.nn as nn
from kernel_pytorch import KernelPyTorchConfig, UnifiedManager

# Define your model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)

# Create and optimize model
model = TransformerModel()

# Configure for production inference
config = KernelPyTorchConfig.for_inference()
manager = UnifiedManager(config)

# Apply optimizations
optimized_model = manager.optimize(model)
print(f"Model optimized for {config.device}")
```

### 1.2 Validate Optimization

```python
from kernel_pytorch.validation import UnifiedValidator

# Validate optimized model
validator = UnifiedValidator()
results = validator.validate_model(
    optimized_model,
    input_shape=(1, 128),  # (batch, seq_len)
    input_dtype=torch.long
)

print(f"Validation: {results.passed}/{results.total_tests} tests passed")
assert results.passed == results.total_tests, "Validation failed!"
```

### 1.3 Benchmark Performance

```python
import time

# Warmup
sample_input = torch.randint(0, 10000, (1, 128))
for _ in range(10):
    with torch.no_grad():
        _ = optimized_model(sample_input)

# Benchmark
iterations = 100
start = time.perf_counter()
for _ in range(iterations):
    with torch.no_grad():
        _ = optimized_model(sample_input)
elapsed = time.perf_counter() - start

print(f"Average latency: {elapsed/iterations*1000:.2f} ms")
print(f"Throughput: {iterations/elapsed:.1f} inferences/sec")
```

---

## Step 2: Model Export

### 2.1 Export to TorchScript

```python
from kernel_pytorch.deployment import TorchScriptExporter

# Export to TorchScript
exporter = TorchScriptExporter()
sample_input = torch.randint(0, 10000, (1, 128))

# Trace the model
result = exporter.export(
    optimized_model,
    sample_inputs=sample_input,
    output_path="model_traced.pt",
    method="trace"  # or "script" for dynamic control flow
)

print(f"TorchScript model saved: {result.output_path}")
print(f"Model size: {result.model_size_mb:.2f} MB")

# Verify exported model
loaded_model = torch.jit.load("model_traced.pt")
with torch.no_grad():
    output = loaded_model(sample_input)
print(f"Output shape: {output.shape}")
```

### 2.2 Export to ONNX

```python
from kernel_pytorch.deployment import ONNXExporter

# Export to ONNX with dynamic axes
onnx_exporter = ONNXExporter()
result = onnx_exporter.export(
    optimized_model,
    sample_inputs=sample_input,
    output_path="model.onnx",
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size", 1: "seq_len"}
    },
    opset_version=17
)

print(f"ONNX model saved: {result.output_path}")

# Validate with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
onnx_output = session.run(None, {"input": sample_input.numpy()})
print(f"ONNX inference successful, output shape: {onnx_output[0].shape}")
```

---

## Step 3: Inference Server Setup

### 3.1 FastAPI Server

Create `server.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
from typing import List

app = FastAPI(title="KernelPyTorch Model Server", version="0.4.3")

# Load optimized model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = torch.jit.load("model_traced.pt")
    model.eval()
    print("Model loaded successfully")

class PredictionRequest(BaseModel):
    input_ids: List[List[int]]

class PredictionResponse(BaseModel):
    logits: List[List[List[float]]]
    latency_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    import time

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_tensor = torch.tensor(request.input_ids, dtype=torch.long)

        start = time.perf_counter()
        with torch.no_grad():
            output = model(input_tensor)
        latency = (time.perf_counter() - start) * 1000

        return PredictionResponse(
            logits=output.tolist(),
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run the server:
```bash
pip install fastapi uvicorn
python server.py
```

Test the endpoint:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input_ids": [[1, 2, 3, 4, 5]]}'
```

### 3.2 TorchServe Deployment

Create model archive:
```bash
# Install TorchServe
pip install torchserve torch-model-archiver

# Create handler (handler.py)
cat > handler.py << 'EOF'
import torch
from ts.torch_handler.base_handler import BaseHandler

class TransformerHandler(BaseHandler):
    def initialize(self, context):
        self.model = torch.jit.load(context.model_dir + "/model_traced.pt")
        self.model.eval()

    def preprocess(self, data):
        inputs = [d.get("data") or d.get("body") for d in data]
        return torch.tensor(inputs, dtype=torch.long)

    def inference(self, inputs):
        with torch.no_grad():
            return self.model(inputs)

    def postprocess(self, outputs):
        return outputs.tolist()
EOF

# Create model archive
torch-model-archiver \
  --model-name transformer \
  --version 1.0 \
  --serialized-file model_traced.pt \
  --handler handler.py \
  --export-path model_store

# Start TorchServe
torchserve --start --model-store model_store --models transformer=transformer.mar
```

---

## Step 4: Containerization

### 4.1 Production Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and server code
COPY model_traced.pt .
COPY server.py .

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "server.py"]
```

Create `requirements.txt`:
```
torch>=2.0.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
```

Build and run:
```bash
# Build image
docker build -t kernelpytorch-model:v0.4.3 .

# Run container
docker run -p 8000:8000 kernelpytorch-model:v0.4.3

# Test
curl http://localhost:8000/health
```

### 4.2 GPU-Enabled Dockerfile

```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements-gpu.txt .
RUN pip3 install --no-cache-dir -r requirements-gpu.txt

# Copy model and server
COPY model_traced.pt .
COPY server.py .

EXPOSE 8000

CMD ["python3", "server.py"]
```

Run with GPU:
```bash
docker run --gpus all -p 8000:8000 kernelpytorch-model:v0.4.3-gpu
```

---

## Step 5: Cloud Deployment

### 5.1 AWS Deployment (ECS)

Create task definition `ecs-task.json`:
```json
{
  "family": "kernelpytorch-inference",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "inference",
      "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/kernelpytorch-model:v0.4.3",
      "portMappings": [
        {"containerPort": 8000, "protocol": "tcp"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/kernelpytorch",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "inference"
        }
      }
    }
  ]
}
```

Deploy:
```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker tag kernelpytorch-model:v0.4.3 ACCOUNT.dkr.ecr.REGION.amazonaws.com/kernelpytorch-model:v0.4.3
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/kernelpytorch-model:v0.4.3

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task.json

# Create service
aws ecs create-service \
  --cluster my-cluster \
  --service-name kernelpytorch-service \
  --task-definition kernelpytorch-inference \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### 5.2 GCP Deployment (Cloud Run)

```bash
# Build and push to Artifact Registry
gcloud builds submit --tag gcr.io/PROJECT/kernelpytorch-model:v0.4.3

# Deploy to Cloud Run
gcloud run deploy kernelpytorch-service \
  --image gcr.io/PROJECT/kernelpytorch-model:v0.4.3 \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --port 8000 \
  --allow-unauthenticated \
  --min-instances 1 \
  --max-instances 10
```

### 5.3 Kubernetes Deployment

Create `k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kernelpytorch-inference
  labels:
    app: kernelpytorch
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kernelpytorch
  template:
    metadata:
      labels:
        app: kernelpytorch
    spec:
      containers:
      - name: inference
        image: kernelpytorch-model:v0.4.3
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: kernelpytorch-service
spec:
  selector:
    app: kernelpytorch
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods -l app=kernelpytorch
kubectl get svc kernelpytorch-service
```

---

## Step 6: Monitoring and Observability

### 6.1 Add Prometheus Metrics

Update `server.py`:
```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse

# Metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total inference requests')
REQUEST_LATENCY = Histogram('inference_latency_seconds', 'Inference latency')

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    REQUEST_COUNT.inc()

    with REQUEST_LATENCY.time():
        # ... inference code ...
        pass

    return response

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain")
```

### 6.2 Logging Configuration

```python
import logging
import json
from datetime import datetime

# Structured logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module
        })

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# Usage in endpoints
@app.post("/predict")
async def predict(request: PredictionRequest):
    logger.info(f"Inference request: batch_size={len(request.input_ids)}")
    # ... inference ...
    logger.info(f"Inference complete: latency_ms={latency:.2f}")
```

### 6.3 Health Check Endpoints

```python
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model": "loaded"}

@app.get("/live")
async def live():
    return {"status": "alive"}
```

---

## Summary Checklist

- [ ] Model optimized with KernelPyTorch
- [ ] Validation tests passed
- [ ] Performance benchmarked
- [ ] Model exported (TorchScript/ONNX)
- [ ] Inference server implemented
- [ ] Docker container built and tested
- [ ] Health checks configured
- [ ] Deployed to cloud platform
- [ ] Monitoring and logging enabled

---

## Next Steps

- [Docker Guide](docker.md) - Detailed container configuration
- [Cloud Testing Guide](cloud_testing_guide.md) - Cloud platform validation
- [Backend Selection](backend_selection.md) - Choose optimal backend for your hardware

---

**Version**: 0.4.3
**Last Updated**: January 18, 2026
