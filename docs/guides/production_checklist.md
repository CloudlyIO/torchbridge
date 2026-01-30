# Production Deployment Checklist

> **Version**: v0.4.27 | **Status**: Current | **Last Updated**: 2026-01-26

Complete checklist for deploying TorchBridge models to production environments.

---

## Pre-Deployment Validation

### Model Validation

- [ ] **Forward pass verification** - Model produces expected output
  ```bash
  tb-doctor --full-report
  ```

- [ ] **Determinism check** - Same input produces same output
  ```python
  from torchbridge.deployment import validate_production_readiness
  result = validate_production_readiness(model, sample_input)
  print(result.summary())
  ```

- [ ] **Numerical accuracy** - Output matches baseline within tolerance
  ```python
  # Compare with reference implementation
  torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-3)
  ```

- [ ] **Edge cases tested** - Empty inputs, max sequence length, batch size 1

### Performance Validation

- [ ] **Latency benchmark**
  ```bash
  tb-benchmark --model model.pt --quick
  ```

- [ ] **Memory profiling**
  ```bash
  tb-profile --model model.pt --mode memory
  ```

- [ ] **Throughput test** - Meets SLA requirements
  ```bash
  tb-benchmark --model model.pt --stress --batch-sizes 1,4,8,16,32
  ```

- [ ] **GPU utilization** - Verify efficient GPU usage
  ```bash
  tb-profile --model model.pt --mode detailed --device cuda
  ```

---

## Export Checklist

### Format Selection

| Use Case | Recommended Format | Command |
|----------|-------------------|---------|
| Cross-platform inference | ONNX | `tb-export --format onnx` |
| PyTorch ecosystem | TorchScript | `tb-export --format torchscript` |
| Secure weight storage | SafeTensors | `tb-export --format safetensors` |
| Multiple formats | All | `tb-export --format all` |

### Export Validation

- [ ] **ONNX export**
  ```bash
  tb-export --model model.pt --format onnx --validate --output model.onnx
  ```

- [ ] **TorchScript export**
  ```bash
  tb-export --model model.pt --format torchscript --validate --method trace
  ```

- [ ] **Export file integrity** - Verify files can be loaded
  ```python
  import onnx
  model = onnx.load("model.onnx")
  onnx.checker.check_model(model)
  ```

- [ ] **Dynamic axes configured** (if variable batch/sequence)
  ```bash
  tb-export --model model.pt --format onnx --dynamic-axes
  ```

---

## Infrastructure Checklist

### Hardware Requirements

- [ ] **GPU memory** - Model + activations + overhead fit in VRAM
  ```python
  # Check model memory requirement
  from torchbridge import get_manager
  manager = get_manager()
  memory_estimate = manager.estimate_memory(model, batch_size=32)
  print(f"Estimated memory: {memory_estimate / 1024**3:.2f} GB")
  ```

- [ ] **CPU fallback** - Model works on CPU (for graceful degradation)
- [ ] **Multi-GPU support** - If needed, test distributed inference

### Container Setup

- [ ] **Base image selected**
  - CPU: `ghcr.io/torchbridge/torchbridge:latest-cpu`
  - NVIDIA: `ghcr.io/torchbridge/torchbridge:latest-nvidia`
  - Production: `ghcr.io/torchbridge/torchbridge:latest-production`

- [ ] **Dependencies pinned** - `requirements.txt` with exact versions

- [ ] **Health checks configured**
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1
  ```

- [ ] **Resource limits set**
  ```yaml
  resources:
    limits:
      nvidia.com/gpu: 1
      memory: 16Gi
    requests:
      memory: 8Gi
  ```

### Kubernetes (if applicable)

- [ ] **Deployment manifest** ready
- [ ] **HPA configured** for auto-scaling
- [ ] **PodDisruptionBudget** set
- [ ] **ServiceMonitor** for Prometheus metrics
- [ ] **Secrets management** configured

---

## Serving Configuration

### Endpoint Setup

- [ ] **Inference endpoint** `/predict` or `/generate`
- [ ] **Health endpoint** `/health` returns 200 when ready
- [ ] **Readiness endpoint** `/ready` indicates model loaded
- [ ] **Metrics endpoint** `/metrics` for Prometheus

### Request Handling

- [ ] **Input validation** - Reject malformed requests
- [ ] **Timeout configuration** - Prevent hanging requests
  ```python
  from torchbridge.deployment.serving import ServerConfig
  config = ServerConfig(
      timeout_seconds=30,
      max_batch_size=32,
  )
  ```

- [ ] **Rate limiting** configured (if needed)
- [ ] **Request queuing** for batch processing

### Response Configuration

- [ ] **Error handling** - Graceful error responses
- [ ] **Response format** documented
- [ ] **Streaming configured** (for LLMs)
  ```python
  server = create_fastapi_server(
      model,
      enable_streaming=True,
  )
  ```

---

## Monitoring & Observability

### Metrics

- [ ] **Latency metrics** - p50, p95, p99
- [ ] **Throughput metrics** - requests/second
- [ ] **Error rate** - 4xx and 5xx responses
- [ ] **GPU metrics** - utilization, memory, temperature
- [ ] **Queue depth** - pending requests

### Logging

- [ ] **Structured logging** enabled
  ```python
  import logging
  logging.basicConfig(
      format='%(asctime)s %(levelname)s %(message)s',
      level=logging.INFO
  )
  ```

- [ ] **Request logging** - input shapes, latencies
- [ ] **Error logging** - stack traces for debugging

### Alerting

- [ ] **Latency alerts** - p99 > threshold
- [ ] **Error rate alerts** - > 1% errors
- [ ] **Resource alerts** - GPU OOM, high CPU
- [ ] **Availability alerts** - health check failures

---

## Security Checklist

### Model Security

- [ ] **No secrets in model** - Check for embedded credentials
- [ ] **Safe loading** - Use `weights_only=True` or SafeTensors
  ```python
  # Safe loading
  from safetensors.torch import load_file
  state_dict = load_file("model.safetensors")
  ```

- [ ] **Input sanitization** - Validate and sanitize inputs

### Network Security

- [ ] **TLS enabled** for HTTPS
- [ ] **Authentication** - API keys or OAuth
- [ ] **Authorization** - Role-based access control
- [ ] **Network policies** - Restrict pod-to-pod traffic

### Compliance

- [ ] **Data privacy** - PII handling documented
- [ ] **Audit logging** - Track inference requests
- [ ] **Model versioning** - Track which model served which request

---

## Deployment Procedure

### Pre-deployment

1. [ ] Run full test suite
   ```bash
   pytest tests/ -m "not gpu" -v
   ```

2. [ ] Verify model checksum
   ```bash
   sha256sum model.pt > model.sha256
   ```

3. [ ] Update changelog and version

### Deployment

1. [ ] **Blue-green or canary deployment** - Don't deploy all at once

2. [ ] **Smoke tests** after deployment
   ```python
   import requests
   response = requests.post(
       "http://model-service/predict",
       json={"input": sample_input}
   )
   assert response.status_code == 200
   ```

3. [ ] **Monitor metrics** for 15-30 minutes

4. [ ] **Gradual traffic shift** (canary)

### Rollback Plan

- [ ] **Previous version available** - Can revert quickly
- [ ] **Rollback procedure documented**
- [ ] **Rollback tested** in staging environment

---

## Post-Deployment

### Verification

- [ ] **End-to-end test** from client
- [ ] **Performance baseline** recorded
- [ ] **Logs verified** - No errors
- [ ] **Metrics baseline** established

### Documentation

- [ ] **API documentation** updated
- [ ] **Runbook** for operations team
- [ ] **Incident response** procedure documented

---

## Quick Commands Reference

```bash
# System validation
tb-doctor --full-report

# Model optimization
tb-optimize --model model.pt --level production --output optimized.pt

# Benchmarking
tb-benchmark --model optimized.pt --comprehensive

# Profiling
tb-profile --model optimized.pt --mode summary

# Export
tb-export --model optimized.pt --format all --validate --output-dir exports/

# Serving (Python)
from torchbridge.deployment.serving import create_fastapi_server, run_server
server = create_fastapi_server(model)
run_server(server, host="0.0.0.0", port=8000)
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM during inference | Reduce batch size, enable gradient checkpointing |
| Slow first inference | Add warmup in server startup |
| Inconsistent outputs | Check for randomness, set seeds |
| High latency | Profile to find bottleneck, try torch.compile |
| Export fails | Simplify model, check for unsupported ops |

---

## Support Resources

- [Deployment Tutorial](deployment_tutorial.md)
- [Docker Guide](docker.md)
- [Distributed Training](distributed_training.md)
- [Backend Selection](backend_selection.md)
- [CLI Reference](../capabilities/cli_reference.md)

---

*Last verified with TorchBridge v0.4.27*
