# 🐳 Docker Guide

**TorchBridge** provides production-ready Docker containers for consistent development and deployment environments.

## Quick Start

### Pull and Run Production Container

```bash
# Pull latest production image
docker pull ghcr.io/torchbridge/torchbridge:latest

# Run system diagnostics
docker run --rm ghcr.io/torchbridge/torchbridge:latest doctor --verbose

# Run with GPU access
docker run --rm --gpus all ghcr.io/torchbridge/torchbridge:latest doctor --full-report
```

### Development Environment

```bash
# Start development container with volume mounting
docker run -it --rm \
  -v $(pwd):/workspace \
  --gpus all \
  ghcr.io/torchbridge/torchbridge:dev
```

## Available Images

### Production Image (`torchbridge:latest`)

**Purpose**: Optimized for production deployment and model serving

**Features**:
- Minimal Ubuntu 22.04 base with CUDA 11.8 runtime
- TorchBridge with core dependencies
- Multi-arch support (x86_64, ARM64)
- Non-root user for security
- Health checks and proper signal handling

**Size**: ~2.5 GB

```bash
# Basic usage
docker run ghcr.io/torchbridge/torchbridge:latest

# Available commands
docker run ghcr.io/torchbridge/torchbridge:latest doctor
docker run ghcr.io/torchbridge/torchbridge:latest server
docker run ghcr.io/torchbridge/torchbridge:latest benchmark
```

### Development Image (`torchbridge:dev`)

**Purpose**: Complete development environment with all tools

**Features**:
- Ubuntu 22.04 with CUDA 11.8 development tools
- All optional dependencies (Triton, Flash Attention, etc.)
- Development tools (Jupyter, TensorBoard, etc.)
- Pre-configured development environment

**Size**: ~8 GB

```bash
# Interactive development
docker run -it --gpus all ghcr.io/torchbridge/torchbridge:dev bash

# Jupyter Lab server
docker run -p 8888:8888 --gpus all ghcr.io/torchbridge/torchbridge:dev \
  jupyter lab --ip=0.0.0.0 --allow-root
```

## Docker Compose

### Complete Development Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  torchbridge:
    image: ghcr.io/torchbridge/torchbridge:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  development:
    image: ghcr.io/torchbridge/torchbridge:dev
    ports:
      - "8888:8888"  # Jupyter
      - "6006:6006"  # TensorBoard
    volumes:
      - .:/workspace
    environment:
      - PYTHONPATH=/workspace/src
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      bash -c "
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
        tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006 &
        tail -f /dev/null
      "
```

```bash
# Start development environment
docker-compose up development

# Start production service
docker-compose up torchbridge
```

## Building Custom Images

### Custom Production Image

```dockerfile
FROM ghcr.io/torchbridge/torchbridge:latest

# Add your model and configurations
COPY my_model.pt /app/models/
COPY config.yaml /app/config/

# Set custom entrypoint
CMD ["torchbridge", "optimize", "--model", "/app/models/my_model.pt", "--level", "production"]
```

### Custom Development Image

```dockerfile
FROM ghcr.io/torchbridge/torchbridge:dev

# Install additional development dependencies
RUN pip install wandb mlflow optuna

# Copy your source code
COPY . /workspace/

# Set working directory
WORKDIR /workspace

# Install in development mode
RUN pip install -e .[all,dev]
```

## Container Commands

### Available Entrypoint Commands

```bash
# System diagnostics
docker run --rm torchbridge:latest doctor

# Start optimization server
docker run -p 8000:8000 torchbridge:latest server

# Run benchmarks
docker run torchbridge:latest benchmark

# Model optimization
docker run -v $(pwd):/data torchbridge:latest optimize \
  --model /data/model.pt --output /data/optimized_model.pt

# Interactive bash shell
docker run -it torchbridge:latest bash

# Python directly
docker run torchbridge:latest python -c "import torchbridge; print('✓ Working')"
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | `0` |
| `PORT` | Server port | `8000` |
| `WORKERS` | Number of workers | `2` |
| `PYTHONPATH` | Python path | `/app` |

## GPU Support

### NVIDIA GPU Setup

**Prerequisites**:
- NVIDIA Docker runtime installed
- CUDA-compatible GPU
- Recent GPU drivers

```bash
# Install NVIDIA Container Toolkit (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Usage with GPU**:
```bash
# Single GPU
docker run --gpus all torchbridge:latest doctor --category hardware

# Specific GPU
docker run --gpus '"device=0"' torchbridge:latest

# Multiple GPUs
docker run --gpus 2 torchbridge:latest
```

### AMD GPU Support (ROCm)

```bash
# Use ROCm base image (custom build required)
docker build -f Dockerfile.rocm -t torchbridge:rocm .

docker run --device=/dev/kfd --device=/dev/dri torchbridge:rocm
```

## Production Deployment

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: torchbridge-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: torchbridge
  template:
    metadata:
      labels:
        app: torchbridge
    spec:
      containers:
      - name: torchbridge
        image: ghcr.io/torchbridge/torchbridge:latest
        ports:
        - containerPort: 8000
        env:
        - name: PORT
          value: "8000"
        - name: WORKERS
          value: "4"
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

```bash
kubectl apply -f k8s/deployment.yaml
kubectl expose deployment torchbridge-service --port=80 --target-port=8000
```

### AWS ECS Deployment

```json
{
  "family": "torchbridge-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "2048",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "torchbridge",
      "image": "ghcr.io/torchbridge/torchbridge:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "PORT", "value": "8000"},
        {"name": "WORKERS", "value": "4"}
      ],
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ]
    }
  ]
}
```

## Development Workflows

### Local Development

```bash
# Start development container
docker run -it --rm \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  -p 6006:6006 \
  --gpus all \
  ghcr.io/torchbridge/torchbridge:dev

# Inside container
cd /workspace
pip install -e .[dev]
pytest tests/
jupyter lab --ip=0.0.0.0 --allow-root
```

### Code Testing

```bash
# Run tests in container
docker run --rm -v $(pwd):/workspace torchbridge:dev \
  bash -c "cd /workspace && python -m pytest tests/ -v"

# Run specific test
docker run --rm -v $(pwd):/workspace torchbridge:dev \
  bash -c "cd /workspace && python -m pytest tests/cli/ -v"
```

### CI/CD Integration

```yaml
# .github/workflows/docker-test.yml
name: Docker Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Build test image
      run: docker build -f docker/Dockerfile.development -t torchbridge:test .

    - name: Run tests
      run: |
        docker run --rm -v $PWD:/workspace torchbridge:test \
          bash -c "cd /workspace && python -m pytest tests/ -v"

    - name: Run CLI tests
      run: |
        docker run --rm torchbridge:test \
          bash -c "torchbridge doctor && torchbridge --version"
```

## Monitoring and Logging

### Container Health Monitoring

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# View logs
docker logs torchbridge-container

# Monitor resource usage
docker stats torchbridge-container
```

### Prometheus Integration

```yaml
# docker-compose.yml (monitoring stack)
services:
  torchbridge:
    image: ghcr.io/torchbridge/torchbridge:latest
    # ... other config

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=torchbridge
```

## Troubleshooting

### Common Issues

#### GPU Not Available

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi

# Check TorchBridge GPU detection
docker run --rm --gpus all torchbridge:latest python -c \
  "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Permission Issues

```bash
# Run with correct user permissions
docker run --rm -u $(id -u):$(id -g) -v $(pwd):/workspace torchbridge:dev

# Fix file ownership (if needed)
sudo chown -R $(id -u):$(id -g) ./data/
```

#### Memory Issues

```bash
# Increase container memory limits
docker run --memory=8g --shm-size=2g torchbridge:latest

# Monitor memory usage
docker stats --no-stream torchbridge-container
```

#### Network Issues

```bash
# Check port binding
docker run -p 8000:8000 torchbridge:latest server
curl http://localhost:8000/health

# Debug network
docker run -it --rm torchbridge:latest bash
# Inside container: test network connectivity
```

## Performance Optimization

### Container Performance Tips

1. **Use specific GPU devices**: `--gpus '"device=0"'` instead of `--gpus all`
2. **Optimize memory**: Set appropriate `--shm-size` for large models
3. **Volume mounting**: Use bind mounts for development, volumes for production
4. **Multi-stage builds**: Keep production images lean

### Resource Limits

```bash
# Production container with resource limits
docker run \
  --memory=4g \
  --cpus=2.0 \
  --gpus=1 \
  --shm-size=1g \
  ghcr.io/torchbridge/torchbridge:latest
```

---

**Need help?** Check container health with `docker run --rm torchbridge:latest doctor`