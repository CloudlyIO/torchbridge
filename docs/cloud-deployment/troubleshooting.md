# Cloud Testing Troubleshooting Guide

This guide covers common issues and solutions for KernelPyTorch cloud testing.

## Quick Diagnostics

### Check Instance Status

```bash
# AWS
aws ec2 describe-instance-status --instance-ids i-0123456789abcdef0

# GCP
gcloud compute instances describe my-instance --zone=us-central1-a
```

### Check GPU Status

```bash
# On the instance
nvidia-smi
nvidia-smi -q -d MEMORY,UTILIZATION,TEMPERATURE

# Check for errors
dmesg | grep -i nvidia
```

### Check TPU Status

```bash
# GCP TPU
gcloud compute tpus tpu-vm describe my-tpu --zone=us-central1-a

# On TPU VM
ls /dev/accel*
python -c "import torch_xla; print(torch_xla.devices())"
```

## Common Issues

### 1. Instance Launch Failures

#### Insufficient Capacity

**Symptom**: `InsufficientInstanceCapacity` error

**Solution**:
```bash
# Try a different availability zone
aws ec2 run-instances --availability-zone us-west-2b ...

# Or use spot fleet with multiple AZs
aws ec2 request-spot-fleet --spot-fleet-request-config file://fleet-config.json
```

#### Quota Exceeded

**Symptom**: `VcpuLimitExceeded` or `QuotaExceeded` error

**Solution**:
```bash
# AWS: Check and request quota increase
aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-7212CCBC

aws service-quotas request-service-quota-increase \
    --service-code ec2 \
    --quota-code L-7212CCBC \
    --desired-value 16

# GCP: Request quota increase via console
# https://console.cloud.google.com/iam-admin/quotas
```

#### Invalid AMI/Image

**Symptom**: `InvalidAMIID.NotFound` or image not found

**Solution**:
```bash
# AWS: Find valid AMI for region
aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=*Deep Learning*PyTorch*" \
    --region us-west-2

# GCP: List available images
gcloud compute images list \
    --project deeplearning-platform-release \
    --filter="name~pytorch"
```

### 2. SSH Connection Issues

#### Connection Refused

**Symptom**: `ssh: connect to host ... port 22: Connection refused`

**Causes & Solutions**:

1. **Instance not ready**: Wait for instance to pass status checks
   ```bash
   aws ec2 wait instance-status-ok --instance-ids i-xxx
   ```

2. **Security group blocking SSH**:
   ```bash
   aws ec2 authorize-security-group-ingress \
       --group-id sg-xxx \
       --protocol tcp \
       --port 22 \
       --cidr 0.0.0.0/0
   ```

3. **Wrong key pair**:
   ```bash
   ssh -i correct-key.pem ubuntu@ip-address
   ```

#### Connection Timeout

**Symptom**: `ssh: connect to host ... port 22: Connection timed out`

**Solutions**:

1. **Check public IP assignment**:
   ```bash
   aws ec2 describe-instances --instance-ids i-xxx \
       --query "Reservations[].Instances[].PublicIpAddress"
   ```

2. **Check subnet route table** (needs internet gateway)

3. **Check network ACLs** (ensure port 22 is allowed)

### 3. GPU/CUDA Issues

#### CUDA Out of Memory

**Symptom**: `CUDA out of memory`

**Solutions**:

1. **Reduce batch size**:
   ```python
   # In test configuration
   pytest --batch-size 16  # Instead of 32
   ```

2. **Clear GPU cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Use smaller model for tests**:
   ```python
   config.hidden_size = 256  # Instead of 768
   ```

#### CUDA Driver Mismatch

**Symptom**: `CUDA driver version is insufficient for CUDA runtime version`

**Solution**:
```bash
# Check versions
nvidia-smi  # Driver version
nvcc --version  # CUDA toolkit version

# Use matching DLAMI or update driver
sudo apt-get update
sudo apt-get install nvidia-driver-535
sudo reboot
```

#### GPU Not Detected

**Symptom**: `torch.cuda.is_available()` returns False

**Solutions**:

1. **Check GPU attachment**:
   ```bash
   lspci | grep -i nvidia
   ```

2. **Check NVIDIA driver**:
   ```bash
   nvidia-smi
   # If not found, reinstall driver
   sudo apt-get install --reinstall nvidia-driver-535
   ```

3. **Check PyTorch CUDA support**:
   ```python
   import torch
   print(torch.version.cuda)  # Should match installed CUDA
   ```

### 4. TPU Issues

#### TPU Not Found

**Symptom**: `No TPU devices found`

**Solutions**:

1. **Check TPU status**:
   ```bash
   gcloud compute tpus tpu-vm describe my-tpu --zone=us-central1-a
   ```

2. **Verify TPU VM connection**:
   ```bash
   gcloud compute tpus tpu-vm ssh my-tpu --zone=us-central1-a
   ```

3. **Check libtpu**:
   ```bash
   ls /lib/libtpu.so
   ```

#### XLA Compilation Timeout

**Symptom**: Tests hang during first run (XLA compiling)

**Solutions**:

1. **Increase timeout**:
   ```python
   harness.run_tests(timeout_seconds=7200)  # 2 hours for first run
   ```

2. **Use compilation cache**:
   ```bash
   export XLA_FLAGS="--xla_dump_to=/tmp/xla_cache"
   ```

3. **Check for infinite loops in model**

#### TPU Memory Issues

**Symptom**: `Resource exhausted: Out of memory`

**Solutions**:

1. **Use smaller batch size**:
   ```python
   # XLA benefits from powers of 2
   batch_size = 32  # Instead of 48
   ```

2. **Enable memory optimization**:
   ```python
   import torch_xla.core.xla_model as xm
   xm.optimization_barrier_([tensor])
   ```

### 5. Test Failures

#### Flaky Tests

**Symptom**: Tests pass locally but fail on cloud, or intermittent failures

**Solutions**:

1. **Add retries**:
   ```bash
   pytest --reruns 3 --reruns-delay 5
   ```

2. **Check for race conditions**:
   ```python
   torch.cuda.synchronize()  # Before assertions
   ```

3. **Use deterministic mode**:
   ```python
   torch.use_deterministic_algorithms(True)
   torch.backends.cudnn.deterministic = True
   ```

#### Numerical Precision Issues

**Symptom**: `AssertionError: Tensors not close`

**Solutions**:

1. **Increase tolerance**:
   ```python
   torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
   ```

2. **Check dtype**:
   ```python
   # Ensure same dtype
   actual = actual.float()
   expected = expected.float()
   ```

3. **Account for hardware differences**:
   ```python
   # A100 vs H100 may have slight differences
   if "H100" in gpu_name:
       atol = 1e-4
   else:
       atol = 1e-5
   ```

### 6. Cost Issues

#### Runaway Costs

**Symptom**: Unexpected high charges

**Immediate Actions**:
```bash
# Terminate all test instances
aws ec2 describe-instances \
    --filters "Name=tag:Purpose,Values=kernelpytorch-testing" \
    --query "Reservations[].Instances[].InstanceId" \
    --output text | xargs -r aws ec2 terminate-instances --instance-ids

# GCP
gcloud compute instances list \
    --filter="labels.purpose=kernelpytorch-testing" \
    --format="value(name,zone)" | \
    while read name zone; do
        gcloud compute instances delete $name --zone=$zone --quiet
    done
```

**Prevention**:
```python
# Set max runtime in harness
harness = AWSTestHarness(config, max_runtime_hours=4)

# Set budget alerts
# See cost-optimization.md for details
```

#### Spot Instance Termination

**Symptom**: Instance terminated mid-test

**Solutions**:

1. **Use on-demand for critical tests**
2. **Implement checkpointing**:
   ```python
   @pytest.fixture(autouse=True)
   def checkpoint_on_spot_termination():
       # Check for termination notice every minute
       pass
   ```
3. **Use spot fleet across AZs**

### 7. Network Issues

#### S3/GCS Upload Failures

**Symptom**: `NoCredentialsError` or `Permission denied`

**Solutions**:

1. **Check IAM role/service account**:
   ```bash
   # AWS
   aws sts get-caller-identity

   # GCP
   gcloud auth list
   ```

2. **Verify bucket permissions**:
   ```bash
   # AWS
   aws s3 ls s3://bucket-name/

   # GCP
   gsutil ls gs://bucket-name/
   ```

3. **Check instance profile**:
   ```bash
   curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
   ```

#### Timeout During Package Installation

**Symptom**: `pip install` times out

**Solutions**:

1. **Use pre-built AMI with dependencies**
2. **Mirror packages in S3/GCS**:
   ```bash
   pip install --index-url https://your-mirror/simple/ package
   ```
3. **Increase timeout**:
   ```bash
   pip install --timeout 120 package
   ```

## Debugging Checklist

### Before Running Tests

- [ ] Verify AWS/GCP credentials are valid
- [ ] Check quota availability
- [ ] Confirm instance type is available in region
- [ ] Verify AMI/image exists and is accessible
- [ ] Check security group/firewall rules
- [ ] Confirm S3/GCS bucket exists and is accessible

### During Test Execution

- [ ] Monitor instance health via console
- [ ] Check GPU utilization with nvidia-smi
- [ ] Monitor memory usage
- [ ] Check for error messages in logs
- [ ] Verify network connectivity

### After Test Failure

- [ ] Collect logs from instance
- [ ] Check dmesg for system errors
- [ ] Review pytest output
- [ ] Check for known issues in this guide
- [ ] Document new issues for team

## Getting Help

### Log Collection

```bash
# Collect diagnostic logs
mkdir -p /tmp/debug-logs
dmesg > /tmp/debug-logs/dmesg.log
journalctl -u docker > /tmp/debug-logs/docker.log 2>/dev/null
nvidia-smi -q > /tmp/debug-logs/nvidia.log 2>/dev/null
pip freeze > /tmp/debug-logs/pip-freeze.txt
cat /etc/os-release > /tmp/debug-logs/os-release.txt

# Compress and upload
tar -czf debug-logs.tar.gz /tmp/debug-logs/
aws s3 cp debug-logs.tar.gz s3://bucket/debug/$(date +%Y%m%d-%H%M%S).tar.gz
```

### Contact Points

1. **Internal Issues**: Post in #kernelpytorch-cloud-testing
2. **AWS Support**: Open support case for quota/capacity issues
3. **GCP Support**: Use Cloud Console support
4. **PyTorch/XLA Issues**: Check GitHub issues or forums

## Next Steps

- [AWS Setup Guide](aws-setup.md)
- [GCP Setup Guide](gcp-setup.md)
- [Team Workflow](team-workflow.md)
