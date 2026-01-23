"""
Test suite for Native FP8 Operations

Tests the native FP8 implementation including:
- FP8 quantization and dequantization
- NativeFP8Linear layer
- FP8InferenceEngine
- Model conversion utilities
- Numerical stability
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import FP8 native modules
try:
    from kernel_pytorch.precision import (
        # Types
        FP8Dtype,
        FP8TensorSpec,
        FP8QuantizedTensor,
        # Layer
        NativeFP8Linear,
        # Inference
        FP8InferenceEngine,
        # Functions
        is_fp8_available,
        get_fp8_info,
        get_fp8_dtype,
        compute_fp8_scale,
        quantize_to_fp8,
        dequantize_from_fp8,
        convert_model_to_native_fp8,
        benchmark_fp8_layer,
        # Constants
        FP8_NATIVE_AVAILABLE,
        FP8_DTYPES_AVAILABLE,
        FP8_SCALED_MM_AVAILABLE,
    )
    FP8_NATIVE_IMPORT_SUCCESS = True
except ImportError as e:
    FP8_NATIVE_IMPORT_SUCCESS = False
    pytestmark = pytest.mark.skip(f"FP8 native not available: {e}")


class SimpleModel(nn.Module):
    """Simple model for testing"""

    def __init__(self, d_model=128, d_ff=512):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        return self.norm(x)


@pytest.fixture
def device():
    """Get test device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def simple_model(device):
    """Create simple model for testing"""
    return SimpleModel(d_model=128, d_ff=256).to(device)


# =============================================================================
# FP8 Availability Tests
# =============================================================================

@pytest.mark.skipif(not FP8_NATIVE_IMPORT_SUCCESS, reason="FP8 native not available")
class TestFP8Availability:
    """Test FP8 availability detection"""

    def test_is_fp8_available(self):
        """Test is_fp8_available function"""
        result = is_fp8_available()
        assert isinstance(result, bool)

    def test_get_fp8_info(self):
        """Test get_fp8_info function"""
        info = get_fp8_info()

        assert 'pytorch_version' in info
        assert 'fp8_native_available' in info
        assert 'fp8_scaled_mm_available' in info
        assert 'supported_formats' in info
        assert 'e4m3_max_value' in info
        assert 'e5m2_max_value' in info
        assert 'recommended_use' in info

    def test_fp8_dtype_enum(self):
        """Test FP8Dtype enum"""
        assert FP8Dtype.E4M3.value == "e4m3fn"
        assert FP8Dtype.E5M2.value == "e5m2"

    def test_get_fp8_dtype(self):
        """Test get_fp8_dtype function"""
        # Should return None if native not available, dtype otherwise
        e4m3_dtype = get_fp8_dtype(FP8Dtype.E4M3)
        e5m2_dtype = get_fp8_dtype(FP8Dtype.E5M2)

        if FP8_DTYPES_AVAILABLE:
            assert e4m3_dtype is not None
            assert e5m2_dtype is not None
        else:
            assert e4m3_dtype is None
            assert e5m2_dtype is None


# =============================================================================
# FP8 Quantization Tests
# =============================================================================

@pytest.mark.skipif(not FP8_NATIVE_IMPORT_SUCCESS, reason="FP8 native not available")
class TestFP8Quantization:
    """Test FP8 quantization and dequantization"""

    def test_compute_fp8_scale_e4m3(self, device):
        """Test scale computation for E4M3"""
        x = torch.randn(4, 256, device=device)
        scale = compute_fp8_scale(x, FP8Dtype.E4M3)

        assert scale.shape == torch.Size([])
        assert scale > 0
        assert torch.isfinite(scale)

    def test_compute_fp8_scale_e5m2(self, device):
        """Test scale computation for E5M2"""
        x = torch.randn(4, 256, device=device)
        scale = compute_fp8_scale(x, FP8Dtype.E5M2)

        assert scale.shape == torch.Size([])
        assert scale > 0
        assert torch.isfinite(scale)

    def test_compute_fp8_scale_with_margin(self, device):
        """Test scale computation with safety margin"""
        x = torch.randn(4, 256, device=device)
        scale_no_margin = compute_fp8_scale(x, FP8Dtype.E4M3, margin=0)
        scale_with_margin = compute_fp8_scale(x, FP8Dtype.E4M3, margin=2)

        # Scale with margin should be smaller (more conservative)
        assert scale_with_margin < scale_no_margin

    def test_quantize_to_fp8_e4m3(self, device):
        """Test E4M3 quantization"""
        x = torch.randn(4, 256, device=device)
        scale = compute_fp8_scale(x, FP8Dtype.E4M3)
        quantized, returned_scale = quantize_to_fp8(x, scale, FP8Dtype.E4M3)

        assert quantized.shape == x.shape
        # Convert to float for isfinite check (FP8 types don't support isfinite)
        assert torch.isfinite(quantized.to(torch.float32)).all()
        assert returned_scale == scale

    def test_quantize_to_fp8_e5m2(self, device):
        """Test E5M2 quantization"""
        x = torch.randn(4, 256, device=device)
        scale = compute_fp8_scale(x, FP8Dtype.E5M2)
        quantized, returned_scale = quantize_to_fp8(x, scale, FP8Dtype.E5M2)

        assert quantized.shape == x.shape
        # Convert to float for isfinite check (FP8 types don't support isfinite)
        assert torch.isfinite(quantized.to(torch.float32)).all()
        assert returned_scale == scale

    def test_dequantize_from_fp8(self, device):
        """Test FP8 dequantization"""
        x = torch.randn(4, 256, device=device)
        scale = compute_fp8_scale(x, FP8Dtype.E4M3)
        quantized, _ = quantize_to_fp8(x, scale, FP8Dtype.E4M3)
        dequantized = dequantize_from_fp8(quantized, scale)

        assert dequantized.shape == x.shape
        assert torch.isfinite(dequantized).all()

    def test_roundtrip_accuracy(self, device):
        """Test quantization roundtrip accuracy"""
        x = torch.randn(4, 256, device=device)
        scale = compute_fp8_scale(x, FP8Dtype.E4M3)
        quantized, _ = quantize_to_fp8(x, scale, FP8Dtype.E4M3)
        dequantized = dequantize_from_fp8(quantized, scale)

        # Check MSE is reasonable
        mse = F.mse_loss(x, dequantized)
        relative_error = mse.sqrt() / x.std()

        # Error should be small (< 10% relative error)
        assert relative_error < 0.1

    def test_e4m3_vs_e5m2_precision(self, device):
        """Test that E4M3 has better precision than E5M2"""
        x = torch.randn(4, 256, device=device)

        # E4M3 quantization
        scale_e4m3 = compute_fp8_scale(x, FP8Dtype.E4M3)
        quant_e4m3, _ = quantize_to_fp8(x, scale_e4m3, FP8Dtype.E4M3)
        dequant_e4m3 = dequantize_from_fp8(quant_e4m3, scale_e4m3)
        mse_e4m3 = F.mse_loss(x, dequant_e4m3)

        # E5M2 quantization
        scale_e5m2 = compute_fp8_scale(x, FP8Dtype.E5M2)
        quant_e5m2, _ = quantize_to_fp8(x, scale_e5m2, FP8Dtype.E5M2)
        dequant_e5m2 = dequantize_from_fp8(quant_e5m2, scale_e5m2)
        mse_e5m2 = F.mse_loss(x, dequant_e5m2)

        # E4M3 should have lower error (higher precision)
        assert mse_e4m3 < mse_e5m2


# =============================================================================
# FP8QuantizedTensor Tests
# =============================================================================

@pytest.mark.skipif(not FP8_NATIVE_IMPORT_SUCCESS, reason="FP8 native not available")
class TestFP8QuantizedTensor:
    """Test FP8QuantizedTensor class"""

    def test_from_tensor(self, device):
        """Test creating FP8QuantizedTensor from tensor"""
        x = torch.randn(4, 256, device=device)
        fp8_tensor = FP8QuantizedTensor.from_tensor(x, FP8Dtype.E4M3)

        assert fp8_tensor.shape == x.shape
        assert fp8_tensor.device.type == device.type  # Compare device types (cuda vs cuda:0)
        assert fp8_tensor.format == FP8Dtype.E4M3

    def test_dequantize(self, device):
        """Test dequantizing FP8QuantizedTensor"""
        x = torch.randn(4, 256, device=device)
        fp8_tensor = FP8QuantizedTensor.from_tensor(x, FP8Dtype.E4M3)
        dequantized = fp8_tensor.dequantize()

        assert dequantized.shape == x.shape
        assert torch.isfinite(dequantized).all()

    def test_to_device(self, device):
        """Test moving FP8QuantizedTensor to device"""
        x = torch.randn(4, 256)
        fp8_tensor = FP8QuantizedTensor.from_tensor(x, FP8Dtype.E4M3)
        moved = fp8_tensor.to(device)

        assert moved.device.type == device.type  # Compare device types (cuda vs cuda:0)

    def test_with_custom_scale(self, device):
        """Test FP8QuantizedTensor with custom scale"""
        x = torch.randn(4, 256, device=device)
        custom_scale = torch.tensor(100.0, device=device)
        fp8_tensor = FP8QuantizedTensor.from_tensor(x, FP8Dtype.E4M3, scale=custom_scale)

        assert fp8_tensor.scale == custom_scale


# =============================================================================
# NativeFP8Linear Tests
# =============================================================================

@pytest.mark.skipif(not FP8_NATIVE_IMPORT_SUCCESS, reason="FP8 native not available")
class TestNativeFP8Linear:
    """Test NativeFP8Linear layer"""

    def test_layer_creation(self, device):
        """Test NativeFP8Linear creation"""
        layer = NativeFP8Linear(256, 128, device=device)

        assert layer.in_features == 256
        assert layer.out_features == 128
        assert layer.bias is not None

    def test_layer_creation_no_bias(self, device):
        """Test NativeFP8Linear without bias"""
        layer = NativeFP8Linear(256, 128, bias=False, device=device)

        assert layer.bias is None

    def test_layer_forward(self, device):
        """Test NativeFP8Linear forward pass"""
        layer = NativeFP8Linear(256, 128, device=device)
        x = torch.randn(4, 256, device=device)
        output = layer(x)

        assert output.shape == (4, 128)
        assert torch.isfinite(output).all()

    def test_layer_forward_batch(self, device):
        """Test NativeFP8Linear with batched input"""
        layer = NativeFP8Linear(256, 128, device=device)
        x = torch.randn(8, 32, 256, device=device)  # (batch, seq, features)
        output = layer(x)

        assert output.shape == (8, 32, 128)
        assert torch.isfinite(output).all()

    def test_layer_gradient_flow(self, device):
        """Test gradient flow through NativeFP8Linear"""
        layer = NativeFP8Linear(256, 128, device=device)
        x = torch.randn(4, 256, device=device, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()

    def test_layer_weight_formats(self, device):
        """Test different weight formats"""
        layer_e4m3 = NativeFP8Linear(
            256, 128, weight_format=FP8Dtype.E4M3, device=device
        )
        layer_e5m2 = NativeFP8Linear(
            256, 128, weight_format=FP8Dtype.E5M2, device=device
        )

        assert layer_e4m3.weight_format == FP8Dtype.E4M3
        assert layer_e5m2.weight_format == FP8Dtype.E5M2

    def test_layer_get_fp8_info(self, device):
        """Test get_fp8_info method"""
        layer = NativeFP8Linear(256, 128, device=device)
        _ = layer(torch.randn(4, 256, device=device))
        info = layer.get_fp8_info()

        assert 'in_features' in info
        assert 'out_features' in info
        assert 'weight_format' in info
        assert 'weight_scale' in info
        assert 'activation_scale' in info
        assert 'fp8_native' in info

    def test_layer_sync_weights(self, device):
        """Test weight synchronization"""
        layer = NativeFP8Linear(256, 128, device=device)

        # Modify master weights
        layer.weight_master.data.fill_(1.0)

        # Sync weights
        layer.sync_weights()

        # FP8 weights should be updated (convert to float for sum operation)
        assert layer.weight_fp8.to(torch.float32).sum() > 0

    def test_layer_training_mode(self, device):
        """Test layer behavior in training mode"""
        layer = NativeFP8Linear(256, 128, device=device)
        layer.train()

        x = torch.randn(4, 256, device=device)
        _ = layer(x)

        # AMAX should be updated during training
        assert layer.activation_amax > 0

    def test_layer_eval_mode(self, device):
        """Test layer behavior in eval mode"""
        layer = NativeFP8Linear(256, 128, device=device)
        layer.eval()

        x = torch.randn(4, 256, device=device)
        output = layer(x)

        assert torch.isfinite(output).all()


# =============================================================================
# FP8InferenceEngine Tests
# =============================================================================

@pytest.mark.skipif(not FP8_NATIVE_IMPORT_SUCCESS, reason="FP8 native not available")
class TestFP8InferenceEngine:
    """Test FP8InferenceEngine"""

    def test_engine_creation(self, simple_model):
        """Test FP8InferenceEngine creation"""
        engine = FP8InferenceEngine(simple_model)

        assert engine.model is simple_model
        assert not engine._prepared

    def test_engine_prepare(self, simple_model, device):
        """Test preparing model for FP8 inference"""
        engine = FP8InferenceEngine(simple_model)
        engine.prepare(device)

        assert engine._prepared
        assert len(engine._fp8_layers) > 0

    def test_engine_infer(self, simple_model, device):
        """Test FP8 inference"""
        engine = FP8InferenceEngine(simple_model)
        engine.prepare(device)

        x = torch.randn(2, 16, 128, device=device)
        output = engine.infer(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_engine_infer_not_prepared(self, simple_model):
        """Test that inference fails if not prepared"""
        engine = FP8InferenceEngine(simple_model)

        x = torch.randn(2, 16, 128)
        with pytest.raises(RuntimeError):
            engine.infer(x)

    def test_engine_memory_savings(self, simple_model, device):
        """Test memory savings calculation"""
        engine = FP8InferenceEngine(simple_model)
        engine.prepare(device)

        savings = engine.get_memory_savings()

        assert 'fp8_memory_mb' in savings
        assert 'fp32_memory_mb' in savings
        assert 'savings_ratio' in savings
        assert 'savings_percent' in savings
        assert savings['savings_ratio'] > 0  # Should have some savings

    def test_engine_layer_info(self, simple_model, device):
        """Test getting layer info"""
        engine = FP8InferenceEngine(simple_model)
        engine.prepare(device)

        layer_info = engine.get_layer_info()

        assert len(layer_info) > 0
        for name, info in layer_info.items():
            assert 'weight_scale' in info
            assert 'activation_scale' in info

    def test_engine_with_calibration(self, simple_model, device):
        """Test engine with calibration data"""
        calibration_data = torch.randn(4, 16, 128, device=device)
        engine = FP8InferenceEngine(simple_model, calibration_data=calibration_data)
        engine.prepare(device)

        assert engine._prepared


# =============================================================================
# Model Conversion Tests
# =============================================================================

@pytest.mark.skipif(not FP8_NATIVE_IMPORT_SUCCESS, reason="FP8 native not available")
class TestModelConversion:
    """Test model conversion to FP8"""

    def test_convert_model(self, simple_model, device):
        """Test converting model to native FP8"""
        fp8_model = convert_model_to_native_fp8(simple_model)

        # Should have NativeFP8Linear layers
        fp8_count = sum(1 for m in fp8_model.modules() if isinstance(m, NativeFP8Linear))
        assert fp8_count > 0

    def test_convert_model_inplace(self, simple_model, device):
        """Test in-place conversion"""
        original_id = id(simple_model)
        fp8_model = convert_model_to_native_fp8(simple_model, inplace=True)

        assert id(fp8_model) == original_id

    def test_convert_model_not_inplace(self, simple_model, device):
        """Test non-in-place conversion"""
        fp8_model = convert_model_to_native_fp8(simple_model, inplace=False)

        # Should be a different object
        assert fp8_model is not simple_model

    def test_converted_model_forward(self, simple_model, device):
        """Test forward pass through converted model"""
        fp8_model = convert_model_to_native_fp8(simple_model)

        x = torch.randn(2, 16, 128, device=device)
        output = fp8_model(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_converted_model_backward(self, simple_model, device):
        """Test backward pass through converted model"""
        fp8_model = convert_model_to_native_fp8(simple_model)

        x = torch.randn(2, 16, 128, device=device, requires_grad=True)
        output = fp8_model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# =============================================================================
# Numerical Stability Tests
# =============================================================================

@pytest.mark.skipif(not FP8_NATIVE_IMPORT_SUCCESS, reason="FP8 native not available")
class TestNumericalStability:
    """Test numerical stability of FP8 operations"""

    def test_large_values(self, device):
        """Test with large input values"""
        x = torch.randn(4, 256, device=device) * 100
        scale = compute_fp8_scale(x, FP8Dtype.E4M3)
        quantized, _ = quantize_to_fp8(x, scale, FP8Dtype.E4M3)
        dequantized = dequantize_from_fp8(quantized, scale)

        assert torch.isfinite(dequantized).all()

    def test_small_values(self, device):
        """Test with small input values"""
        x = torch.randn(4, 256, device=device) * 0.01
        scale = compute_fp8_scale(x, FP8Dtype.E4M3)
        quantized, _ = quantize_to_fp8(x, scale, FP8Dtype.E4M3)
        dequantized = dequantize_from_fp8(quantized, scale)

        assert torch.isfinite(dequantized).all()

    def test_mixed_range(self, device):
        """Test with mixed range values"""
        x = torch.cat([
            torch.randn(2, 256, device=device) * 100,
            torch.randn(2, 256, device=device) * 0.01,
        ])
        scale = compute_fp8_scale(x, FP8Dtype.E4M3)
        quantized, _ = quantize_to_fp8(x, scale, FP8Dtype.E4M3)
        dequantized = dequantize_from_fp8(quantized, scale)

        assert torch.isfinite(dequantized).all()

    def test_fp8_linear_numerical_stability(self, device):
        """Test NativeFP8Linear with various input ranges"""
        layer = NativeFP8Linear(256, 128, device=device)

        # Normal range
        x_normal = torch.randn(4, 256, device=device)
        output_normal = layer(x_normal)
        assert torch.isfinite(output_normal).all()

        # Large range
        x_large = torch.randn(4, 256, device=device) * 50
        output_large = layer(x_large)
        assert torch.isfinite(output_large).all()

        # Small range
        x_small = torch.randn(4, 256, device=device) * 0.01
        output_small = layer(x_small)
        assert torch.isfinite(output_small).all()


# =============================================================================
# Benchmark Tests
# =============================================================================

@pytest.mark.skipif(not FP8_NATIVE_IMPORT_SUCCESS, reason="FP8 native not available")
class TestBenchmark:
    """Test FP8 benchmarking utilities"""

    def test_benchmark_fp8_layer(self, device):
        """Test benchmark_fp8_layer function"""
        results = benchmark_fp8_layer(
            in_features=256,
            out_features=128,
            batch_size=8,
            num_iterations=10,
            device=device
        )

        assert 'fp8_time_ms' in results
        assert 'standard_time_ms' in results
        assert 'speedup' in results
        assert results['fp8_time_ms'] > 0
        assert results['standard_time_ms'] > 0


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not FP8_NATIVE_IMPORT_SUCCESS, reason="FP8 native not available")
class TestIntegration:
    """Integration tests for FP8 native operations"""

    def test_end_to_end_training(self, device):
        """Test end-to-end training with FP8 layers"""
        # Create model with FP8 layers
        model = nn.Sequential(
            NativeFP8Linear(128, 256, device=device),
            nn.ReLU(),
            NativeFP8Linear(256, 64, device=device),
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        for _ in range(5):
            x = torch.randn(8, 128, device=device)
            target = torch.randn(8, 64, device=device)

            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, target)
            loss.backward()

            # Sync weights after optimizer step
            optimizer.step()
            for module in model.modules():
                if isinstance(module, NativeFP8Linear):
                    module.sync_weights()

        assert torch.isfinite(loss)

    def test_fp8_with_standard_layers(self, device):
        """Test FP8 layers mixed with standard layers"""
        model = nn.Sequential(
            nn.Linear(128, 256),  # Standard
            nn.ReLU(),
            NativeFP8Linear(256, 128, device=device),  # FP8
            nn.LayerNorm(128),  # Standard
        ).to(device)

        x = torch.randn(4, 128, device=device)
        output = model(x)

        assert output.shape == (4, 128)
        assert torch.isfinite(output).all()


# =============================================================================
# Parametrized Tests
# =============================================================================

@pytest.mark.skipif(not FP8_NATIVE_IMPORT_SUCCESS, reason="FP8 native not available")
@pytest.mark.parametrize("format_type", [FP8Dtype.E4M3, FP8Dtype.E5M2])
def test_quantization_formats(format_type, device):
    """Test quantization with different FP8 formats"""
    x = torch.randn(4, 256, device=device)
    scale = compute_fp8_scale(x, format_type)
    quantized, _ = quantize_to_fp8(x, scale, format_type)
    dequantized = dequantize_from_fp8(quantized, scale)

    assert torch.isfinite(dequantized).all()


@pytest.mark.skipif(not FP8_NATIVE_IMPORT_SUCCESS, reason="FP8 native not available")
@pytest.mark.parametrize("in_features,out_features", [
    (64, 64),
    (256, 128),
    (512, 512),
    (1024, 256),
])
def test_layer_sizes(in_features, out_features, device):
    """Test NativeFP8Linear with different sizes"""
    layer = NativeFP8Linear(in_features, out_features, device=device)
    x = torch.randn(4, in_features, device=device)
    output = layer(x)

    assert output.shape == (4, out_features)
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    # Run basic smoke test
    if FP8_NATIVE_IMPORT_SUCCESS:
        print("Running FP8 native smoke test...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # Test basic functionality
        layer = NativeFP8Linear(128, 64, device=device)
        x = torch.randn(4, 128, device=device)
        output = layer(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output finite: {torch.isfinite(output).all()}")
        print(f"FP8 info: {layer.get_fp8_info()}")

        print("\nFP8 native smoke test PASSED!")
    else:
        print("FP8 native not available - skipping smoke test")
