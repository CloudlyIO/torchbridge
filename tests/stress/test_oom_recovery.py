import pytest
import torch


@pytest.mark.stress
class TestOOMRecovery:
    def test_large_tensor_allocation_fails_gracefully(self):
        """Attempting to allocate impossibly large tensor raises clean error."""
        # Use a shape so large that PyTorch rejects it before the OS allocator
        # (2^63 elements overflows internal size calculations)
        with pytest.raises((RuntimeError, OverflowError, MemoryError, OSError)):
            _ = torch.empty(2**63 - 1)

    def test_recovery_after_failed_allocation(self):
        """After a failed allocation, normal operations still work."""
        try:
            _ = torch.empty(2**63 - 1)
        except (RuntimeError, OverflowError, MemoryError, OSError):
            pass
        # Normal operation should work fine
        x = torch.randn(32, 64)
        y = torch.matmul(x, x.T)
        assert y.shape == (32, 32)

    @pytest.mark.gpu
    def test_gpu_oom_recovery(self):
        """GPU OOM triggers cleanup and allows subsequent operations."""
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        # Try to allocate more than GPU memory
        total_mem = torch.cuda.get_device_properties(0).total_memory
        try:
            # Allocate 2x total GPU memory
            elems = (total_mem * 2) // 4  # float32 = 4 bytes
            _ = torch.randn(elems, device=device)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pass

        # Cleanup and verify normal ops work
        torch.cuda.empty_cache()
        x = torch.randn(128, 128, device=device)
        y = torch.matmul(x, x.T)
        assert y.shape == (128, 128)
        del x, y
        torch.cuda.empty_cache()

    @pytest.mark.gpu
    def test_gpu_oom_during_model_inference(self, minilm_model_and_tokenizer):
        """Model inference with absurdly large batch triggers OOM, recovers."""
        model, tokenizer = minilm_model_and_tokenizer
        device = torch.device("cuda")
        model_gpu = model.to(device)

        # Try batch of 100K -- should OOM on most GPUs
        try:
            texts = ["test"] * 100_000
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                _ = model_gpu(**inputs)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pass

        # Recovery: small batch should work
        torch.cuda.empty_cache()
        small_inputs = tokenizer(["recovery test"], return_tensors="pt")
        small_inputs = {k: v.to(device) for k, v in small_inputs.items()}
        with torch.no_grad():
            output = model_gpu(**small_inputs)
        assert output.last_hidden_state.shape[0] == 1
        model.cpu()
