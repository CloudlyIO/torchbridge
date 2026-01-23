"""
Whisper (Speech Recognition) optimization.

This module provides optimizations for Whisper models including:
- Audio encoding optimization
- Decoder optimization
- Real-time transcription support
"""

from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
from .base import (
    BaseMultiModalOptimizer,
    MultiModalOptimizationConfig,
    MultiModalType,
    OptimizationLevel,
    ModalityType,
    count_parameters,
)


class WhisperOptimizer(BaseMultiModalOptimizer):
    """Optimizer for Whisper models."""

    def __init__(self, config: Optional[MultiModalOptimizationConfig] = None):
        """Initialize Whisper optimizer.

        Args:
            config: Optimization configuration
        """
        if config is None:
            config = MultiModalOptimizationConfig(
                model_type=MultiModalType.WHISPER,
                modalities=[ModalityType.AUDIO, ModalityType.TEXT]
            )
        elif config.model_type != MultiModalType.WHISPER:
            config.model_type = MultiModalType.WHISPER

        super().__init__(config)

    def optimize(self, model: Any) -> Any:
        """Optimize Whisper model for inference.

        Args:
            model: Whisper model to optimize

        Returns:
            Optimized Whisper model
        """
        # Apply cuDNN optimizations
        self.apply_cudnn_optimization()

        # Move to device
        model = model.to(self.device)

        # Apply precision optimization
        if self.config.use_fp16:
            model = model.to(torch.float16)
            self.optimizations_applied.append("fp16")
        elif self.config.use_bf16:
            model = model.to(torch.bfloat16)
            self.optimizations_applied.append("bf16")

        # Apply gradient checkpointing if enabled
        if self.config.enable_gradient_checkpointing:
            model = self.apply_gradient_checkpointing(model)

        # Apply torch.compile
        if self.config.compile_model:
            try:
                # Compile encoder and decoder separately
                if hasattr(model, "encoder"):
                    model.encoder = torch.compile(model.encoder, mode="reduce-overhead")
                if hasattr(model, "decoder"):
                    model.decoder = torch.compile(model.decoder, mode="reduce-overhead")
                self.optimizations_applied.append("torch_compile")
            except Exception:
                pass

        return model

    def transcribe(
        self,
        model: Any,
        audio: Union[torch.Tensor, str],
        language: Optional[str] = None,
        task: str = "transcribe",
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """Transcribe audio to text.

        Args:
            model: Whisper model
            audio: Audio tensor or file path
            language: Target language (None for auto-detect)
            task: "transcribe" or "translate"
            batch_size: Batch size for long audio
            **kwargs: Additional transcription arguments

        Returns:
            Transcribed text
        """
        batch_size = batch_size or self.config.batch_size

        # Load audio if path is provided
        if isinstance(audio, str):
            try:
                import torchaudio
                audio, sample_rate = torchaudio.load(audio)
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    audio = resampler(audio)
            except ImportError:
                raise ImportError("torchaudio required for audio loading")

        # Ensure correct device
        audio = audio.to(self.device)

        # Ensure correct precision
        if self.config.use_fp16:
            audio = audio.half()
        elif self.config.use_bf16:
            audio = audio.bfloat16()

        # Transcribe
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=self.config.use_fp16 or self.config.use_bf16
        ):
            if hasattr(model, "transcribe"):
                result = model.transcribe(
                    audio,
                    language=language,
                    task=task,
                    **kwargs
                )
            elif hasattr(model, "generate"):
                # Use generate method for transformers models
                result = model.generate(audio, **kwargs)
            else:
                raise AttributeError("Model must have transcribe or generate method")

        return result


def create_whisper_optimizer(
    model_name: str = "openai/whisper-base",
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    batch_size: int = 1,
    device: str = "cuda",
    **kwargs
) -> tuple[Any, WhisperOptimizer]:
    """Create and optimize a Whisper model.

    Args:
        model_name: Whisper model name or path
        optimization_level: Optimization level
        batch_size: Batch size for transcription
        device: Device for inference
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    # Try transformers first, fall back to whisper package
    model = None

    try:
        from transformers import WhisperForConditionalGeneration
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    except ImportError:
        pass

    if model is None:
        try:
            import whisper
            # Extract model size from name (e.g., "openai/whisper-base" -> "base")
            model_size = model_name.split("/")[-1].replace("whisper-", "")
            model = whisper.load_model(model_size, device=device)
        except ImportError:
            raise ImportError(
                "Either transformers or whisper is required for Whisper models. "
                "Install with: pip install transformers or pip install openai-whisper"
            )

    # Create config
    config = MultiModalOptimizationConfig.from_optimization_level(
        optimization_level,
        model_type=MultiModalType.WHISPER,
        batch_size=batch_size,
        device=device,
        modalities=[ModalityType.AUDIO, ModalityType.TEXT],
        **kwargs
    )

    # Create optimizer
    optimizer = WhisperOptimizer(config)

    # Optimize model
    model = optimizer.optimize(model)

    return model, optimizer


class WhisperBenchmark:
    """Benchmark Whisper models."""

    def __init__(self, model: Any, optimizer: WhisperOptimizer):
        """Initialize benchmark.

        Args:
            model: Whisper model
            optimizer: Whisper optimizer
        """
        self.model = model
        self.optimizer = optimizer
        self.device = optimizer.device

    def benchmark_transcription(
        self,
        num_iterations: int = 10,
        audio_duration_seconds: int = 30,
        sample_rate: int = 16000,
    ) -> Dict[str, float]:
        """Benchmark transcription performance.

        Args:
            num_iterations: Number of iterations to benchmark
            audio_duration_seconds: Duration of test audio
            sample_rate: Audio sample rate

        Returns:
            Dictionary with benchmark results
        """
        import time

        # Create dummy audio (silence)
        num_samples = audio_duration_seconds * sample_rate
        dummy_audio = torch.randn(1, num_samples, device=self.device)

        if self.optimizer.config.use_fp16:
            dummy_audio = dummy_audio.half()
        elif self.optimizer.config.use_bf16:
            dummy_audio = dummy_audio.bfloat16()

        # Warmup (1 iteration)
        _ = self.optimizer.transcribe(self.model, dummy_audio)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()

        for _ in range(num_iterations):
            _ = self.optimizer.transcribe(self.model, dummy_audio)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        time_per_transcription = total_time / num_iterations

        # Real-time factor (RTF): processing_time / audio_duration
        # Lower is better; <1 means faster than real-time
        rtf = time_per_transcription / audio_duration_seconds

        return {
            "total_time_seconds": total_time,
            "time_per_transcription_seconds": time_per_transcription,
            "num_iterations": num_iterations,
            "audio_duration_seconds": audio_duration_seconds,
            "real_time_factor": rtf,
            "is_real_time": rtf < 1.0,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model information
        """
        total_params, trainable_params = count_parameters(self.model)

        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "optimization_summary": self.optimizer.get_optimization_summary(),
        }

        # Get component parameters if available
        if hasattr(self.model, "encoder"):
            encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
            info["encoder_parameters"] = encoder_params

        if hasattr(self.model, "decoder"):
            decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
            info["decoder_parameters"] = decoder_params

        return info


# Pre-configured optimizers for common Whisper variants
def create_whisper_base_optimized(
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    **kwargs
) -> tuple[Any, WhisperOptimizer]:
    """Create optimized Whisper-Base.

    Args:
        optimization_level: Optimization level
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    return create_whisper_optimizer(
        "openai/whisper-base",
        optimization_level,
        **kwargs
    )


def create_whisper_small_optimized(
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    **kwargs
) -> tuple[Any, WhisperOptimizer]:
    """Create optimized Whisper-Small.

    Args:
        optimization_level: Optimization level
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    return create_whisper_optimizer(
        "openai/whisper-small",
        optimization_level,
        **kwargs
    )


def create_whisper_large_optimized(
    optimization_level: OptimizationLevel = OptimizationLevel.O2,
    **kwargs
) -> tuple[Any, WhisperOptimizer]:
    """Create optimized Whisper-Large.

    Args:
        optimization_level: Optimization level
        **kwargs: Additional configuration options

    Returns:
        Tuple of (optimized_model, optimizer)
    """
    return create_whisper_optimizer(
        "openai/whisper-large-v2",
        optimization_level,
        **kwargs
    )
