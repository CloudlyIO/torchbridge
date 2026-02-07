#!/usr/bin/env python3
"""
BERT SQuAD Training with TorchBridge HAL

Hardware-agnostic fine-tuning of BERT-base on SQuAD 2.0.
Works on NVIDIA CUDA, AMD ROCm, Intel XPU, Apple MPS, and CPU.

Usage:
    python train.py --epochs 2 --batch-size 16
    python train.py --backend cuda --epochs 3
    python train.py --quick  # Fast validation run
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# TorchBridge HAL imports
try:
    from torchbridge.backends import BackendType, detect_best_backend
    from torchbridge.core.hardware_detector import detect_hardware
    from torchbridge.core.unified_manager import get_manager
    TORCHBRIDGE_AVAILABLE = True
except ImportError:
    TORCHBRIDGE_AVAILABLE = False
    print("Warning: TorchBridge not available, using basic PyTorch")

# Transformers
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_name: str = "bert-base-uncased"
    dataset_name: str = "squad_v2"
    max_length: int = 384
    doc_stride: int = 128
    batch_size: int = 16
    learning_rate: float = 3e-5
    epochs: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    output_dir: str = "checkpoints"
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    fp16: bool = True
    quick_mode: bool = False  # Use subset for validation


class BERTSquadTrainer:
    """Hardware-agnostic BERT trainer using TorchBridge HAL."""

    def __init__(self, config: TrainingConfig, backend: str | None = None):
        self.config = config
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        # Detect and setup backend
        self.device, self.backend_info = self._setup_backend(backend)
        logger.info(f"Using device: {self.device} ({self.backend_info['name']})")

        # Set seed for reproducibility
        self._set_seed(config.seed)

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []

    def _setup_backend(self, requested_backend: str | None) -> tuple[torch.device, dict]:
        """Setup compute backend using TorchBridge HAL."""
        backend_info = {"name": "cpu", "type": "CPU"}

        if TORCHBRIDGE_AVAILABLE:
            # Use TorchBridge for backend detection
            if requested_backend:
                backend_map = {
                    "cuda": BackendType.NVIDIA,
                    "rocm": BackendType.AMD,
                    "xpu": BackendType.INTEL,
                    "mps": BackendType.NVIDIA,  # Fallback
                    "cpu": BackendType.CPU,
                }
                backend_type = backend_map.get(requested_backend, BackendType.CPU)
            else:
                backend_type = detect_best_backend()

            hw_info = detect_hardware()

            if backend_type == BackendType.NVIDIA and torch.cuda.is_available():
                device = torch.device("cuda")
                backend_info = {
                    "name": torch.cuda.get_device_name(0),
                    "type": "NVIDIA CUDA",
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                }
            elif backend_type == BackendType.AMD and torch.cuda.is_available():
                # ROCm uses CUDA API
                device = torch.device("cuda")
                backend_info = {
                    "name": torch.cuda.get_device_name(0),
                    "type": "AMD ROCm",
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                }
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                backend_info = {"name": "Apple Silicon", "type": "MPS"}
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device = torch.device("xpu")
                backend_info = {"name": "Intel XPU", "type": "Intel"}
            else:
                device = torch.device("cpu")
                backend_info = {"name": "CPU", "type": "CPU"}
        else:
            # Fallback without TorchBridge
            if torch.cuda.is_available():
                device = torch.device("cuda")
                backend_info = {
                    "name": torch.cuda.get_device_name(0),
                    "type": "CUDA",
                }
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                backend_info = {"name": "Apple Silicon", "type": "MPS"}
            else:
                device = torch.device("cpu")

        return device, backend_info

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _synchronize(self):
        """Backend-agnostic synchronization."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        elif self.device.type == "xpu":
            if hasattr(torch, "xpu"):
                torch.xpu.synchronize()

    def setup(self):
        """Initialize model, tokenizer, and data."""
        logger.info("=" * 60)
        logger.info("BERT SQuAD Training Setup")
        logger.info("=" * 60)

        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Load model
        logger.info(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.config.model_name)

        # Apply TorchBridge optimizations
        if TORCHBRIDGE_AVAILABLE:
            try:
                manager = get_manager()
                self.model = manager.prepare_model(self.model, optimization_level="O2")
                logger.info("Applied TorchBridge O2 optimizations")
            except Exception as e:
                logger.warning(f"TorchBridge optimization failed: {e}")

        # Move to device
        self.model = self.model.to(self.device)

        # Setup data
        self._setup_data()

        # Setup optimizer and scheduler
        self._setup_optimizer()

        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Training samples: {len(self.train_dataloader.dataset):,}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Total steps: {len(self.train_dataloader) * self.config.epochs:,}")

    def _setup_data(self):
        """Load and preprocess SQuAD dataset."""
        logger.info("Loading SQuAD dataset...")

        # Load dataset
        if self.config.dataset_name == "squad_v2":
            dataset = load_dataset("squad_v2")
        else:
            dataset = load_dataset("squad")

        # Use subset for quick mode
        if self.config.quick_mode:
            dataset["train"] = dataset["train"].select(range(min(1000, len(dataset["train"]))))
            dataset["validation"] = dataset["validation"].select(range(min(200, len(dataset["validation"]))))
            logger.info("Quick mode: using subset of data")

        # Preprocessing function
        def preprocess(examples):
            questions = [q.strip() for q in examples["question"]]
            contexts = examples["context"]

            tokenized = self.tokenizer(
                questions,
                contexts,
                max_length=self.config.max_length,
                truncation="only_second",
                stride=self.config.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            # Process answers
            sample_mapping = tokenized.pop("overflow_to_sample_mapping")
            offset_mapping = tokenized.pop("offset_mapping")

            tokenized["start_positions"] = []
            tokenized["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                sample_idx = sample_mapping[i]
                answers = examples["answers"][sample_idx]

                if len(answers["answer_start"]) == 0:
                    # No answer (SQuAD 2.0)
                    tokenized["start_positions"].append(0)
                    tokenized["end_positions"].append(0)
                else:
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    # Find token positions
                    sequence_ids = tokenized.sequence_ids(i)
                    context_start = 0
                    while sequence_ids[context_start] != 1:
                        context_start += 1
                    context_end = len(sequence_ids) - 1
                    while sequence_ids[context_end] != 1:
                        context_end -= 1

                    # Check if answer is in context
                    if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
                        tokenized["start_positions"].append(0)
                        tokenized["end_positions"].append(0)
                    else:
                        # Find start and end tokens
                        token_start = context_start
                        while token_start <= context_end and offsets[token_start][0] <= start_char:
                            token_start += 1
                        tokenized["start_positions"].append(token_start - 1)

                        token_end = context_end
                        while token_end >= context_start and offsets[token_end][1] >= end_char:
                            token_end -= 1
                        tokenized["end_positions"].append(token_end + 1)

            return tokenized

        # Tokenize datasets
        tokenized_train = dataset["train"].map(
            preprocess,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing train"
        )
        tokenized_eval = dataset["validation"].map(
            preprocess,
            batched=True,
            remove_columns=dataset["validation"].column_names,
            desc="Tokenizing validation"
        )

        # Set format for PyTorch
        tokenized_train.set_format("torch")
        tokenized_eval.set_format("torch")

        # Create dataloaders
        self.train_dataloader = DataLoader(
            tokenized_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=(self.device.type == "cuda"),
        )
        self.eval_dataloader = DataLoader(
            tokenized_eval,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )

        # Scheduler
        total_steps = len(self.train_dataloader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Mixed precision setup
        use_amp = self.config.fp16 and self.device.type == "cuda"
        scaler = torch.amp.GradScaler() if use_amp else None
        autocast_dtype = torch.float16 if use_amp else torch.float32

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            leave=True,
        )

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with autocast
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=use_amp):
                outputs = self.model(**batch)
                loss = outputs.loss

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # Log interval
            if self.global_step % self.config.log_interval == 0:
                self.training_history.append({
                    "step": self.global_step,
                    "loss": avg_loss,
                    "lr": self.scheduler.get_last_lr()[0],
                })

            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

        return {"loss": total_loss / num_batches}

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

        return {"eval_loss": total_loss / num_batches}

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.config.__dict__,
            "backend_info": self.backend_info,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def train(self) -> dict:
        """Run full training loop."""
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Evaluate
            eval_metrics = self.evaluate()

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Eval Loss: {eval_metrics['eval_loss']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save best model
            if eval_metrics["eval_loss"] < self.best_loss:
                self.best_loss = eval_metrics["eval_loss"]
                self._save_checkpoint("bert_squad_best.pt")

        total_time = time.time() - start_time

        # Save final model
        self._save_checkpoint("bert_squad_final.pt")

        # Save training results
        results = {
            "total_time_seconds": total_time,
            "final_train_loss": train_metrics["loss"],
            "final_eval_loss": eval_metrics["eval_loss"],
            "best_eval_loss": self.best_loss,
            "total_steps": self.global_step,
            "backend": self.backend_info,
            "config": self.config.__dict__,
            "training_history": self.training_history,
        }

        results_path = self.results_dir / f"training_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Total time: {total_time / 60:.1f} minutes")
        logger.info(f"Best eval loss: {self.best_loss:.4f}")
        logger.info(f"Results saved: {results_path}")
        logger.info("=" * 60)

        return results


def main():
    parser = argparse.ArgumentParser(description="BERT SQuAD Training with TorchBridge HAL")

    # Model and data
    parser.add_argument("--model-name", default="bert-base-uncased", help="Model name")
    parser.add_argument("--dataset", default="squad_v2", choices=["squad", "squad_v2"])

    # Training
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=384, help="Max sequence length")

    # Backend
    parser.add_argument("--backend", choices=["cuda", "rocm", "xpu", "mps", "cpu"],
                       help="Force specific backend (default: auto-detect)")

    # Options
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable FP16")
    parser.add_argument("--quick", action="store_true", help="Quick validation run")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        model_name=args.model_name,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        fp16=args.fp16,
        quick_mode=args.quick,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Train
    trainer = BERTSquadTrainer(config, backend=args.backend)
    trainer.setup()
    results = trainer.train()

    return 0 if results["best_eval_loss"] < 2.0 else 1


if __name__ == "__main__":
    sys.exit(main())
