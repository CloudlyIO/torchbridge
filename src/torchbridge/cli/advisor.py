"""
Advisor command for TorchBridge CLI.

Inspects hardware topology and recommends optimal distributed training
parallelism configuration from model size and cluster setup.
"""

import argparse
import json
import sys

from torchbridge.core.config import HardwareBackend


class AdvisorCommand:
    """Distributed training parallelism advisor command."""

    @staticmethod
    def register(subparsers) -> None:
        """Register the advisor command with argument parser."""
        parser = subparsers.add_parser(
            "advisor",
            help="Recommend distributed training parallelism configuration",
            description="Analyze hardware and model to recommend optimal parallelism",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  torchbridge advisor --model-params 7e9 --world-size 8
  torchbridge advisor --model-params 70e9 --world-size 16 --gpus-per-node 8
  torchbridge advisor --model-params 32e9 --world-size 8 --backend nvidia --ci
  torchbridge advisor --model-params 7e9 --world-size 4 --toml
            """,
        )

        parser.add_argument(
            "--model-params",
            type=float,
            required=True,
            help="Total model parameters (e.g., 7e9 for 7B)",
        )

        parser.add_argument(
            "--world-size",
            type=int,
            default=1,
            help="Total number of ranks/GPUs (default: 1)",
        )

        parser.add_argument(
            "--gpus-per-node",
            type=int,
            default=None,
            help="GPUs per node (default: same as world-size)",
        )

        parser.add_argument(
            "--backend",
            choices=["auto", "nvidia", "amd", "trainium", "tpu", "cpu"],
            default="auto",
            help="Target backend (default: auto-detect)",
        )

        parser.add_argument(
            "--ci",
            action="store_true",
            help="Output JSON for CI pipelines",
        )

        parser.add_argument(
            "--toml",
            action="store_true",
            help="Output full TOML configuration",
        )

        parser.add_argument(
            "--topology",
            action="store_true",
            help="Detect and display cluster topology",
        )

    @staticmethod
    def execute(args) -> int:
        """Execute the advisor command."""
        from torchbridge.distributed.config import (
            DistributedConfig,
            recommend_parallelism,
        )

        backend = _resolve_backend(args.backend)
        model_params = int(args.model_params)
        world_size = args.world_size
        gpus_per_node = args.gpus_per_node

        # Topology detection mode
        if args.topology:
            return _show_topology(backend, ci=args.ci)

        # Compute recommendation
        recommendation = recommend_parallelism(
            model_params=model_params,
            backend=backend,
            world_size=world_size,
            gpus_per_node=gpus_per_node,
        )

        # Full distributed config (for TOML output)
        if args.toml:
            config = DistributedConfig.auto(
                model_params=model_params,
                backend=backend,
                world_size=world_size,
                gpus_per_node=gpus_per_node,
            )
            print(config.to_toml())
            return 0

        # JSON output for CI
        if args.ci:
            output = {
                "model_params": model_params,
                "world_size": world_size,
                "backend": backend.value,
                "recommendation": recommendation.to_dict(),
            }
            json.dump(output, sys.stdout, indent=2)
            print()
            return 0

        # Human-readable output
        _print_recommendation(model_params, world_size, backend, recommendation)
        return 0


def _resolve_backend(backend_str: str) -> HardwareBackend:
    """Resolve backend string to HardwareBackend enum."""
    if backend_str == "auto":
        return _detect_backend()

    mapping = {
        "nvidia": HardwareBackend.CUDA,
        "amd": HardwareBackend.AMD,
        "trainium": HardwareBackend.TRAINIUM,
        "tpu": HardwareBackend.TPU,
        "cpu": HardwareBackend.CPU,
    }
    return mapping.get(backend_str, HardwareBackend.CPU)


def _detect_backend() -> HardwareBackend:
    """Auto-detect hardware backend."""
    try:
        import torch

        if torch.cuda.is_available():
            if hasattr(torch.version, "hip") and torch.version.hip:
                return HardwareBackend.AMD
            return HardwareBackend.CUDA
    except ImportError:
        pass
    return HardwareBackend.CPU


def _show_topology(backend: HardwareBackend, ci: bool = False) -> int:
    """Detect and display cluster topology."""
    from torchbridge.distributed.collective_backend import CollectiveBackendMatrix
    from torchbridge.distributed.topology import TopologyDetector

    mesh = TopologyDetector.detect_mesh_from_environment()
    collective = CollectiveBackendMatrix.get_optimal_backend(backend)

    if ci:
        output = {
            "mesh": mesh.to_dict(),
            "collective_backend": collective.value,
        }
        json.dump(output, sys.stdout, indent=2)
        print()
        return 0

    print("Cluster Topology")
    print("=" * 50)
    print(f"  World size:       {mesh.world_size}")
    print(f"  GPUs per node:    {mesh.local_world_size}")
    print(f"  Nodes:            {mesh.num_nodes}")
    print(f"  Intra-node:       {mesh.interconnect_intra.value}")
    print(f"  Inter-node:       {mesh.interconnect_inter.value}")
    print(f"  Mesh shape:       {list(mesh.mesh_shape)}")
    print(f"  Mesh dims:        {list(mesh.mesh_dim_names)}")
    print(f"  Collective:       {collective.value}")
    return 0


def _print_recommendation(
    model_params: int,
    world_size: int,
    backend: HardwareBackend,
    recommendation,
) -> None:
    """Print human-readable recommendation."""
    print("Distributed Training Recommendation")
    print("=" * 50)
    print()
    print(f"  Model:            {model_params / 1e9:.1f}B parameters")
    print(f"  World size:       {world_size}")
    print(f"  Backend:          {backend.value}")
    print()
    print("Parallelism Strategy")
    print("-" * 50)
    print(f"  Tensor parallel:  {recommendation.tensor_parallel_degree}")
    print(f"  Pipeline stages:  {recommendation.pipeline_parallel_stages}")
    print(f"  FSDP strategy:    {recommendation.fsdp_strategy}")
    print(f"  Mixed precision:  {recommendation.mixed_precision}")
    print()
    print("Estimates")
    print("-" * 50)
    print(f"  Memory/rank:      {recommendation.estimated_memory_per_rank_gb:.2f} GB")
    print(f"  Comm volume:      {recommendation.estimated_communication_volume_gb:.2f} GB/step")
    print()

    if recommendation.notes:
        print("Notes")
        print("-" * 50)
        for note in recommendation.notes:
            print(f"  - {note}")
        print()


def main(args=None):
    """Entry point for tb-advisor."""
    parser = argparse.ArgumentParser(
        prog="tb-advisor",
        description="Recommend distributed training parallelism configuration",
    )
    parser.add_argument("--model-params", type=float, required=True,
                        help="Total model parameters (e.g., 7e9 for 7B)")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Total number of ranks/GPUs (default: 1)")
    parser.add_argument("--gpus-per-node", type=int, default=None,
                        help="GPUs per node (default: same as world-size)")
    parser.add_argument(
        "--backend",
        choices=["auto", "nvidia", "amd", "trainium", "tpu", "cpu"],
        default="auto",
        help="Target backend (default: auto-detect)",
    )
    parser.add_argument("--ci", action="store_true",
                        help="Output JSON for CI pipelines")
    parser.add_argument("--toml", action="store_true",
                        help="Output full TOML configuration")
    parser.add_argument("--topology", action="store_true",
                        help="Detect and display cluster topology")

    if args is None:
        args = sys.argv[1:]

    parsed = parser.parse_args(args)
    return AdvisorCommand.execute(parsed)
