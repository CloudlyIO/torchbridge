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

        parser.add_argument(
            "--mode",
            choices=["training", "disaggregated", "heterogeneous"],
            default="training",
            help="Operating mode: training (default), disaggregated serving, or heterogeneous cluster",
        )

        parser.add_argument(
            "--nvidia",
            metavar="ARCH:COUNT",
            default=None,
            help=(
                "NVIDIA GPU spec for heterogeneous mode: ARCH:COUNT "
                "(e.g. hopper:4, ampere:8, blackwell_dc:2)"
            ),
        )

        parser.add_argument(
            "--amd",
            metavar="ARCH:COUNT",
            default=None,
            help=(
                "AMD GPU spec for heterogeneous mode: ARCH:COUNT "
                "(e.g. cdna3:8, cdna4:4, cdna2:16)"
            ),
        )

        parser.add_argument(
            "--prefill",
            metavar="SPEC",
            default=None,
            help="Prefill hardware spec: BACKEND[:ARCH] (e.g. nvidia:hopper, amd:cdna3)",
        )

        parser.add_argument(
            "--decode",
            metavar="SPEC",
            default=None,
            help="Decode hardware spec: BACKEND[:ARCH] (e.g. amd:cdna3, nvidia:hopper)",
        )

        parser.add_argument(
            "--prefill-memory",
            type=float,
            default=None,
            metavar="N",
            help="Prefill GPU memory in GB (overrides matrix default)",
        )

        parser.add_argument(
            "--decode-memory",
            type=float,
            default=None,
            metavar="N",
            help="Decode GPU memory in GB (overrides matrix default)",
        )

    @staticmethod
    def execute(args) -> int:
        """Execute the advisor command."""
        # Disaggregated serving mode
        if getattr(args, "mode", "training") == "disaggregated":
            return AdvisorCommand._run_disaggregated(args)

        # Heterogeneous cluster mode
        if getattr(args, "mode", "training") == "heterogeneous":
            return AdvisorCommand._run_heterogeneous(args)

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


    @staticmethod
    def _run_disaggregated(args) -> int:
        """Execute the disaggregated fleet advisor."""
        from torchbridge.inference.disaggregated import DisaggregatedFleetAdvisor

        prefill_spec = getattr(args, "prefill", None)
        decode_spec = getattr(args, "decode", None)

        if not prefill_spec:
            print("Error: --mode disaggregated requires --prefill BACKEND[:ARCH]")
            return 1
        if not decode_spec:
            print("Error: --mode disaggregated requires --decode BACKEND[:ARCH]")
            return 1

        prefill_backend, prefill_arch = _parse_hw_spec(prefill_spec)
        decode_backend, decode_arch = _parse_hw_spec(decode_spec)
        model_params = int(getattr(args, "model_params", 0))

        cfg = DisaggregatedFleetAdvisor.recommend(
            model_params=model_params,
            prefill_backend=prefill_backend,
            prefill_arch=prefill_arch,
            decode_backend=decode_backend,
            decode_arch=decode_arch,
            prefill_memory_gb=getattr(args, "prefill_memory", None),
            decode_memory_gb=getattr(args, "decode_memory", None),
        )

        if getattr(args, "ci", False):
            json.dump(cfg.to_dict(), sys.stdout, indent=2)
            print()
            return 0

        _print_fleet_config(cfg)
        return 0

    @staticmethod
    def _run_heterogeneous(args) -> int:
        """Execute the heterogeneous cluster config advisor."""
        from torchbridge.distributed.hetero import HeterogeneousClusterAdvisor

        nvidia_spec = getattr(args, "nvidia", None)
        amd_spec = getattr(args, "amd", None)

        if not nvidia_spec:
            print("Error: --mode heterogeneous requires --nvidia ARCH:COUNT")
            return 1
        if not amd_spec:
            print("Error: --mode heterogeneous requires --amd ARCH:COUNT")
            return 1

        nvidia_arch, nvidia_count = _parse_hetero_spec(nvidia_spec, "nvidia")
        amd_arch, amd_count = _parse_hetero_spec(amd_spec, "amd")
        model_params = int(getattr(args, "model_params", 0))

        cfg = HeterogeneousClusterAdvisor.recommend(
            nvidia_count=nvidia_count,
            nvidia_arch=nvidia_arch,
            amd_count=amd_count,
            amd_arch=amd_arch,
            model_params=model_params,
        )

        if getattr(args, "ci", False):
            json.dump(cfg.to_dict(), sys.stdout, indent=2)
            print()
            return 0

        _print_hetero_config(cfg)
        return 0


def _parse_hw_spec(spec: str) -> tuple[str, str | None]:
    """Parse BACKEND[:ARCH] spec into (backend, arch) tuple.

    Vendor aliases are normalised:
      nvidia → cuda, amd → rocm
    """
    parts = spec.lower().split(":", 1)
    raw_backend = parts[0].strip()
    arch = parts[1].strip() if len(parts) > 1 else None

    backend_aliases = {
        "nvidia": "cuda",
        "amd": "rocm",
        "hip": "rocm",
    }
    backend = backend_aliases.get(raw_backend, raw_backend)
    return backend, arch


def _print_fleet_config(cfg) -> None:
    """Print human-readable disaggregated fleet config table."""
    from torchbridge.inference.disaggregated import _ARCH_LABELS

    def arch_label(backend: str, arch) -> str:
        return _ARCH_LABELS.get((backend, arch), f"{backend}/{arch or 'unknown'}")

    print("TorchBridge Disaggregated Fleet Config")
    print("=" * 55)
    print(f"Model     : {cfg.model_params / 1e9:.1f}B parameters")
    print(f"Prefill   : {arch_label(cfg.prefill.backend, cfg.prefill.architecture):<16}"
          f"  ({cfg.prefill.backend} / {cfg.prefill.architecture or 'unknown'})")
    print(f"Decode    : {arch_label(cfg.decode.backend, cfg.decode.architecture):<16}"
          f"  ({cfg.decode.backend} / {cfg.decode.architecture or 'unknown'})")
    print()
    print(f"{'Role':<8}  {'KV dtype':<10}  {'KV budget':<10}  "
          f"{'Max batch':<10}  {'Max seq':<8}  Transfer fmt")
    print(f"{'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*12}")

    p = cfg.prefill
    print(f"{'prefill':<8}  {p.kv_dtype:<10}  {p.kv_cache_budget_gb:<8.1f} GB"
          f"  {p.max_batch_size:<10}  {p.max_seq_len:<8}  → {cfg.kv_transfer_format}")

    d = cfg.decode
    print(f"{'decode':<8}  {d.kv_dtype:<10}  {d.kv_cache_budget_gb:<8.1f} GB"
          f"  {d.max_batch_size:<10}  {d.max_seq_len:<8}  ← {cfg.kv_transfer_format}")

    all_notes = p.notes + d.notes + cfg.notes
    if all_notes:
        print()
        print("Notes:")
        for note in all_notes:
            print(f"  - {note}")
    print()


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


def _parse_hetero_spec(spec: str, vendor: str) -> tuple:
    """Parse ARCH:COUNT spec into (arch_enum_or_None, count).

    Examples::
        "hopper:4"   → (NVIDIAArchitecture.HOPPER, 4)
        "cdna3:8"    → (AMDArchitecture.CDNA3, 8)
        "unknown:2"  → (None, 2)
    """
    from torchbridge.core.config import AMDArchitecture, NVIDIAArchitecture

    _NVIDIA_ARCH_MAP = {
        "pascal": NVIDIAArchitecture.PASCAL,
        "volta": NVIDIAArchitecture.VOLTA,
        "turing": NVIDIAArchitecture.TURING,
        "ampere": NVIDIAArchitecture.AMPERE,
        "ada": NVIDIAArchitecture.ADA,
        "hopper": NVIDIAArchitecture.HOPPER,
        "blackwell_dc": NVIDIAArchitecture.BLACKWELL_DC,
        "blackwell": NVIDIAArchitecture.BLACKWELL_DC,
        "blackwell_consumer": NVIDIAArchitecture.BLACKWELL_CONSUMER,
    }
    _AMD_ARCH_MAP = {
        "cdna": AMDArchitecture.CDNA,
        "cdna2": AMDArchitecture.CDNA2,
        "cdna3": AMDArchitecture.CDNA3,
        "cdna4": AMDArchitecture.CDNA4,
        "rdna2": AMDArchitecture.RDNA2,
        "rdna3": AMDArchitecture.RDNA3,
    }

    parts = spec.lower().split(":", 1)
    arch_str = parts[0].strip()
    count = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 1

    if vendor == "nvidia":
        arch = _NVIDIA_ARCH_MAP.get(arch_str)
    else:
        arch = _AMD_ARCH_MAP.get(arch_str)

    return arch, count


def _print_hetero_config(cfg) -> None:
    """Print human-readable heterogeneous cluster configuration."""
    nvidia_label = cfg.nvidia_arch.value if cfg.nvidia_arch else "unknown"
    amd_label = cfg.amd_arch.value if cfg.amd_arch else "unknown"

    print("TorchBridge Heterogeneous Cluster Config")
    print("=" * 55)
    print(f"Model      : {cfg.model_params / 1e9:.1f}B parameters")
    print(f"NVIDIA     : {cfg.nvidia_count}× {nvidia_label}")
    print(f"AMD        : {cfg.amd_count}× {amd_label}")
    print()
    print(f"{'Setting':<28}  {'NVIDIA':<14}  AMD")
    print(f"{'─'*28}  {'─'*14}  {'─'*14}")
    print(f"{'Collective bridge':<28}  {cfg.collective_bridge:<14}")
    print(f"{'Partition strategy':<28}  {cfg.partition_strategy:<14}")
    print(f"{'FSDP strategy':<28}  {cfg.nvidia_fsdp_strategy:<14}  {cfg.amd_fsdp_strategy}")
    print(f"{'Mixed precision':<28}  {cfg.nvidia_mixed_precision:<14}  {cfg.amd_mixed_precision}")
    print(f"{'Est. cross-vendor comm':<28}  {cfg.estimated_cross_vendor_comm_gb:.2f} GB/step")
    if cfg.notes:
        print()
        print("Notes:")
        for note in cfg.notes:
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
    parser.add_argument("--mode", choices=["training", "disaggregated", "heterogeneous"],
                        default="training",
                        help="Operating mode: training (default), disaggregated serving, or heterogeneous cluster")
    parser.add_argument("--prefill", metavar="SPEC", default=None,
                        help="Prefill hardware spec: BACKEND[:ARCH] (e.g. nvidia:hopper)")
    parser.add_argument("--decode", metavar="SPEC", default=None,
                        help="Decode hardware spec: BACKEND[:ARCH] (e.g. amd:cdna3)")
    parser.add_argument("--prefill-memory", type=float, default=None, metavar="N",
                        help="Prefill GPU memory in GB")
    parser.add_argument("--decode-memory", type=float, default=None, metavar="N",
                        help="Decode GPU memory in GB")
    parser.add_argument("--nvidia", metavar="ARCH:COUNT", default=None,
                        help="NVIDIA GPU spec for heterogeneous mode: ARCH:COUNT (e.g. hopper:4)")
    parser.add_argument("--amd", metavar="ARCH:COUNT", default=None,
                        help="AMD GPU spec for heterogeneous mode: ARCH:COUNT (e.g. cdna3:8)")

    if args is None:
        args = sys.argv[1:]

    parsed = parser.parse_args(args)
    return AdvisorCommand.execute(parsed)
