"""
Checkpoint command for TorchBridge CLI.

Provides checkpoint inspection, listing, frequency advising, and
cross-backend conversion utilities.
"""

import argparse
import json
import sys
from typing import Any


def _register_subcommands(subparsers: Any) -> None:
    """Register checkpoint subcommands (shared by CLI and standalone entry)."""
    # info subcommand
    info_parser = subparsers.add_parser(
        "info",
        help="Show checkpoint metadata",
    )
    info_parser.add_argument(
        "path",
        help="Path to checkpoint directory",
    )

    # list subcommand
    list_parser = subparsers.add_parser(
        "list",
        help="List checkpoints in a directory",
    )
    list_parser.add_argument(
        "path",
        help="Base checkpoint storage path",
    )
    list_parser.add_argument(
        "--ci",
        action="store_true",
        help="Output JSON for CI pipelines",
    )

    # advisor subcommand
    advisor_parser = subparsers.add_parser(
        "advisor",
        help="Recommend checkpoint frequency",
    )
    advisor_parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Total number of GPUs/ranks (default: 1)",
    )
    advisor_parser.add_argument(
        "--checkpoint-time",
        type=float,
        default=60.0,
        help="Estimated checkpoint save time in seconds (default: 60)",
    )
    advisor_parser.add_argument(
        "--step-time",
        type=float,
        default=1.0,
        help="Average training step time in seconds (default: 1.0)",
    )
    advisor_parser.add_argument(
        "--mtbf",
        type=float,
        default=None,
        help="Mean time between failures in hours (default: auto-estimate)",
    )
    advisor_parser.add_argument(
        "--ci",
        action="store_true",
        help="Output JSON for CI pipelines",
    )


class CheckpointCommand:
    """Checkpoint management CLI command."""

    @staticmethod
    def register(subparsers: Any) -> None:
        """Register the checkpoint command with argument parser."""
        parser = subparsers.add_parser(
            "checkpoint",
            help="Checkpoint management and frequency advisor",
            description="Manage checkpoints, inspect metadata, and get frequency recommendations",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  torchbridge checkpoint info ./checkpoints/checkpoint_20260219
  torchbridge checkpoint list ./checkpoints
  torchbridge checkpoint advisor --world-size 64
  torchbridge checkpoint advisor --world-size 256 --checkpoint-time 120 --ci
            """,
        )

        sub = parser.add_subparsers(dest="checkpoint_action", metavar="<action>")
        _register_subcommands(sub)

    @staticmethod
    def execute(args: Any) -> int:
        """Execute the checkpoint command."""
        action = getattr(args, "checkpoint_action", None)

        if action == "info":
            return _show_info(args.path)
        if action == "list":
            return _list_checkpoints(args.path, ci=getattr(args, "ci", False))
        if action == "advisor":
            return _show_advisor(args)

        # No subcommand — show help
        print("Usage: torchbridge checkpoint <info|list|advisor>")
        print("Run 'torchbridge checkpoint --help' for details.")
        return 1


def _show_info(path: str) -> int:
    """Show checkpoint metadata."""
    from pathlib import Path

    from torchbridge.checkpoint.manager import _METADATA_FILENAME
    from torchbridge.checkpoint.metadata import CheckpointMetadata

    meta_path = Path(path) / _METADATA_FILENAME
    if not meta_path.exists():
        print(f"No metadata found at {meta_path}")
        return 1

    metadata = CheckpointMetadata.load(meta_path)

    print("Checkpoint Metadata")
    print("=" * 50)
    print(f"  ID:              {metadata.checkpoint_id}")
    print(f"  Timestamp:       {metadata.timestamp}")
    print(f"  TorchBridge:     {metadata.torchbridge_version}")
    print(f"  PyTorch:         {metadata.pytorch_version}")
    print(f"  Backend:         {metadata.backend}")
    print(f"  Architecture:    {metadata.architecture or 'N/A'}")
    print(f"  World size:      {metadata.world_size}")
    print(f"  Local world:     {metadata.local_world_size}")
    if metadata.model_params:
        print(f"  Model params:    {metadata.model_params / 1e9:.1f}B")
    if metadata.dtype_map:
        unique_dtypes = set(metadata.dtype_map.values())
        print(f"  Dtypes:          {', '.join(sorted(unique_dtypes))}")
    print()
    return 0


def _list_checkpoints(path: str, ci: bool = False) -> int:
    """List checkpoints in a directory."""
    from torchbridge.checkpoint.config import CheckpointConfig
    from torchbridge.checkpoint.manager import CheckpointManager

    config = CheckpointConfig(storage_path=path)
    manager = CheckpointManager(config=config)
    checkpoints = manager.list_checkpoints()

    if not checkpoints:
        if ci:
            json.dump([], sys.stdout)
            print()
        else:
            print(f"No checkpoints found in {path}")
        return 0

    if ci:
        output = [m.to_dict() for m in checkpoints]
        json.dump(output, sys.stdout, indent=2)
        print()
        return 0

    print(f"Checkpoints in {path}")
    print("=" * 60)
    for i, meta in enumerate(checkpoints, 1):
        params = (
            f"{meta.model_params / 1e9:.1f}B"
            if meta.model_params
            else "N/A"
        )
        print(
            f"  {i}. {meta.checkpoint_id}"
        )
        print(
            f"     {meta.timestamp} | {meta.backend} | "
            f"world={meta.world_size} | params={params}"
        )
    print(f"\nTotal: {len(checkpoints)} checkpoints")
    return 0


def _show_advisor(args: Any) -> int:
    """Show checkpoint frequency recommendation."""
    from torchbridge.checkpoint.frequency import CheckpointFrequencyAdvisor

    advisor = CheckpointFrequencyAdvisor()
    rec = advisor.recommend(
        world_size=args.world_size,
        checkpoint_time_seconds=args.checkpoint_time,
        step_time_seconds=args.step_time,
        mtbf_hours=args.mtbf,
    )

    if args.ci:
        output = {
            "world_size": args.world_size,
            "checkpoint_time_seconds": args.checkpoint_time,
            "step_time_seconds": args.step_time,
            "recommendation": rec.to_dict(),
        }
        json.dump(output, sys.stdout, indent=2)
        print()
        return 0

    print("Checkpoint Frequency Recommendation")
    print("=" * 50)
    print()
    print(f"  Cluster size:     {args.world_size} GPUs")
    print(f"  Checkpoint time:  {args.checkpoint_time:.0f}s")
    print(f"  Step time:        {args.step_time:.1f}s")
    print()
    print("Recommendation")
    print("-" * 50)
    print(f"  Interval:         {rec.interval_minutes} minutes")
    if rec.interval_steps:
        print(f"  Interval:         ~{rec.interval_steps} steps")
    print(f"  Risk level:       {rec.risk_level}")
    print(f"  I/O overhead:     {rec.optimal_overhead_pct:.1f}%")
    print()
    print("Reasoning")
    print("-" * 50)
    print(f"  {rec.reasoning}")
    print()
    return 0


def main(args: list[str] | None = None) -> int:
    """Entry point for tb-checkpoint."""
    parser = argparse.ArgumentParser(
        prog="tb-checkpoint",
        description="Checkpoint management and frequency advisor",
    )

    sub = parser.add_subparsers(dest="checkpoint_action", metavar="<action>")
    _register_subcommands(sub)

    if args is None:
        args = sys.argv[1:]

    parsed = parser.parse_args(args)

    if not parsed.checkpoint_action:
        parser.print_help()
        return 1

    return CheckpointCommand.execute(parsed)
