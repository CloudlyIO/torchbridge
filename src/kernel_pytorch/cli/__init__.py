"""
KernelPyTorch Command Line Interface

Professional CLI tools for PyTorch optimization, benchmarking, profiling,
model export, and system validation.
"""

import argparse
import sys
from typing import Optional  # noqa: F401

from .benchmark import BenchmarkCommand
from .doctor import DoctorCommand
from .export import ExportCommand
from .optimize import OptimizeCommand
from .profile import ProfileCommand


def main(args: list[str] | None = None) -> int:
    """
    Main entry point for the kernelpytorch CLI.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        prog='kernelpytorch',
        description='KernelPyTorch: Production-grade PyTorch optimization CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kernelpytorch optimize --model model.pt --level production
  kernelpytorch benchmark --model bert-base-uncased --quick
  kernelpytorch export --model model.pt --format onnx
  kernelpytorch profile --model model.pt --mode summary
  kernelpytorch doctor --full-report

For command-specific help:
  kernelpytorch <command> --help
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.4.30'
    )

    # Add subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='<command>'
    )

    # Register commands
    OptimizeCommand.register(subparsers)
    BenchmarkCommand.register(subparsers)
    ExportCommand.register(subparsers)
    ProfileCommand.register(subparsers)
    DoctorCommand.register(subparsers)

    # Parse arguments
    if args is None:
        args = sys.argv[1:]

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as e:
        # Convert argparse SystemExit to our return code convention
        return 1 if e.code == 2 else e.code

    # Execute command
    if not parsed_args.command:
        parser.print_help()
        return 1

    try:
        if parsed_args.command == 'optimize':
            return OptimizeCommand.execute(parsed_args)
        elif parsed_args.command == 'benchmark':
            return BenchmarkCommand.execute(parsed_args)
        elif parsed_args.command == 'export':
            return ExportCommand.execute(parsed_args)
        elif parsed_args.command == 'profile':
            return ProfileCommand.execute(parsed_args)
        elif parsed_args.command == 'doctor':
            return DoctorCommand.execute(parsed_args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
