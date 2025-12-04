"""
KernelPyTorch Command Line Interface

Professional CLI tools for PyTorch optimization, benchmarking, and system validation.
Provides easy-to-use commands for model optimization and performance analysis.
"""

import argparse
import sys
from typing import List, Optional

from .optimize import OptimizeCommand
from .benchmark import BenchmarkCommand
from .doctor import DoctorCommand


def main(args: Optional[List[str]] = None) -> int:
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
  kernelpytorch doctor --full-report

For command-specific help:
  kernelpytorch <command> --help
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.56'
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
    DoctorCommand.register(subparsers)

    # Parse arguments
    if args is None:
        args = sys.argv[1:]

    parsed_args = parser.parse_args(args)

    # Execute command
    if not parsed_args.command:
        parser.print_help()
        return 1

    try:
        if parsed_args.command == 'optimize':
            return OptimizeCommand.execute(parsed_args)
        elif parsed_args.command == 'benchmark':
            return BenchmarkCommand.execute(parsed_args)
        elif parsed_args.command == 'doctor':
            return DoctorCommand.execute(parsed_args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())