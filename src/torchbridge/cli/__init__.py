"""
TorchBridge Command Line Interface

Professional CLI tools for cross-backend PyTorch support, benchmarking, profiling,
model export, and system validation.
"""

import argparse
import logging
import os
import sys
import warnings
# Suppress noisy warnings before any imports that trigger them
# These are informational, not errors - we report hardware status via 'doctor' command
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
warnings.filterwarnings("ignore", message=".*Transformer Engine not available.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchbridge.precision")

# Suppress PyTorch distributed elastic logging noise on non-Linux platforms
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)

# Suppress pynvml deprecation warning
os.environ.setdefault("PYTORCH_NVML_SUPPRESS_DEPRECATION_WARNING", "1")

from torchbridge import __version__ as _tb_version
from torchbridge.core.errors import TorchBridgeError

from .benchmark import BenchmarkCommand
from .cache import CacheCommand
from .doctor import DoctorCommand
from .export import ExportCommand
from .init import InitCommand
from .migrate import MigrateCommand
from .optimize import OptimizeCommand
from .profile import ProfileCommand
from .quantize import QuantizeCommand
from .validate import ValidateCommand


def _print_error(exc: Exception, verbose: bool = False) -> int:
    """Format and print an error with hints if available."""
    if isinstance(exc, TorchBridgeError):
        print(f"Error: {exc.message}")
        if exc.hint:
            print(f"  Hint: {exc.hint}")
        if exc.details:
            for key, value in exc.details.items():
                print(f"  {key}: {value}")
        if exc.cause:
            print(f"  Caused by: {type(exc.cause).__name__}: {exc.cause}")
    else:
        print(f"Error: {exc}")

    if verbose:
        import traceback
        traceback.print_exc()

    return 1


def main(args: list[str] | None = None) -> str | int | None:
    """
    Main entry point for the torchbridge CLI.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        prog='torchbridge',
        description='TorchBridge: Production-grade PyTorch hardware abstraction CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  torchbridge optimize --model model.pt --level production
  torchbridge benchmark --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --quick
  torchbridge export --model model.pt --format onnx
  torchbridge profile --model model.pt --mode summary
  torchbridge doctor --full-report

For command-specific help:
  torchbridge <command> --help
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {_tb_version}'
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
    InitCommand.register(subparsers)
    ValidateCommand.register(subparsers)
    QuantizeCommand.register(subparsers)
    MigrateCommand.register(subparsers)
    CacheCommand.register(subparsers)

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

    commands = {
        'optimize': OptimizeCommand,
        'benchmark': BenchmarkCommand,
        'export': ExportCommand,
        'profile': ProfileCommand,
        'doctor': DoctorCommand,
        'init': InitCommand,
        'validate': ValidateCommand,
        'quantize': QuantizeCommand,
        'migrate': MigrateCommand,
        'cache': CacheCommand,
    }

    cmd_class = commands.get(parsed_args.command)
    if cmd_class is None:
        parser.print_help()
        return 1

    try:
        return cmd_class.execute(parsed_args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        return _print_error(e, verbose=getattr(parsed_args, 'verbose', False))


if __name__ == '__main__':
    sys.exit(main())
