"""
Main entry point for the KernelPyTorch CLI when run as a module.

This allows running the CLI via:
    python -m kernel_pytorch.cli
"""

import sys

from . import main

if __name__ == '__main__':
    sys.exit(main())
