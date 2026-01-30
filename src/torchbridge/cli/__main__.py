"""
Main entry point for the TorchBridge CLI when run as a module.

This allows running the CLI via:
    python -m torchbridge.cli
"""

import sys

from . import main

if __name__ == '__main__':
    sys.exit(main())
