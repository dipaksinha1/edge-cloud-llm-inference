"""The argument parser shared by every `python -m eclm.*` command (ARCHITECTURE §5)."""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn


class ArgumentParser(argparse.ArgumentParser):
    """Usage errors print the usage and exit 1 (exit 2 is reserved for 'budget reached')."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")
