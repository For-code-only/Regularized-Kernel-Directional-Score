#!/usr/bin/env python3
"""Command-line entry point for the experiment suites."""
import sys
import traceback

from src.runner import main


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
