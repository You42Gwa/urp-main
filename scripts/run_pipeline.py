"""CLI entry point for the multimodal image-gate MVP."""

from __future__ import annotations

import argparse
import sys

from mm_gate.collect import main as collect_main
from mm_gate.download import main as download_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal image-gate pipeline")
    parser.add_argument("stage", choices=("collect", "download"))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.stage == "collect":
        collect_main()
    else:
        download_main()


if __name__ == "__main__":
    main()
