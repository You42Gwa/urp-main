"""CLI entry point for the multimodal image-gate MVP."""

from __future__ import annotations

import argparse
import sys

from mm_gate.collect import main as collect_main
from mm_gate.caption import main as caption_main
from mm_gate.download import main as download_main
from mm_gate.enrich import main as enrich_main
from mm_gate.gate import main as gate_main
from mm_gate.kg import main as kg_main
from mm_gate.report import main as report_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal image-gate pipeline")
    parser.add_argument("stage", choices=("collect", "download", "enrich", "caption", "gate", "kg", "report"))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.stage == "collect":
        collect_main()
    elif args.stage == "download":
        download_main()
    elif args.stage == "caption":
        caption_main()
    elif args.stage == "gate":
        gate_main()
    elif args.stage == "kg":
        kg_main()
    elif args.stage == "report":
        report_main()
    else:
        enrich_main()


if __name__ == "__main__":
    main()
