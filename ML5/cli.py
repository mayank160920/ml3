"""
CLI entry point for Milestone 5 batch execution.

Examples
--------
python ML5/cli.py --manifest ML5/configs/sbc_batch.yaml
python ML5/cli.py --manifest ML5/configs/funsd_batch.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ML5.pipeline import Milestone5Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Milestone 5 batch jobs over the existing CMSVS pipeline."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the ML5 batch manifest YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    summary = Milestone5Pipeline(repo_root=repo_root).run_from_manifest(args.manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
