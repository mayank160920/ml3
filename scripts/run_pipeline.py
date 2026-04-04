"""
CLI entry point for the CMSVS pipeline.

Usage
-----
    python scripts/run_pipeline.py \\
        --doc_a  data/healthcare_sbc/doc_a_sbc/plan_a.pdf \\
        --doc_b  data/healthcare_sbc/doc_b_benefit_grids/unaugmented/grid_a.pdf \\
        --config configs/healthcare_sbc_config.yaml \\
        --output output/report.json

    python scripts/run_pipeline.py \\
        --doc_a  data/funsd/original/doc_a.png \\
        --doc_b  data/funsd/augmented/doc_b.png \\
        --config configs/funsd_ner_config.yaml \\
        --output output/funsd_report.json \\
        --groundtruth_format

Environment variables required
--------------------------------
    NVIDIA_API_KEY : NVIDIA NIM API key
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.cmsvs_pipeline import CMSVSPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CMSVS — Configurable Multimodal Semantic Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--doc_a",
        required=True,
        help="Path to Document A (PDF or image)",
    )
    parser.add_argument(
        "--doc_b",
        required=True,
        help="Path to Document B (PDF or image)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--output",
        default="output/report.json",
        help="Output path for the validation report JSON (default: output/report.json)",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.75,
        help="Confidence threshold for review flagging (default: 0.75)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Number of pages to retrieve per section (default: 2)",
    )
    parser.add_argument(
        "--fallback_top_k",
        type=int,
        default=4,
        help="Expanded pages for fallback retrieval (default: 4)",
    )
    parser.add_argument(
        "--groundtruth_format",
        action="store_true",
        help="Output in M2 ground truth JSON format for M5 evaluation",
    )
    parser.add_argument(
        "--doc_a_name",
        default=None,
        help="Human-readable label for Document A (defaults to filename)",
    )
    parser.add_argument(
        "--doc_b_name",
        default=None,
        help="Human-readable label for Document B (defaults to filename)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args, run pipeline, save report."""
    args = parse_args()

    # ── Validate environment ──────────────────────────────────────────────────
    if not os.environ.get("NVIDIA_API_KEY"):
        print(
            "ERROR: NVIDIA_API_KEY environment variable is not set.\n"
            "Export it before running:\n"
            "  export NVIDIA_API_KEY=nvapi-...",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Validate input files ──────────────────────────────────────────────────
    for label, path in [("doc_a", args.doc_a), ("doc_b", args.doc_b)]:
        if not Path(path).exists():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # ── Build and run pipeline ────────────────────────────────────────────────
    print(f"Loading config: {args.config}")
    pipeline = CMSVSPipeline.from_env(
        config_path=args.config,
        confidence_threshold=args.confidence_threshold,
        default_top_k=args.top_k,
        fallback_top_k=args.fallback_top_k,
    )

    print(f"Processing:\n  Doc A: {args.doc_a}\n  Doc B: {args.doc_b}")

    if args.groundtruth_format:
        report = pipeline.run_groundtruth_format(
            doc_a_path=args.doc_a,
            doc_b_path=args.doc_b,
            doc_a_name=args.doc_a_name,
            doc_b_name=args.doc_b_name,
        )
    else:
        report = pipeline.run(
            doc_a_path=args.doc_a,
            doc_b_path=args.doc_b,
            doc_a_name=args.doc_a_name,
            doc_b_name=args.doc_b_name,
        )

    # ── Save report ───────────────────────────────────────────────────────────
    saved_path = pipeline.save_report(report, args.output)
    print(f"\nReport saved → {saved_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    if not args.groundtruth_format:
        summary = report.get("summary", {})
        print(
            f"\nSummary:\n"
            f"  Total entities  : {summary.get('total_entities', 0)}\n"
            f"  Matches         : {summary.get('total_matches', 0)}\n"
            f"  Mismatches      : {summary.get('total_mismatches', 0)}\n"
            f"  Match rate      : {summary.get('match_rate', 0.0):.1%}\n"
            f"  Review required : {summary.get('review_required', 0)}\n"
            f"  Fast-path       : {summary.get('fast_path_matches', 0)}"
        )
    else:
        entities = report.get("entities", [])
        matches = sum(1 for e in entities if e.get("validation_result") == "match")
        print(
            f"\nGround truth format summary:\n"
            f"  Total entities : {len(entities)}\n"
            f"  Matches        : {matches}\n"
            f"  Mismatches     : {len(entities) - matches}"
        )


if __name__ == "__main__":
    main()
