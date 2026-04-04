"""
Milestone 5 batch execution wrapper over the existing CMSVS pipeline.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .configuration import BatchJobConfig, BatchJobConfigParser, BatchPair
from .discovery import collect_pairs


@dataclass(slots=True)
class PairRunResult:
    """Execution summary for one document pair."""

    pair_id: str
    status: str
    output_path: str | None
    report_format: str
    ground_truth_path: str | None
    doc_a_path: str
    doc_b_path: str
    error: str | None = None
    metadata: dict[str, Any] | None = None


class Milestone5Pipeline:
    """Run configuration-driven M5 batch jobs and write system outputs."""

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.src_root = self.repo_root / "src"
        self._pipeline_cache: dict[tuple[str, float | None, int, int], Any] = {}

        if str(self.src_root) not in sys.path:
            sys.path.insert(0, str(self.src_root))

    def run_from_manifest(self, manifest_path: str | Path) -> dict[str, Any]:
        """Load a manifest and execute the declared batch job."""
        config = BatchJobConfigParser().load(manifest_path)
        return self.run(config=config, manifest_path=manifest_path)

    def run(
        self,
        config: BatchJobConfig,
        manifest_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Execute a batch job and write per-pair outputs plus a batch index."""
        pairs = collect_pairs(config, self.repo_root)
        output_dir = config.resolved_output_dir(self.repo_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[PairRunResult] = []
        for pair in pairs:
            try:
                pipeline = self._get_or_create_pipeline(config)
                report = self._run_pair(pipeline, pair, config.report_format)
                output_path = output_dir / self._resolve_output_name(pair)
                saved_path = pipeline.save_report(report, output_path)

                results.append(
                    PairRunResult(
                        pair_id=pair.pair_id,
                        status="success",
                        output_path=str(saved_path),
                        report_format=config.report_format,
                        ground_truth_path=pair.ground_truth_path,
                        doc_a_path=pair.doc_a_path,
                        doc_b_path=pair.doc_b_path,
                        metadata=dict(pair.metadata),
                    )
                )
            except Exception as exc:
                results.append(
                    PairRunResult(
                        pair_id=pair.pair_id,
                        status="failed",
                        output_path=None,
                        report_format=config.report_format,
                        ground_truth_path=pair.ground_truth_path,
                        doc_a_path=pair.doc_a_path,
                        doc_b_path=pair.doc_b_path,
                        error=str(exc),
                        metadata=dict(pair.metadata),
                    )
                )
                if config.fail_fast:
                    break

        summary = self._build_batch_summary(
            config=config,
            pairs=pairs,
            results=results,
            output_dir=output_dir,
            manifest_path=manifest_path,
        )
        self._write_batch_summary(output_dir / "batch_summary.json", summary)
        return summary

    def _get_or_create_pipeline(self, config: BatchJobConfig):
        from config.config_parser import CMSVSConfigParser
        from models.nvidia_client import NvidiaEmbeddingClient, NvidiaLLMClient
        from pipeline.cmsvs_pipeline import CMSVSPipeline

        config_path = config.resolved_config_path(self.repo_root)
        cache_key = (
            str(config_path),
            config.confidence_threshold,
            config.top_k,
            config.fallback_top_k,
        )
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        import os

        nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
        if not nvidia_key:
            raise EnvironmentError("NVIDIA_API_KEY environment variable is not set.")

        parsed_config = CMSVSConfigParser().load(config_path)
        threshold = (
            config.confidence_threshold
            if config.confidence_threshold is not None
            else parsed_config.validation_settings.confidence_threshold
        )

        pipeline = CMSVSPipeline(
            config=parsed_config,
            embedding_client=NvidiaEmbeddingClient(api_key=nvidia_key),
            llm_client=NvidiaLLMClient(api_key=nvidia_key),
            confidence_threshold=threshold,
            default_top_k=config.top_k,
            fallback_top_k=config.fallback_top_k,
        )
        self._pipeline_cache[cache_key] = pipeline
        return pipeline

    @staticmethod
    def _run_pair(pipeline, pair: BatchPair, report_format: str) -> dict[str, Any]:
        if report_format == "groundtruth":
            return pipeline.run_groundtruth_format(
                doc_a_path=pair.doc_a_path,
                doc_b_path=pair.doc_b_path,
                doc_a_name=pair.doc_a_name,
                doc_b_name=pair.doc_b_name,
            )
        return pipeline.run(
            doc_a_path=pair.doc_a_path,
            doc_b_path=pair.doc_b_path,
            doc_a_name=pair.doc_a_name,
            doc_b_name=pair.doc_b_name,
        )

    @staticmethod
    def _resolve_output_name(pair: BatchPair) -> str:
        return pair.output_name or f"{pair.pair_id}.json"

    @staticmethod
    def _build_batch_summary(
        config: BatchJobConfig,
        pairs: list[BatchPair],
        results: list[PairRunResult],
        output_dir: Path,
        manifest_path: str | Path | None,
    ) -> dict[str, Any]:
        success_count = sum(1 for result in results if result.status == "success")
        failure_count = len(results) - success_count

        return {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "batch_name": config.batch_name,
            "manifest_path": str(Path(manifest_path).resolve()) if manifest_path else None,
            "config_path": str(config.config_path),
            "output_dir": str(output_dir.resolve()),
            "report_format": config.report_format,
            "confidence_threshold": config.confidence_threshold,
            "top_k": config.top_k,
            "fallback_top_k": config.fallback_top_k,
            "total_pairs_requested": len(pairs),
            "total_pairs_processed": len(results),
            "success_count": success_count,
            "failure_count": failure_count,
            "results": [
                {
                    "pair_id": result.pair_id,
                    "status": result.status,
                    "output_path": result.output_path,
                    "report_format": result.report_format,
                    "ground_truth_path": result.ground_truth_path,
                    "doc_a_path": result.doc_a_path,
                    "doc_b_path": result.doc_b_path,
                    "error": result.error,
                    "metadata": result.metadata or {},
                }
                for result in results
            ],
        }

    @staticmethod
    def _write_batch_summary(output_path: Path, summary: dict[str, Any]) -> None:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
