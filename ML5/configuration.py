"""
Configuration models and parser for Milestone 5 batch jobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class BatchPair:
    """Explicit document pair definition for a batch job."""

    pair_id: str
    doc_a_path: str
    doc_b_path: str
    ground_truth_path: str | None = None
    doc_a_name: str | None = None
    doc_b_name: str | None = None
    output_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PairCollectionConfig:
    """Dataset-specific pair discovery settings."""

    kind: str
    doc_a_dir: str
    doc_b_dir: str
    ground_truth_dir: str | None = None
    pair_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BatchJobConfig:
    """Top-level batch job configuration."""

    batch_name: str
    config_path: str
    output_dir: str
    report_format: str = "groundtruth"
    confidence_threshold: float | None = None
    top_k: int = 2
    fallback_top_k: int = 4
    fail_fast: bool = False
    pair_collection: PairCollectionConfig | None = None
    pairs: list[BatchPair] = field(default_factory=list)

    def resolved_config_path(self, repo_root: Path) -> Path:
        return _resolve_path(repo_root, self.config_path)

    def resolved_output_dir(self, repo_root: Path) -> Path:
        return _resolve_path(repo_root, self.output_dir)


def _resolve_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


class BatchJobConfigParser:
    """Load Milestone 5 batch job configuration from YAML."""

    def load(self, config_path: str | Path) -> BatchJobConfig:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Batch config not found: {path}")

        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        self._validate_top_level(raw, path)

        collection = None
        if raw.get("pair_collection"):
            collection = self._parse_collection(raw["pair_collection"])

        pairs = [self._parse_pair(item) for item in raw.get("pairs", [])]
        if not collection and not pairs:
            raise ValueError(
                f"Batch config '{path}' must define either 'pair_collection' or 'pairs'."
            )

        report_format = str(raw.get("report_format", "groundtruth")).lower()
        if report_format not in {"groundtruth", "full"}:
            raise ValueError(
                f"Unsupported report_format '{report_format}' in {path}. "
                "Use 'groundtruth' or 'full'."
            )

        top_k = int(raw.get("top_k", 2))
        fallback_top_k = int(raw.get("fallback_top_k", 4))
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1 in {path}")
        if fallback_top_k < top_k:
            raise ValueError(
                f"fallback_top_k ({fallback_top_k}) must be >= top_k ({top_k}) in {path}"
            )

        confidence_threshold = raw.get("confidence_threshold")
        if confidence_threshold is not None:
            confidence_threshold = float(confidence_threshold)
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError(
                    f"confidence_threshold must be in [0, 1] in {path}"
                )

        return BatchJobConfig(
            batch_name=str(raw["batch_name"]),
            config_path=str(raw["config_path"]),
            output_dir=str(raw["output_dir"]),
            report_format=report_format,
            confidence_threshold=confidence_threshold,
            top_k=top_k,
            fallback_top_k=fallback_top_k,
            fail_fast=bool(raw.get("fail_fast", False)),
            pair_collection=collection,
            pairs=pairs,
        )

    @staticmethod
    def _validate_top_level(raw: dict[str, Any], path: Path) -> None:
        required = {"batch_name", "config_path", "output_dir"}
        missing = required - set(raw.keys())
        if missing:
            raise ValueError(f"Missing required keys in {path}: {sorted(missing)}")

    @staticmethod
    def _parse_collection(raw: dict[str, Any]) -> PairCollectionConfig:
        required = {"kind", "doc_a_dir", "doc_b_dir"}
        missing = required - set(raw.keys())
        if missing:
            raise ValueError(
                f"Missing required pair_collection keys: {sorted(missing)}"
            )
        return PairCollectionConfig(
            kind=str(raw["kind"]).lower(),
            doc_a_dir=str(raw["doc_a_dir"]),
            doc_b_dir=str(raw["doc_b_dir"]),
            ground_truth_dir=(
                str(raw["ground_truth_dir"])
                if raw.get("ground_truth_dir") is not None
                else None
            ),
            pair_ids=[str(item) for item in raw.get("pair_ids", [])],
        )

    @staticmethod
    def _parse_pair(raw: dict[str, Any]) -> BatchPair:
        required = {"pair_id", "doc_a_path", "doc_b_path"}
        missing = required - set(raw.keys())
        if missing:
            raise ValueError(f"Missing required pair keys: {sorted(missing)}")
        return BatchPair(
            pair_id=str(raw["pair_id"]),
            doc_a_path=str(raw["doc_a_path"]),
            doc_b_path=str(raw["doc_b_path"]),
            ground_truth_path=(
                str(raw["ground_truth_path"])
                if raw.get("ground_truth_path") is not None
                else None
            ),
            doc_a_name=str(raw["doc_a_name"]) if raw.get("doc_a_name") else None,
            doc_b_name=str(raw["doc_b_name"]) if raw.get("doc_b_name") else None,
            output_name=str(raw["output_name"]) if raw.get("output_name") else None,
            metadata=dict(raw.get("metadata", {})),
        )
