"""
Dataset pair discovery for Milestone 5 evaluation jobs.
"""
from __future__ import annotations

from pathlib import Path

from .configuration import BatchJobConfig, BatchPair, PairCollectionConfig


def collect_pairs(config: BatchJobConfig, repo_root: Path) -> list[BatchPair]:
    """Return all resolved pairs for a batch job."""
    pairs: list[BatchPair] = []

    if config.pair_collection is not None:
        pairs.extend(_collect_from_dataset(config.pair_collection, repo_root))

    for pair in config.pairs:
        doc_a_path = _resolve_pair_path(repo_root, pair.doc_a_path)
        doc_b_path = _resolve_pair_path(repo_root, pair.doc_b_path)
        ground_truth_path = (
            _resolve_pair_path(repo_root, pair.ground_truth_path)
            if pair.ground_truth_path
            else None
        )

        _assert_exists(doc_a_path)
        _assert_exists(doc_b_path)
        if ground_truth_path:
            _assert_exists(ground_truth_path)

        pairs.append(
            BatchPair(
                pair_id=pair.pair_id,
                doc_a_path=str(doc_a_path),
                doc_b_path=str(doc_b_path),
                ground_truth_path=str(ground_truth_path) if ground_truth_path else None,
                doc_a_name=pair.doc_a_name,
                doc_b_name=pair.doc_b_name,
                output_name=pair.output_name,
                metadata=dict(pair.metadata),
            )
        )

    return pairs


def _collect_from_dataset(
    collection: PairCollectionConfig,
    repo_root: Path,
) -> list[BatchPair]:
    kind = collection.kind.lower()
    if kind == "sbc":
        return _collect_sbc_pairs(collection, repo_root)
    if kind == "funsd":
        return _collect_funsd_pairs(collection, repo_root)
    raise ValueError(f"Unsupported pair collection kind: {collection.kind}")


def _collect_sbc_pairs(
    collection: PairCollectionConfig,
    repo_root: Path,
) -> list[BatchPair]:
    doc_a_dir = _resolve_pair_path(repo_root, collection.doc_a_dir)
    doc_b_dir = _resolve_pair_path(repo_root, collection.doc_b_dir)
    gt_dir = (
        _resolve_pair_path(repo_root, collection.ground_truth_dir)
        if collection.ground_truth_dir
        else None
    )

    pair_ids = collection.pair_ids or _discover_sbc_pair_ids(gt_dir)
    resolved_pairs: list[BatchPair] = []

    for pair_id in pair_ids:
        index = _parse_sbc_index(pair_id)
        canonical_pair_id = f"sbc_{index:03d}"
        doc_a_path = doc_a_dir / f"sbc ({index}).pdf"
        doc_b_path = doc_b_dir / f"SBCBG{index}.pdf"
        ground_truth_path = (
            gt_dir / f"{canonical_pair_id}_ground_truth.json" if gt_dir else None
        )

        _assert_exists(doc_a_path)
        _assert_exists(doc_b_path)
        if ground_truth_path:
            _assert_exists(ground_truth_path)

        resolved_pairs.append(
            BatchPair(
                pair_id=canonical_pair_id,
                doc_a_path=str(doc_a_path),
                doc_b_path=str(doc_b_path),
                ground_truth_path=str(ground_truth_path) if ground_truth_path else None,
                doc_a_name=f"{canonical_pair_id}_doc_a",
                doc_b_name=f"{canonical_pair_id}_doc_b",
                output_name=f"{canonical_pair_id}.json",
                metadata={"dataset": "SBC", "pair_index": index},
            )
        )

    return resolved_pairs


def _collect_funsd_pairs(
    collection: PairCollectionConfig,
    repo_root: Path,
) -> list[BatchPair]:
    doc_a_dir = _resolve_pair_path(repo_root, collection.doc_a_dir)
    doc_b_dir = _resolve_pair_path(repo_root, collection.doc_b_dir)
    gt_dir = (
        _resolve_pair_path(repo_root, collection.ground_truth_dir)
        if collection.ground_truth_dir
        else None
    )

    if collection.pair_ids:
        stems = [pair_id.replace("_gt", "").replace("_aug", "") for pair_id in collection.pair_ids]
    else:
        stems = sorted(path.stem for path in doc_a_dir.glob("*.png"))

    resolved_pairs: list[BatchPair] = []

    for stem in stems:
        doc_a_path = doc_a_dir / f"{stem}.png"
        doc_b_path = doc_b_dir / f"{stem}_aug.png"
        ground_truth_path = gt_dir / f"{stem}_gt.json" if gt_dir else None

        _assert_exists(doc_a_path)
        _assert_exists(doc_b_path)
        if ground_truth_path:
            _assert_exists(ground_truth_path)

        resolved_pairs.append(
            BatchPair(
                pair_id=stem,
                doc_a_path=str(doc_a_path),
                doc_b_path=str(doc_b_path),
                ground_truth_path=str(ground_truth_path) if ground_truth_path else None,
                doc_a_name=f"{stem}_original",
                doc_b_name=f"{stem}_augmented",
                output_name=f"{stem}.json",
                metadata={"dataset": "FUNSD"},
            )
        )

    return resolved_pairs


def _discover_sbc_pair_ids(ground_truth_dir: Path | None) -> list[str]:
    if ground_truth_dir is None:
        raise ValueError(
            "SBC pair discovery requires 'ground_truth_dir' when pair_ids are omitted."
        )
    return sorted(
        path.stem.replace("_ground_truth", "")
        for path in ground_truth_dir.glob("sbc_*_ground_truth.json")
    )


def _parse_sbc_index(pair_id: str) -> int:
    token = pair_id
    if pair_id.startswith("sbc_"):
        token = pair_id.split("_", 1)[1]
    return int(token)


def _resolve_pair_path(repo_root: Path, path_value: str | None) -> Path:
    if path_value is None:
        raise ValueError("Expected a non-null path value")
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required batch input not found: {path}")
