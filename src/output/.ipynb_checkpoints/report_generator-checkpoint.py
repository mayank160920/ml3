"""
--------------------------------
Generates structured JSON validation reports from ValidationReport objects.

The output schema is designed for direct comparison against M2 ground
truth JSON files, enabling automated M5 evaluation metrics:
  - Precision, recall, F1 per entity
  - Per-scenario-type accuracy
  - Per-section validation accuracy
  - Human review escalation rate
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared_types import (
    FinalEntityValue,
    ValidationReport,
    ValidationStatus,
)


class ReportGenerator:
    """
    Serialises pipeline outputs to structured JSON reports.

    Usage
    -----
    >>> generator = ReportGenerator()
    >>> report_dict = generator.generate(
    ...     validation_report=report,
    ...     doc_a_entities=doc_a_values,
    ...     doc_b_entities=doc_b_values,
    ... )
    >>> generator.save(report_dict, output_path="output/report.json")
    """

    def generate(
        self,
        validation_report: ValidationReport,
        doc_a_entities: dict[str, FinalEntityValue],
        doc_b_entities: dict[str, FinalEntityValue],
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """
        Build the complete report dictionary.

        Parameters
        ----------
        validation_report : ValidationReport from Layer 7
        doc_a_entities    : extracted entities for Document A
        doc_b_entities    : extracted entities for Document B
        metadata          : optional additional metadata (e.g. run_id, model)

        Returns
        -------
        dict : fully serialisable report
        """
        summary = self._build_summary(validation_report)
        entity_extractions = self._build_extractions(
            doc_a_entities, doc_b_entities
        )
        validation_results = self._build_validation_results(validation_report)

        report: dict[str, Any] = {
            "schema_version":    "1.0",
            "generated_at":      datetime.now(timezone.utc).isoformat(),
            "config_name":       validation_report.config_name,
            "doc_a_path":        validation_report.doc_a_path,
            "doc_b_path":        validation_report.doc_b_path,
            "summary":           summary,
            "entity_extractions": entity_extractions,
            "validation_results": validation_results,
        }

        if metadata:
            report["metadata"] = metadata

        return report

    def save(
        self,
        report: dict,
        output_path: str | Path,
        indent: int = 2,
    ) -> Path:
        """
        Save the report dictionary to a JSON file.

        Parameters
        ----------
        report      : report dictionary from generate()
        output_path : destination file path
        indent      : JSON indentation level

        Returns
        -------
        Path : absolute path to the saved file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=indent, default=str)
        return path.resolve()

    def generate_groundtruth_format(
        self,
        validation_report: ValidationReport,
    ) -> dict:
        """
        Generate output in the M2 ground truth JSON format for M5 evaluation.

        Format matches the dataset ground truth schema:
          { "entities": [ { entity_name, doc_a_value, doc_b_value,
                            normalized_value, validation_type,
                            validation_result } ] }

        Parameters
        ----------
        validation_report : ValidationReport from Layer 7

        Returns
        -------
        dict matching M2 ground truth schema
        """
        entities = []
        for section in validation_report.section_results:
            for result in section.entity_results:
                status_lower = result.validation_status.value.lower()

                # Map to ground truth validation_result vocabulary
                if result.validation_status == ValidationStatus.MATCH:
                    gt_result = "match"
                elif result.validation_status == ValidationStatus.MISMATCH:
                    gt_result = "mismatch"
                elif result.validation_status == ValidationStatus.PARTIAL_MATCH:
                    gt_result = "partial_match"
                else:
                    gt_result = "ineligible"

                # Map to ground truth validation_type vocabulary
                if result.fast_path_match:
                    gt_type = "exact_match"
                elif result.validation_status == ValidationStatus.MATCH:
                    gt_type = "semantic_match"
                elif result.validation_status == ValidationStatus.MISMATCH:
                    gt_type = "conflict"
                else:
                    gt_type = "semantic_match"

                entities.append({
                    "entity_name":       result.entity_name,
                    "doc_a_value":       result.doc_a_value,
                    "doc_b_value":       result.doc_b_value,
                    "normalized_value":  result.doc_a_normalized,
                    "validation_type":   gt_type,
                    "validation_result": gt_result,
                })

        return {"entities": entities}

    # ── private ───────────────────────────────────────────────────────────────

    def _build_summary(self, report: ValidationReport) -> dict:
        """Build high-level summary statistics."""
        total = report.total_entities
        matches = report.total_matches
        mismatches = report.total_mismatches
        review_count = sum(
            1
            for s in report.section_results
            for r in s.entity_results
            if r.review_required
        )
        fast_path_count = sum(
            1
            for s in report.section_results
            for r in s.entity_results
            if r.fast_path_match
        )

        return {
            "total_entities":      total,
            "total_matches":       matches,
            "total_mismatches":    mismatches,
            "total_ineligible":    total - matches - mismatches,
            "match_rate":          round(matches / total, 4) if total else 0.0,
            "review_required":     review_count,
            "fast_path_matches":   fast_path_count,
            "sections_processed":  len(report.section_results),
        }

    def _build_extractions(
        self,
        doc_a_entities: dict[str, FinalEntityValue],
        doc_b_entities: dict[str, FinalEntityValue],
    ) -> dict:
        """Build extraction results for both documents."""
        def _serialise(fev: FinalEntityValue) -> dict:
            d = {
                "entity_name":       fev.entity_name,
                "extracted_value":   fev.extracted_value,
                "extraction_status": fev.extraction_status.value,
                "entity_type":       fev.entity_type.value,
                "confidence":        fev.confidence,
                "source_page":       fev.source_page,
                "source_region":     fev.source_region,
                "review_required":   fev.review_required,
                "fallback_triggered": fev.fallback_triggered,
            }
            if fev.expression_audit:
                d["expression_audit"] = fev.expression_audit
            return d

        return {
            "doc_a": {
                name: _serialise(fev)
                for name, fev in doc_a_entities.items()
            },
            "doc_b": {
                name: _serialise(fev)
                for name, fev in doc_b_entities.items()
            },
        }

    def _build_validation_results(self, report: ValidationReport) -> list[dict]:
        """Build per-section, per-entity validation results."""
        sections_out = []
        for section in report.section_results:
            entities_out = []
            for result in section.entity_results:
                entities_out.append({
                    "entity_name":       result.entity_name,
                    "doc_a_value":       result.doc_a_value,
                    "doc_b_value":       result.doc_b_value,
                    "doc_a_normalized":  result.doc_a_normalized,
                    "doc_b_normalized":  result.doc_b_normalized,
                    "validation_status": result.validation_status.value,
                    "discrepancy_type":  result.discrepancy_type.value,
                    "reasoning":         result.reasoning,
                    "confidence":        result.confidence,
                    "review_required":   result.review_required,
                    "fast_path_match":   result.fast_path_match,
                })
            sections_out.append({
                "section_name":    section.section_name,
                "match_count":     section.match_count,
                "mismatch_count":  section.mismatch_count,
                "entities":        entities_out,
            })
        return sections_out
