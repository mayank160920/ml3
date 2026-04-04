"""
-----------
Central type contracts for the CMSVS pipeline.
All dataclasses, enums, and type aliases used across layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from PIL import Image


# ── Input Layer ───────────────────────────────────────────────────────────────

class InputType(str, Enum):
    """Supported document input types."""
    PDF = "pdf"
    IMAGE = "image"


@dataclass(slots=True)
class PageImage:
    """A single rendered page image with metadata."""
    page_number: int
    image: Image.Image
    image_base64: str
    mime_type: str
    width: int
    height: int


@dataclass(slots=True)
class LoadedInput:
    """Result of loading a document (PDF or Image)."""
    input_type: InputType
    source_path: Path
    total_pages: int
    page_images: dict[int, PageImage]


# ── OCR Layer ─────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class StructuredPage:
    """
    OCR output for a single page, structured for embedding.

    index_text is the composite text fed into the NVIDIA NemoRetriever
    embedding model — formatted as:
        SECTIONS: <headers> | KEY_VALUES: <pairs> | RAW_TEXT:\\n<raw>
    """
    page_number: int
    raw_text: str
    section_headers: list[str]
    key_value_pairs: dict[str, str]
    index_text: str


# ── Extraction Layer ──────────────────────────────────────────────────────────

class ExtractionStatus(str, Enum):
    """Status of a single entity extraction."""
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    ERROR = "ERROR"


class EntityType(str, Enum):
    """How the entity value is derived."""
    DIRECT = "DIRECT"
    EXPRESSION = "EXPRESSION"


@dataclass
class RawExtraction:
    """
    Raw MLLM extraction result for one entity before confidence
    thresholding or fallback logic.
    """
    entity_name: str
    extracted_value: Optional[str]
    extraction_status: ExtractionStatus
    source_page: Optional[int]
    source_region: str
    confidence: float
    raw_context: str


@dataclass
class EntityResult:
    """
    Per-entity extraction result after fallback logic is applied.
    Used internally between Layer 5 and the final output builder.
    """
    entity_name: str
    extracted_value: Optional[str]
    extraction_status: ExtractionStatus
    source_page: Optional[int]
    confidence: float
    review_required: bool = False
    raw_context: str = ""
    fallback_triggered: bool = False


@dataclass
class FinalEntityValue:
    """
    Unified output structure for a single extracted entity.
    Canonical output of Layer 5 + Layer 6.
    """
    entity_name: str
    extracted_value: Optional[str]
    extraction_status: ExtractionStatus
    entity_type: EntityType
    confidence: float
    source_page: Optional[int]
    source_region: str
    raw_context: str
    review_required: bool
    fallback_triggered: bool
    expression_audit: Optional[dict]   # populated for EXPRESSION entities only


# ── Validation Layer ──────────────────────────────────────────────────────────

class ValidationStatus(str, Enum):
    """Per-entity validation verdict."""
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    INELIGIBLE = "INELIGIBLE"


class DiscrepancyType(str, Enum):
    """Classification of a mismatch discrepancy."""
    NUMERIC_DIFFERENCE = "NUMERIC_DIFFERENCE"
    TERMINOLOGY_VARIANT = "TERMINOLOGY_VARIANT"
    COVERAGE_RECLASSIFICATION = "COVERAGE_RECLASSIFICATION"
    FORMAT_DIFFERENCE = "FORMAT_DIFFERENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class EntityValidationResult:
    """Validation result for a single entity pair (Doc A vs Doc B)."""
    entity_name: str
    section_name: str
    doc_a_value: Optional[str]
    doc_b_value: Optional[str]
    doc_a_normalized: Optional[str]
    doc_b_normalized: Optional[str]
    validation_status: ValidationStatus
    discrepancy_type: DiscrepancyType
    reasoning: str
    confidence: float
    review_required: bool
    fast_path_match: bool   # True if resolved by rule-based pre-normalization


@dataclass
class SectionValidationResult:
    """All entity validation results for one configuration section."""
    section_name: str
    entity_results: list[EntityValidationResult]

    @property
    def match_count(self) -> int:
        """Number of MATCH verdicts in this section."""
        return sum(
            1 for r in self.entity_results
            if r.validation_status == ValidationStatus.MATCH
        )

    @property
    def mismatch_count(self) -> int:
        """Number of MISMATCH verdicts in this section."""
        return sum(
            1 for r in self.entity_results
            if r.validation_status == ValidationStatus.MISMATCH
        )


@dataclass
class ValidationReport:
    """
    Complete validation report for one document pair.
    Output of Layer 7 (Section-Wise Semantic Validation).
    """
    doc_a_path: str
    doc_b_path: str
    config_name: str
    section_results: list[SectionValidationResult] = field(default_factory=list)

    @property
    def total_entities(self) -> int:
        return sum(len(s.entity_results) for s in self.section_results)

    @property
    def total_matches(self) -> int:
        return sum(s.match_count for s in self.section_results)

    @property
    def total_mismatches(self) -> int:
        return sum(s.mismatch_count for s in self.section_results)