"""
Pydantic request / response models for all API endpoints.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    nvidia_key_set: bool = False
    configs_available: list[str] = []


# ── Config ─────────────────────────────────────────────────────────────────────

class EntityInfo(BaseModel):
    entity_name: str
    entity_description: str
    entity_extraction_logic: str   # DIRECT | EXPRESSION
    entity_example_value: str
    data_type: str


class SectionInfo(BaseModel):
    section_name: str
    section_description: str
    section_keywords: list[str]
    entities: list[EntityInfo]


class ConfigDetailResponse(BaseModel):
    config_name: str
    version: str
    domain: str
    total_sections: int
    total_entities: int
    sections: list[SectionInfo]


class ConfigListResponse(BaseModel):
    configs: list[str]


# ── Extraction ─────────────────────────────────────────────────────────────────

class ExtractionRequest(BaseModel):
    config_name: str = Field(
        ...,
        description="Config name without .yaml (e.g. 'funsd_ner_config')",
        example="funsd_ner_config",
    )
    confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum confidence before flagging for review",
    )


class ExtractedEntity(BaseModel):
    entity_name: str
    extracted_value: Optional[str] = None
    extraction_status: Optional[str] = None
    entity_type: Optional[str] = None
    confidence: float = 0.0
    source_page: Optional[int] = None
    source_region: Optional[str] = None          # ← was str, now Optional[str]
    review_required: bool = False
    fallback_triggered: bool = False
    expression_audit: Optional[dict] = None


class ExtractionResponse(BaseModel):
    job_id: str
    document_path: str
    config_name: str
    total_entities: int
    found_count: int
    review_count: int
    processing_time_s: float
    entities: dict[str, ExtractedEntity]


# ── Validation ──────────────────────────────────────────────────────────────────

class ValidationEntityResult(BaseModel):
    entity_name: str
    section_name: Optional[str] = None
    doc_a_value: Optional[str] = None
    doc_b_value: Optional[str] = None
    doc_a_normalized: Optional[str] = None
    doc_b_normalized: Optional[str] = None
    validation_status: Optional[str] = None
    discrepancy_type: Optional[str] = None
    reasoning: Optional[str] = None              # ← was str, now Optional[str]
    confidence: float = 0.0
    review_required: bool = False
    fast_path_match: bool = False


class ValidationSectionResult(BaseModel):
    section_name: str
    match_count: int
    mismatch_count: int
    entities: list[ValidationEntityResult]


class ValidationSummary(BaseModel):
    total_entities: int
    total_matches: int
    total_mismatches: int
    total_ineligible: int
    match_rate: float
    review_required: int
    fast_path_matches: int
    sections_processed: int


class ValidationResponse(BaseModel):
    job_id: str
    doc_a_name: str
    doc_b_name: str
    config_name: str
    processing_time_s: float
    summary: ValidationSummary
    sections: list[ValidationSectionResult]
    doc_a_entities: dict[str, ExtractedEntity]
    doc_b_entities: dict[str, ExtractedEntity]


class GroundTruthEntity(BaseModel):
    entity_name: str
    doc_a_value: Optional[str] = None
    doc_b_value: Optional[str] = None
    normalized_value: Optional[str] = None
    validation_type: Optional[str] = None
    validation_result: Optional[str] = None


class GroundTruthResponse(BaseModel):
    job_id: str
    doc_a_name: str
    doc_b_name: str
    config_name: str
    processing_time_s: float
    entities: list[GroundTruthEntity]


# ── Config Builder (CSV → YAML + Markdown) ──────────────────────────────────────

class FieldDefinition(BaseModel):
    """A single field/entity the user wants to extract."""
    field_name: str = Field(..., description="Unique field identifier (snake_case)")
    field_description: str = Field(default="", description="What this field represents")
    section: str = Field(default="General", description="Logical grouping / section name")
    data_type: str = Field(default="text", description="text | monetary | percentage | date | number")
    example_value: str = Field(default="", description="Example of expected extracted value")
    extraction_logic: str = Field(default="DIRECT", description="DIRECT | EXPRESSION")
    expression_template: Optional[str] = Field(default=None, description="Math expression for EXPRESSION fields")


class ConfigCreateRequest(BaseModel):
    """Request body for creating a new config from user-defined fields."""
    config_name: str = Field(..., description="Unique name for this configuration (no spaces)")
    domain: str = Field(default="general", description="Domain label (e.g. healthcare, finance)")
    fields: list[FieldDefinition] = Field(..., min_length=1, description="List of fields to extract")


class ConfigCreateResponse(BaseModel):
    config_name: str
    yaml_path: str
    markdown_path: str
    total_sections: int
    total_fields: int
    markdown_preview: str


class MarkdownListResponse(BaseModel):
    """List of saved markdown config files."""
    markdowns: list[str]


class MarkdownDetailResponse(BaseModel):
    config_name: str
    markdown_content: str
    yaml_exists: bool


class DeleteConfigResponse(BaseModel):
    config_name: str
    deleted_files: list[str]
    message: str


# ── Error ───────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: str
    job_id: Optional[str] = None