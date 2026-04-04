from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ExtractionStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    INELIGIBLE = "INELIGIBLE"
    UNCERTAIN = "UNCERTAIN"
    ERROR = "ERROR"


class ExtractionMode(str, Enum):
    DIRECT = "DIRECT"
    EXPRESSION = "EXPRESSION"


class ValidationStatus(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    INELIGIBLE = "INELIGIBLE"


class InputType(str, Enum):
    PDF = "PDF"
    IMAGE = "IMAGE"


@dataclass
class PageImage:
    page_number: int
    base64_image: str
    width: int = 0
    height: int = 0


@dataclass
class LoadedInput:
    input_type: InputType
    source_path: str
    pages: List[PageImage] = field(default_factory=list)


@dataclass
class ExpressionVariable:
    name: str
    description: str
    example: str


@dataclass
class EntityConfig:
    entity_name: str
    entity_description: str
    entity_extraction_logic: str
    entity_example_value: str
    expression_variables: Dict[str, ExpressionVariable] = field(default_factory=dict)
    expression_template: str = ""
    data_type: str = "text"


@dataclass
class SectionConfig:
    section_name: str
    section_keywords: List[str]
    entities: List[EntityConfig] = field(default_factory=list)

    @property
    def direct_entities(self) -> List[EntityConfig]:
        return [e for e in self.entities if e.entity_extraction_logic == "DIRECT"]

    @property
    def expression_entities(self) -> List[EntityConfig]:
        return [e for e in self.entities if e.entity_extraction_logic == "EXPRESSION"]

    @property
    def all_extraction_targets(self) -> List[Dict]:
        targets = []
        for entity in self.direct_entities:
            targets.append({
                "entity_name": entity.entity_name,
                "entity_description": entity.entity_description,
                "entity_extraction_logic": "DIRECT",
                "entity_example_value": entity.entity_example_value,
                "parent_entity": None,
            })
        for entity in self.expression_entities:
            for var_name, var_config in entity.expression_variables.items():
                targets.append({
                    "entity_name": f"{entity.entity_name}__VAR__{var_name}",
                    "entity_description": var_config.description,
                    "entity_extraction_logic": "EXPRESSION_VARIABLE",
                    "entity_example_value": var_config.example,
                    "parent_entity": entity.entity_name,
                })
        return targets


@dataclass
class ExtractionResult:
    entity_name: str
    extracted_value: Optional[str]
    status: ExtractionStatus
    source_page: Optional[int]
    source_region: Optional[str]
    confidence: float
    raw_context: Optional[str]
    requires_human_review: bool = False

    def __repr__(self):
        return (
            f"ExtractionResult(entity={self.entity_name}, "
            f"value={self.extracted_value}, status={self.status}, "
            f"confidence={self.confidence})"
        )


@dataclass
class FinalEntityValue:
    entity_name: str
    final_value: Optional[str]
    numeric_value: Optional[float]
    extraction_mode: ExtractionMode
    status: ExtractionStatus
    confidence: float
    source_pages: List[int]
    audit_trail: Dict[str, Any]
    requires_human_review: bool = False

    def __repr__(self):
        return (
            f"FinalEntityValue(entity={self.entity_name}, "
            f"value={self.final_value}, mode={self.extraction_mode}, "
            f"status={self.status}, confidence={self.confidence})"
        )


@dataclass
class ValidationResult:
    entity_name: str
    value_doc_a: Optional[str]
    value_doc_b: Optional[str]
    extraction_mode_a: ExtractionMode
    extraction_mode_b: ExtractionMode
    validation_status: ValidationStatus
    discrepancy_type: Optional[str]
    reasoning: Optional[str]
    confidence: float
    audit_trail: Dict[str, Any]
