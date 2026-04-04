"""
----------------------------
YAML configuration parser for the CMSVS pipeline.

Reads healthcare_sbc_config.yaml or funsd_ner_config.yaml and returns
structured EntityConfig / SectionConfig dataclasses consumed by every
downstream layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclass
class ExpressionVariable:
    """A single variable used in an EXPRESSION entity template."""
    name: str
    description: str
    example_value: str


@dataclass
class EntityConfig:
    """
    Configuration for a single entity to be extracted.

    entity_extraction_logic is either 'DIRECT' or 'EXPRESSION'.
    For EXPRESSION entities, expression_template and expression_variables
    are populated; for DIRECT entities they are empty / None.
    """
    entity_name: str
    entity_description: str
    entity_extraction_logic: str          # "DIRECT" | "EXPRESSION"
    entity_example_value: str
    data_type: str                        # monetary | percentage | coverage_classification | text
    expression_template: Optional[str] = None
    expression_variables: list[ExpressionVariable] = field(default_factory=list)


@dataclass
class SectionConfig:
    """
    Configuration for one logical document section.

    section_keywords feed directly into the retrieval query construction
    in Layer 4 (DenseRetriever).
    """
    section_name: str
    section_description: str
    section_keywords: list[str]
    entities: list[EntityConfig]


@dataclass
class ValidationSettings:
    """Global validation parameters from the config file."""
    confidence_threshold: float = 0.75
    high_stakes_entities: list[str] = field(default_factory=list)
    human_review_escalation: bool = True


@dataclass
class CMSVSConfig:
    """
    Top-level configuration object parsed from a YAML config file.
    Consumed by the pipeline orchestrator and all child layers.
    """
    config_name: str
    version: str
    domain: str
    sections: list[SectionConfig]
    validation_settings: ValidationSettings

    def get_section(self, section_name: str) -> Optional[SectionConfig]:
        """Retrieve a section by name; returns None if not found."""
        for s in self.sections:
            if s.section_name == section_name:
                return s
        return None

    def all_entity_names(self) -> list[str]:
        """Flat list of every entity_name across all sections."""
        return [
            e.entity_name
            for s in self.sections
            for e in s.entities
        ]


# ── Parser ────────────────────────────────────────────────────────────────────

class CMSVSConfigParser:
    """
    Loads and validates a CMSVS YAML configuration file.

    Usage
    -----
    >>> parser = CMSVSConfigParser()
    >>> config = parser.load("configs/healthcare_sbc_config.yaml")
    """

    def load(self, config_path: str | Path) -> CMSVSConfig:
        """
        Parse a YAML config file and return a CMSVSConfig object.

        Parameters
        ----------
        config_path : path to the YAML configuration file

        Returns
        -------
        CMSVSConfig

        Raises
        ------
        FileNotFoundError  : if config_path does not exist
        ValueError         : if required top-level keys are missing
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        self._validate_top_level(raw)
        

        sections = [
            self._parse_section(s)
            for s in raw.get("sections", [])
        ]

        vs_raw = raw.get("validation_settings", {})
        validation_settings = ValidationSettings(
            confidence_threshold=vs_raw.get("confidence_threshold", 0.75),
            high_stakes_entities=vs_raw.get("high_stakes_entities", []),
            human_review_escalation=vs_raw.get("human_review_escalation", True),
        )

        return CMSVSConfig(
            config_name=raw["config_name"],
            version=str(raw.get("version", "1.0")),
            domain=raw.get("domain", "general"),
            sections=sections,
            validation_settings=validation_settings,
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _validate_top_level(self, raw: dict) -> None:
        required = {"config_name", "sections"}
        missing = required - set(raw.keys())
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    def _parse_section(self, raw: dict) -> SectionConfig:
        entities = [self._parse_entity(e) for e in raw.get("entities", [])]
        return SectionConfig(
            section_name=raw["section_name"],
            section_description=raw.get("section_description", ""),
            section_keywords=raw.get("section_keywords", []),
            entities=entities,
        )

    def _parse_entity(self, raw: dict) -> EntityConfig:
        logic = raw.get("entity_extraction_logic", "DIRECT").upper()

        # Parse expression variables if this is an EXPRESSION entity
        expr_vars: list[ExpressionVariable] = []
        if logic == "EXPRESSION":
            for var_name, var_data in raw.get("expression_variables", {}).items():
                if isinstance(var_data, dict):
                    expr_vars.append(ExpressionVariable(
                        name=var_name,
                        description=var_data.get("description", ""),
                        example_value=str(var_data.get("example_value", "")),
                    ))
                else:
                    # Plain string description
                    expr_vars.append(ExpressionVariable(
                        name=var_name,
                        description=str(var_data),
                        example_value="",
                    ))

        return EntityConfig(
            entity_name=raw["entity_name"],
            entity_description=raw.get("entity_description", ""),
            entity_extraction_logic=logic,
            entity_example_value=str(raw.get("entity_example_value", "")),
            data_type=raw.get("data_type", "text"),
            expression_template=raw.get("expression_template"),
            expression_variables=expr_vars,
        )
