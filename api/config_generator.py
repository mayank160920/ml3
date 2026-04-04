"""
config_generator.py
--------------------
Converts user-defined fields (from CSV or form input) into:
  1. A CMSVS YAML config file   → configs/<name>.yaml
  2. A Markdown instruction file → configs/markdowns/<name>.md

The Markdown file is a structured LLM extraction prompt that the CMSVS
pipeline can use for domain-agnostic entity extraction.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from api.models import ConfigCreateResponse, FieldDefinition

# ── Paths ─────────────────────────────────────────────────────────────────────

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
MARKDOWNS_DIR = CONFIGS_DIR / "markdowns"


def _slugify(name: str) -> str:
    """Convert a human name to a safe snake_case identifier."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return s.strip("_")


# ══════════════════════════════════════════════════════════════════════════════
# YAML generation
# ══════════════════════════════════════════════════════════════════════════════

def _build_yaml_dict(
    config_name: str,
    domain: str,
    fields: list[FieldDefinition],
) -> dict[str, Any]:
    """Build the YAML-serialisable dict matching CMSVSConfig schema."""

    # Group fields by section
    sections_map: dict[str, list[FieldDefinition]] = defaultdict(list)
    for f in fields:
        sections_map[f.section or "General"].append(f)

    sections = []
    for section_name, section_fields in sections_map.items():
        entities = []
        for f in section_fields:
            entity: dict[str, Any] = {
                "entity_name": _slugify(f.field_name),
                "entity_description": f.field_description or f.field_name,
                "entity_extraction_logic": f.extraction_logic.upper(),
                "entity_example_value": f.example_value or "",
                "data_type": f.data_type or "text",
            }
            if f.extraction_logic.upper() == "EXPRESSION" and f.expression_template:
                entity["expression_template"] = f.expression_template
            entities.append(entity)

        # Build section keywords from field names + section name
        keywords = [_slugify(w) for w in section_name.split()] + [
            _slugify(f.field_name) for f in section_fields
        ]

        sections.append({
            "section_name": section_name,
            "section_description": f"Fields related to {section_name}.",
            "section_keywords": list(dict.fromkeys(keywords)),  # deduplicate, keep order
            "entities": entities,
        })

    return {
        "config_name": _slugify(config_name),
        "version": "1.0",
        "domain": domain or "general",
        "sections": sections,
    }


def _write_yaml(config_name: str, yaml_dict: dict) -> Path:
    """Write YAML config to disk and return the path."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = CONFIGS_DIR / f"{_slugify(config_name)}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(yaml_dict, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return yaml_path


# ══════════════════════════════════════════════════════════════════════════════
# Markdown generation  —  structured LLM extraction instructions
# ══════════════════════════════════════════════════════════════════════════════

def _generate_markdown(
    config_name: str,
    domain: str,
    fields: list[FieldDefinition],
) -> str:
    """
    Generate a Markdown document that serves as a structured prompt
    for the LLM to extract the user-defined fields from any document.
    """
    slug = _slugify(config_name)

    lines: list[str] = []
    lines.append(f"# Extraction Configuration: {config_name}")
    lines.append(f"**Domain:** {domain}  ")
    lines.append(f"**Config ID:** `{slug}`  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Instructions for Entity Extraction")
    lines.append("")
    lines.append(
        "Extract **each** of the entities listed below from the provided document. "
        "For every entity return:")
    lines.append("")
    lines.append("| Key | Description |")
    lines.append("|-----|-------------|")
    lines.append("| `entity_name` | Exact identifier from this spec |")
    lines.append("| `value` | Extracted value as a string |")
    lines.append("| `confidence` | Float 0-1 indicating extraction certainty |")
    lines.append("| `source_region` | Approximate location in the document |")
    lines.append("")
    lines.append("If a value cannot be found, set `value` to `null` and `confidence` to `0.0`.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Group by section
    sections_map: dict[str, list[FieldDefinition]] = defaultdict(list)
    for f in fields:
        sections_map[f.section or "General"].append(f)

    for idx, (section_name, section_fields) in enumerate(sections_map.items(), 1):
        lines.append(f"## Section {idx}: {section_name}")
        lines.append("")

        for f in section_fields:
            entity_slug = _slugify(f.field_name)
            lines.append(f"### `{entity_slug}`")
            lines.append("")
            lines.append(f"- **Description:** {f.field_description or f.field_name}")
            lines.append(f"- **Data Type:** `{f.data_type or 'text'}`")
            lines.append(f"- **Extraction Logic:** `{f.extraction_logic.upper()}`")
            if f.example_value:
                lines.append(f"- **Example Value:** `{f.example_value}`")
            if f.extraction_logic.upper() == "EXPRESSION" and f.expression_template:
                lines.append(f"- **Expression:** `{f.expression_template}`")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Summary table
    lines.append("## Field Summary")
    lines.append("")
    lines.append("| # | Field Name | Section | Data Type | Logic |")
    lines.append("|---|-----------|---------|-----------|-------|")
    for i, f in enumerate(fields, 1):
        lines.append(
            f"| {i} | `{_slugify(f.field_name)}` | {f.section or 'General'} "
            f"| {f.data_type or 'text'} | {f.extraction_logic.upper()} |"
        )
    lines.append("")

    return "\n".join(lines)


def _write_markdown(config_name: str, content: str) -> Path:
    """Write markdown to disk and return the path."""
    MARKDOWNS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = MARKDOWNS_DIR / f"{_slugify(config_name)}.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return md_path


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def create_config_from_fields(
    config_name: str,
    domain: str,
    fields: list[FieldDefinition],
) -> ConfigCreateResponse:
    """
    End-to-end: take user field definitions → write YAML + Markdown → return response.
    """
    # 1. Build & write YAML
    yaml_dict = _build_yaml_dict(config_name, domain, fields)
    yaml_path = _write_yaml(config_name, yaml_dict)

    # 2. Build & write Markdown
    md_content = _generate_markdown(config_name, domain, fields)
    md_path = _write_markdown(config_name, md_content)

    # 3. Compute totals
    sections_map: dict[str, list] = defaultdict(list)
    for f in fields:
        sections_map[f.section or "General"].append(f)

    return ConfigCreateResponse(
        config_name=_slugify(config_name),
        yaml_path=str(yaml_path),
        markdown_path=str(md_path),
        total_sections=len(sections_map),
        total_fields=len(fields),
        markdown_preview=md_content,
    )


def list_markdowns() -> list[str]:
    """Return names of all saved markdown config files."""
    if not MARKDOWNS_DIR.exists():
        return []
    return sorted(p.stem for p in MARKDOWNS_DIR.glob("*.md"))


def get_markdown(config_name: str) -> tuple[str, str, bool]:
    """
    Return (config_name, markdown_content, yaml_exists) for a given config.
    Raises FileNotFoundError if the markdown does not exist.
    """
    slug = _slugify(config_name)
    md_path = MARKDOWNS_DIR / f"{slug}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown '{slug}.md' not found in {MARKDOWNS_DIR}")
    yaml_exists = (CONFIGS_DIR / f"{slug}.yaml").exists()
    return slug, md_path.read_text(encoding="utf-8"), yaml_exists


def delete_config(config_name: str) -> list[str]:
    """
    Delete YAML + Markdown files for a config. Returns list of deleted file paths.
    Raises FileNotFoundError if neither file exists.
    """
    slug = _slugify(config_name)
    yaml_path = CONFIGS_DIR / f"{slug}.yaml"
    md_path = MARKDOWNS_DIR / f"{slug}.md"

    deleted = []
    if yaml_path.exists():
        yaml_path.unlink()
        deleted.append(str(yaml_path))
    if md_path.exists():
        md_path.unlink()
        deleted.append(str(md_path))

    if not deleted:
        raise FileNotFoundError(
            f"No config files found for '{slug}' in {CONFIGS_DIR}"
        )
    return deleted
