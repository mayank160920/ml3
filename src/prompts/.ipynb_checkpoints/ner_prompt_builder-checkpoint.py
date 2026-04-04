"""
----------------------------------
Builds MLLM extraction prompts for the NER (Named Entity Recognition) layer.

Design principles
-----------------
1. Section-batched  — all entities in a section are extracted in one call,
   reducing API calls from N_entities to N_sections (~72% reduction).
2. Type-aware       — DIRECT and EXPRESSION variable entities get different
   instructions so the MLLM knows what role each field plays.
3. Schema-enforced  — strict JSON schema is embedded in the prompt so the
   MLLM returns a parseable structure on the first attempt.
4. Evidence-grounded— the MLLM is instructed to cite source region and
   surrounding context for every extraction (auditability).
"""
from __future__ import annotations

from config.config_parser import EntityConfig, SectionConfig


class NERPromptBuilder:
    """
    Constructs extraction prompts for the MLLM visual extraction layer.

    Usage
    -----
    >>> builder = NERPromptBuilder()
    >>> prompt = builder.build_section_prompt(section_config)
    """

    def build_section_prompt(self, section: SectionConfig) -> str:
        """
        Build a complete extraction prompt for all entities in a section.

        The prompt instructs the MLLM to:
          - Locate each entity in the provided document page images
          - Return the value exactly as it appears (no paraphrasing)
          - Distinguish DIRECT entities from EXPRESSION variable entities
          - Fill the strict JSON schema for every entity

        Parameters
        ----------
        section : SectionConfig with all entity definitions

        Returns
        -------
        str : complete prompt ready for the MLLM
        """
        entity_block = self._build_entity_block(section.entities)
        schema_block = self._build_schema_block(section)

        return f"""You are a document intelligence assistant extracting structured data from document page images.

SECTION: {section.section_name}
SECTION DESCRIPTION: {section.section_description}
SECTION KEYWORDS: {", ".join(section.section_keywords)}

ENTITIES TO EXTRACT:
{entity_block}

INSTRUCTIONS:
1. Examine the provided document page image(s) carefully.
2. For each entity, locate its value in the document using visual context (tables, labels, spatial layout).
3. Extract the value EXACTLY as it appears — do NOT paraphrase, normalise, or reformat monetary amounts, percentages, or dates.
4. For EXPRESSION_VARIABLE entities, extract the raw component value only — do NOT compute the expression.
5. If an entity is not visible in the pages, set extracted_value to null and extraction_status to NOT_FOUND.
6. Set source_page to the 0-based page index where the value was found.
7. Describe the source_region (e.g. "Row 3 of In-Network Deductibles table").
8. Set confidence between 0.0 (uncertain) and 1.0 (certain).
9. Include up to 100 characters of surrounding raw_context for audit.

RETURN ONLY the following JSON object — no markdown fences, no explanation:
{schema_block}"""

    def build_fallback_prompt(self, section: SectionConfig) -> str:
        """
        Build a fallback extraction prompt with expanded search instructions.

        Used when initial extraction has low-confidence entities. The prompt
        instructs the MLLM to search more broadly across all provided pages.

        Parameters
        ----------
        section : SectionConfig

        Returns
        -------
        str : fallback prompt
        """
        base = self.build_section_prompt(section)
        fallback_note = (
            "\n\nNOTE: This is a FALLBACK extraction pass. "
            "Some entities had low confidence on the first attempt. "
            "Search ALL provided pages more carefully. "
            "Check tables, footnotes, headers, and secondary columns. "
            "Pay special attention to in-network vs out-of-network distinctions."
        )
        return base + fallback_note

    # ── private ───────────────────────────────────────────────────────────────

    def _build_entity_block(self, entities: list[EntityConfig]) -> str:
        """Format the list of entities into a numbered block for the prompt."""
        lines: list[str] = []
        for i, entity in enumerate(entities, start=1):
            entity_type = (
                "EXPRESSION_VARIABLE"
                if entity.entity_extraction_logic == "EXPRESSION"
                else "DIRECT"
            )
            lines.append(
                f"{i}. entity_name: \"{entity.entity_name}\"\n"
                f"   type: {entity_type}\n"
                f"   description: \"{entity.entity_description}\"\n"
                f"   example_value: \"{entity.entity_example_value}\"\n"
                f"   data_type: {entity.data_type}"
            )
            if (entity.entity_extraction_logic == "EXPRESSION"
                    and entity.expression_variables):
                var_names = [v.name for v in entity.expression_variables]
                lines.append(
                    f"   note: Extract this as a raw variable value. "
                    f"   It will be used in: {entity.expression_template}\n"
                    f"   Related variables: {', '.join(var_names)}"
                )
        return "\n\n".join(lines)

    def _build_schema_block(self, section: SectionConfig) -> str:
        """Build the expected JSON output schema for the prompt."""
        entity_schemas = []
        for entity in section.entities:
            entity_schemas.append(
                "    {\n"
                f'      "entity_name": "{entity.entity_name}",\n'
                '      "extracted_value": "<value or null>",\n'
                '      "extraction_status": "<FOUND|NOT_FOUND|AMBIGUOUS>",\n'
                '      "source_page": <0-based int or null>,\n'
                '      "source_region": "<description of location>",\n'
                '      "confidence": <0.0 to 1.0>,\n'
                '      "raw_context": "<surrounding text snippet>"\n'
                "    }"
            )
        entities_json = ",\n".join(entity_schemas)
        return (
            "{\n"
            f'  "section_name": "{section.section_name}",\n'
            '  "extractions": [\n'
            f"{entities_json}\n"
            "  ]\n"
            "}"
        )
