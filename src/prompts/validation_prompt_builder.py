"""
-----------------------------------------
Builds Chain-of-Thought (CoT) semantic validation prompts for Layer 7.

Design principles
-----------------
1. Section-batched   — all non-exact entity pairs in a section are validated
   in one MLLM call (72% API call reduction vs entity-by-entity).
2. Five-step CoT     — structured reasoning guides the model through
   normalisation → alignment → discrepancy → status → confidence.
3. Context-rich      — section description and entity metadata are included
   so the MLLM has relational context for accurate judgements.
4. Schema-enforced   — strict JSON schema ensures parseable output.
"""
from __future__ import annotations

from config.config_parser import SectionConfig
from shared_types import EntityValidationResult, ValidationStatus


class ValidationPromptBuilder:
    """
    Builds Chain-of-Thought validation prompts for section-level entity
    comparison between Document A and Document B.

    Usage
    -----
    >>> builder = ValidationPromptBuilder()
    >>> prompt = builder.build_section_validation_prompt(
    ...     section, entity_pairs, doc_a_name, doc_b_name
    ... )
    """

    def build_section_validation_prompt(
        self,
        section: SectionConfig,
        entity_pairs: list[dict],
        doc_a_name: str = "Document A",
        doc_b_name: str = "Document B",
    ) -> str:
        """
        Build a CoT validation prompt for all entity pairs in a section.

        Parameters
        ----------
        section       : SectionConfig providing context and entity descriptions
        entity_pairs  : list of dicts with keys:
                          entity_name, description, data_type,
                          doc_a_value, doc_b_value,
                          doc_a_normalized, doc_b_normalized
        doc_a_name    : human-readable label for Document A
        doc_b_name    : human-readable label for Document B

        Returns
        -------
        str : complete CoT validation prompt
        """
        pairs_block = self._build_pairs_block(entity_pairs)
        schema_block = self._build_validation_schema(entity_pairs)

        return f"""You are a document validation expert comparing entity values between two documents.

SECTION: {section.section_name}
SECTION DESCRIPTION: {section.section_description}
DOCUMENT A: {doc_a_name}
DOCUMENT B: {doc_b_name}

ENTITY PAIRS TO VALIDATE:
{pairs_block}

VALIDATION INSTRUCTIONS — follow these five steps for EACH entity pair:

STEP 1 — NORMALISATION REVIEW:
  Verify the pre-normalised values. Apply any additional normalisation:
  - Monetary: "$1,500" / "1500" / "$1,500.00" → "1500.00 USD"
  - Percentage: "20%" / "0.20" / "20 percent" → "20.0%"
  - Coverage equivalents: "No charge" / "Covered in full" → "0.00 USD"
  - Coverage equivalents: "Not covered" / "Member pays 100%" → "MEMBER_PAYS_100_PERCENT"

STEP 2 — SEMANTIC ALIGNMENT CHECK:
  Determine whether the normalised values express the same underlying fact.
  Account for: paraphrase equivalence, unit equivalence, abbreviation expansion,
  and partial name matches (e.g., "Kapuler" matches "Kapuler Marketing Research").

STEP 3 — DISCREPANCY ANALYSIS:
  If values differ, classify the discrepancy:
  - NUMERIC_DIFFERENCE       : different numbers (e.g., 400 vs 500)
  - TERMINOLOGY_VARIANT      : same meaning, different words
  - COVERAGE_RECLASSIFICATION: covered vs not covered / tier change
  - FORMAT_DIFFERENCE        : same value, different format (resolve as MATCH)

STEP 4 — STATUS ASSIGNMENT:
  - MATCH         : values are semantically equivalent after normalisation
  - MISMATCH      : values express genuinely different facts
  - PARTIAL_MATCH : values overlap but are not fully equivalent
  - INELIGIBLE    : one or both values are null/missing

STEP 5 — CONFIDENCE CALIBRATION:
  Assign a confidence score 0.0–1.0 reflecting your certainty in the verdict.

RETURN ONLY the following JSON object — no markdown fences, no explanation:
{schema_block}"""

    def build_single_entity_prompt(
        self,
        entity_name: str,
        entity_description: str,
        data_type: str,
        doc_a_value: str | None,
        doc_b_value: str | None,
        doc_a_name: str = "Document A",
        doc_b_name: str = "Document B",
    ) -> str:
        """
        Build a validation prompt for a single entity pair.

        Used as a fallback when section-level validation fails to parse.

        Parameters
        ----------
        entity_name        : canonical entity name
        entity_description : natural language description
        data_type          : monetary | percentage | coverage_classification | text
        doc_a_value        : value extracted from Document A
        doc_b_value        : value extracted from Document B
        doc_a_name         : label for Document A
        doc_b_name         : label for Document B

        Returns
        -------
        str : single-entity validation prompt
        """
        return f"""You are a document validation expert.

Compare these two values for entity "{entity_name}":
  Description : {entity_description}
  Data type   : {data_type}
  {doc_a_name} value : {doc_a_value or "null"}
  {doc_b_name} value : {doc_b_value or "null"}

Determine whether these values are semantically equivalent.
Apply normalisation (monetary, percentage, coverage equivalents) before comparing.

Return ONLY this JSON (no markdown, no explanation):
{{
  "entity_name": "{entity_name}",
  "doc_a_normalized": "<normalised A value>",
  "doc_b_normalized": "<normalised B value>",
  "validation_status": "<MATCH|MISMATCH|PARTIAL_MATCH|INELIGIBLE>",
  "discrepancy_type": "<NUMERIC_DIFFERENCE|TERMINOLOGY_VARIANT|COVERAGE_RECLASSIFICATION|FORMAT_DIFFERENCE|NOT_APPLICABLE>",
  "reasoning": "<one sentence explanation>",
  "confidence": <0.0 to 1.0>
}}"""

    # ── private ───────────────────────────────────────────────────────────────

    def _build_pairs_block(self, entity_pairs: list[dict]) -> str:
        """Format entity pairs into a numbered block for the prompt."""
        lines: list[str] = []
        for i, pair in enumerate(entity_pairs, start=1):
            lines.append(
                f"{i}. ENTITY: {pair['entity_name']}\n"
                f"   Description   : {pair.get('description', '')}\n"
                f"   Data type     : {pair.get('data_type', 'text')}\n"
                f"   Doc A raw     : {pair.get('doc_a_value') or 'null'}\n"
                f"   Doc B raw     : {pair.get('doc_b_value') or 'null'}\n"
                f"   Doc A normalized: {pair.get('doc_a_normalized') or 'null'}\n"
                f"   Doc B normalized: {pair.get('doc_b_normalized') or 'null'}"
            )
        return "\n\n".join(lines)

    def _build_validation_schema(self, entity_pairs: list[dict]) -> str:
        """Build the expected JSON validation output schema."""
        entity_schemas = []
        for pair in entity_pairs:
            entity_schemas.append(
                "    {\n"
                f'      "entity_name": "{pair["entity_name"]}",\n'
                '      "doc_a_normalized": "<normalised value or null>",\n'
                '      "doc_b_normalized": "<normalised value or null>",\n'
                '      "validation_status": "<MATCH|MISMATCH|PARTIAL_MATCH|INELIGIBLE>",\n'
                '      "discrepancy_type": "<type or NOT_APPLICABLE>",\n'
                '      "reasoning": "<chain-of-thought explanation>",\n'
                '      "confidence": <0.0 to 1.0>\n'
                "    }"
            )
        entities_json = ",\n".join(entity_schemas)
        return (
            "{\n"
            '  "section_name": "<section name>",\n'
            '  "validations": [\n'
            f"{entities_json}\n"
            "  ]\n"
            "}"
        )
