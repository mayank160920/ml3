from typing import List


SYSTEM_PROMPT = (
    "You are a precision document extraction agent. "
    "Extract only content that is explicitly visible in the provided document pages. "
    "Never infer, assume, or hallucinate values. "
    "Always cite the source region where the value was found. "
    "If a value is not present in the document, return null for extracted_value and set status to INELIGIBLE. "
    "Return only valid JSON. No markdown. No explanation outside the JSON."
)


class NERPromptBuilder:

    def build_section_prompt(self, section_name: str, targets: list, page_numbers: List[int]) -> str:
        page_list = ", ".join(str(p) for p in page_numbers)

        lines = []
        lines.append(f"Section: {section_name}")
        lines.append(f"Pages provided: {page_list}")
        lines.append("")
        lines.append("Extract the following entities from the document pages above.")
        lines.append("")

        for target in targets:
            entity_name = target.get("entity_name")
            extraction_logic = target.get("entity_extraction_logic", "DIRECT")
            description = target.get("entity_description", "")
            example = target.get("entity_example_value", "")
            parent = target.get("parent_entity", None)

            if extraction_logic == "DIRECT":
                lines.append(f"[DIRECT ENTITY]")
            else:
                lines.append(f"[EXPRESSION VARIABLE for '{parent}']")

            lines.append(f"  entity_name: {entity_name}")
            if description:
                lines.append(f"  description: {description}")
            if example:
                lines.append(f"  example_value: {example}")
            lines.append("")

        lines.append("Return your response as a single JSON object with this exact structure:")
        lines.append("{")
        lines.append(f'  "section": "{section_name}",')
        lines.append(f'  "source_pages": [{page_list}],')
        lines.append('  "extractions": [')
        lines.append('    {')
        lines.append('      "entity_name": "<name>",')
        lines.append('      "extracted_value": "<value or null>",')
        lines.append('      "status": "EXTRACTED or INELIGIBLE",')
        lines.append('      "source_page": <page number>,')
        lines.append('      "source_region": "<table name, row, or area>",')
        lines.append('      "confidence": <0.0 to 1.0>,')
        lines.append('      "raw_context": "<exact text snippet from document>"')
        lines.append('    }')
        lines.append('  ]')
        lines.append('}')

        return "\n".join(lines)

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT
