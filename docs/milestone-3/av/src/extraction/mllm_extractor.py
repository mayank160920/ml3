import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_types import (
    ExtractionResult, ExtractionStatus, SectionConfig
)
from prompts.ner_prompt_builder import NERPromptBuilder
from typing import Dict, List


class MLLMExtractor:

    def __init__(self, mllm_client):
        self.client = mllm_client
        self.prompt_builder = NERPromptBuilder()

    def extract_section_entities(
        self,
        section: SectionConfig,
        page_images: list
    ) -> Dict[str, ExtractionResult]:
        targets = section.all_extraction_targets
        page_numbers = [p.page_number for p in page_images]
        prompt = self.prompt_builder.build_section_prompt(
            section.section_name, targets, page_numbers
        )
        raw_response = self.client.generate(prompt, page_images)
        return self._parse_section_response(raw_response, targets, page_numbers)

    def _parse_section_response(
        self,
        raw_response: str,
        targets: list,
        pages: List[int]
    ) -> Dict[str, ExtractionResult]:
        target_names = {t["entity_name"] for t in targets}
        results: Dict[str, ExtractionResult] = {}

        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            for name in target_names:
                results[name] = ExtractionResult(
                    entity_name=name,
                    extracted_value=None,
                    status=ExtractionStatus.UNCERTAIN,
                    source_page=None,
                    source_region=None,
                    confidence=0.0,
                    raw_context=None,
                    requires_human_review=True,
                )
            return results

        extractions = parsed.get("extractions", [])
        found_names = set()

        for item in extractions:
            entity_name = item.get("entity_name")
            if not entity_name:
                continue

            status_str = item.get("status", "INELIGIBLE")
            if status_str == "EXTRACTED":
                status = ExtractionStatus.EXTRACTED
            else:
                status = ExtractionStatus.INELIGIBLE

            results[entity_name] = ExtractionResult(
                entity_name=entity_name,
                extracted_value=item.get("extracted_value"),
                status=status,
                source_page=item.get("source_page"),
                source_region=item.get("source_region"),
                confidence=float(item.get("confidence", 0.0)),
                raw_context=item.get("raw_context"),
                requires_human_review=False,
            )
            found_names.add(entity_name)

        for name in target_names:
            if name not in found_names:
                results[name] = ExtractionResult(
                    entity_name=name,
                    extracted_value=None,
                    status=ExtractionStatus.INELIGIBLE,
                    source_page=None,
                    source_region=None,
                    confidence=0.0,
                    raw_context=None,
                    requires_human_review=False,
                )

        return results
