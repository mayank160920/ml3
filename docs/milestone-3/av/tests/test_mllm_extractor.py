import sys
import os
sys.path.insert(0, "/home/claude")
sys.path.insert(0, "/home/claude/src")

import pytest
import json
from shared_types import (
    SectionConfig, EntityConfig, ExpressionVariable,
    ExtractionStatus
)
from mock_mllm_client import MockMLLMClient
from extraction.mllm_extractor import MLLMExtractor
from prompts.ner_prompt_builder import NERPromptBuilder


def make_doctor_visits_section():
    return SectionConfig(
        section_name="Doctor Visits",
        section_keywords=["doctor", "primary care", "specialist"],
        entities=[
            EntityConfig(
                entity_name="Primary_Care_Visit",
                entity_description="Primary care visit copay in-network",
                entity_extraction_logic="DIRECT",
                entity_example_value="$20 copay/visit",
            ),
            EntityConfig(
                entity_name="Specialist_Visit",
                entity_description="Specialist visit copay in-network",
                entity_extraction_logic="DIRECT",
                entity_example_value="$20 copay/visit",
            ),
        ]
    )


def make_expression_section():
    return SectionConfig(
        section_name="Plan Overview",
        section_keywords=["deductible", "out-of-pocket"],
        entities=[
            EntityConfig(
                entity_name="Individual_Deductible_In_Network",
                entity_description="Individual deductible in-network",
                entity_extraction_logic="DIRECT",
                entity_example_value="$0",
            ),
            EntityConfig(
                entity_name="Total_Family_Deductible_Combined",
                entity_description="Combined family deductible all tiers",
                entity_extraction_logic="EXPRESSION",
                entity_example_value="$3,500",
                expression_template="tier1 + tier2",
                expression_variables={
                    "tier1": ExpressionVariable(name="tier1", description="Tier 1 deductible", example="1500"),
                    "tier2": ExpressionVariable(name="tier2", description="Tier 2 deductible", example="2000"),
                },
                data_type="monetary",
            ),
        ]
    )


def test_prompt_contains_direct_label():
    builder = NERPromptBuilder()
    section = make_doctor_visits_section()
    prompt = builder.build_section_prompt(
        section.section_name,
        section.all_extraction_targets,
        [1, 2]
    )
    assert "[DIRECT ENTITY]" in prompt


def test_prompt_contains_expression_variable_label():
    builder = NERPromptBuilder()
    section = make_expression_section()
    prompt = builder.build_section_prompt(
        section.section_name,
        section.all_extraction_targets,
        [1]
    )
    assert "[EXPRESSION VARIABLE for" in prompt


def test_prompt_contains_entity_names():
    builder = NERPromptBuilder()
    section = make_doctor_visits_section()
    prompt = builder.build_section_prompt(
        section.section_name,
        section.all_extraction_targets,
        [1]
    )
    assert "Primary_Care_Visit" in prompt
    assert "Specialist_Visit" in prompt


def test_extraction_returns_correct_values_sbc():
    client = MockMLLMClient(document_type="sbc")
    extractor = MLLMExtractor(client)
    section = make_doctor_visits_section()
    from shared_types import PageImage
    pages = [PageImage(page_number=2, base64_image="fake")]
    results = extractor.extract_section_entities(section, pages)
    assert "Primary_Care_Visit" in results
    assert results["Primary_Care_Visit"].extracted_value == "$20 copay/visit"
    assert results["Primary_Care_Visit"].status == ExtractionStatus.EXTRACTED


def test_extraction_returns_correct_values_bg():
    client = MockMLLMClient(document_type="bg")
    extractor = MLLMExtractor(client)
    section = make_doctor_visits_section()
    from shared_types import PageImage
    pages = [PageImage(page_number=1, base64_image="fake")]
    results = extractor.extract_section_entities(section, pages)
    assert "Primary_Care_Visit" in results
    assert results["Primary_Care_Visit"].status == ExtractionStatus.EXTRACTED


def test_malformed_json_returns_uncertain_not_crash():
    from mock_mllm_client import MockMLLMClient

    class BrokenClient:
        def generate(self, prompt, page_images):
            return "this is not json at all {{{"

    extractor = MLLMExtractor(BrokenClient())
    section = make_doctor_visits_section()
    from shared_types import PageImage
    pages = [PageImage(page_number=1, base64_image="fake")]
    results = extractor.extract_section_entities(section, pages)
    for name, result in results.items():
        assert result.status == ExtractionStatus.UNCERTAIN
        assert result.requires_human_review is True


def test_missing_entity_returns_ineligible():
    class PartialClient:
        def generate(self, prompt, page_images):
            return json.dumps({
                "section": "Doctor Visits",
                "source_pages": [1],
                "extractions": [
                    {
                        "entity_name": "Primary_Care_Visit",
                        "extracted_value": "$20 copay/visit",
                        "status": "EXTRACTED",
                        "source_page": 1,
                        "source_region": "table row",
                        "confidence": 0.99,
                        "raw_context": "$20 copay/visit"
                    }
                ]
            })

    extractor = MLLMExtractor(PartialClient())
    section = make_doctor_visits_section()
    from shared_types import PageImage
    pages = [PageImage(page_number=1, base64_image="fake")]
    results = extractor.extract_section_entities(section, pages)
    assert results["Specialist_Visit"].status == ExtractionStatus.INELIGIBLE


def test_var_key_format_for_expression_variables():
    section = make_expression_section()
    targets = section.all_extraction_targets
    var_keys = [t["entity_name"] for t in targets if "__VAR__" in t["entity_name"]]
    assert "Total_Family_Deductible_Combined__VAR__tier1" in var_keys
    assert "Total_Family_Deductible_Combined__VAR__tier2" in var_keys


def test_markdown_stripped_from_response():
    class MarkdownClient:
        def generate(self, prompt, page_images):
            return "```json\n" + json.dumps({
                "section": "Doctor Visits",
                "source_pages": [1],
                "extractions": [
                    {
                        "entity_name": "Primary_Care_Visit",
                        "extracted_value": "$20 copay/visit",
                        "status": "EXTRACTED",
                        "source_page": 1,
                        "source_region": "row",
                        "confidence": 0.98,
                        "raw_context": "$20 copay/visit"
                    }
                ]
            }) + "\n```"

    extractor = MLLMExtractor(MarkdownClient())
    section = make_doctor_visits_section()
    from shared_types import PageImage
    pages = [PageImage(page_number=1, base64_image="fake")]
    results = extractor.extract_section_entities(section, pages)
    assert results["Primary_Care_Visit"].status == ExtractionStatus.EXTRACTED
