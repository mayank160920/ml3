import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_types import (
    SectionConfig, FinalEntityValue, ExtractionResult,
    ExtractionMode, ExtractionStatus
)
from extraction.mllm_extractor import MLLMExtractor
from extraction.expression_evaluator import ExpressionEvaluator, EvalStatus
from typing import Dict, Optional


class ExpressionOrchestrator:

    def __init__(self, mllm_client):
        self.extractor = MLLMExtractor(mllm_client)
        self.evaluator = ExpressionEvaluator()

    def process_section(
        self,
        section_config: SectionConfig,
        page_images: list
    ) -> Dict[str, FinalEntityValue]:
        raw_extractions: Dict[str, ExtractionResult] = (
            self.extractor.extract_section_entities(section_config, page_images)
        )

        results: Dict[str, FinalEntityValue] = {}

        for entity in section_config.direct_entities:
            name = entity.entity_name
            extraction = raw_extractions.get(name)

            if extraction and extraction.status == ExtractionStatus.EXTRACTED:
                results[name] = FinalEntityValue(
                    entity_name=name,
                    final_value=extraction.extracted_value,
                    numeric_value=None,
                    extraction_mode=ExtractionMode.DIRECT,
                    status=ExtractionStatus.EXTRACTED,
                    confidence=extraction.confidence,
                    source_pages=[extraction.source_page] if extraction.source_page else [],
                    audit_trail={},
                    requires_human_review=extraction.requires_human_review,
                )
            else:
                results[name] = FinalEntityValue(
                    entity_name=name,
                    final_value=None,
                    numeric_value=None,
                    extraction_mode=ExtractionMode.DIRECT,
                    status=ExtractionStatus.INELIGIBLE,
                    confidence=0.0,
                    source_pages=[],
                    audit_trail={},
                    requires_human_review=False,
                )

        for entity in section_config.expression_entities:
            name = entity.entity_name
            variable_extractions = {}

            for var_name in entity.expression_variables:
                key = f"{name}__VAR__{var_name}"
                extraction = raw_extractions.get(key)
                if extraction:
                    variable_extractions[key] = {
                        "extracted_value": extraction.extracted_value,
                        "confidence": extraction.confidence,
                    }

            eval_result = self.evaluator.evaluate(
                entity_name=name,
                expression_template=entity.expression_template,
                expression_variables=entity.expression_variables,
                variable_extractions=variable_extractions,
            )

            if eval_result.status == EvalStatus.SUCCESS:
                numeric = eval_result.evaluated_result
                if entity.data_type == "percentage":
                    final_value = f"{numeric * 100:.1f}%"
                else:
                    final_value = f"${numeric:,.2f}"

                results[name] = FinalEntityValue(
                    entity_name=name,
                    final_value=final_value,
                    numeric_value=numeric,
                    extraction_mode=ExtractionMode.EXPRESSION,
                    status=ExtractionStatus.EXTRACTED,
                    confidence=eval_result.confidence,
                    source_pages=[],
                    audit_trail={
                        "expression_template": eval_result.expression_template,
                        "variable_values": eval_result.variable_values,
                        "evaluated_result": numeric,
                    },
                    requires_human_review=False,
                )
            else:
                results[name] = FinalEntityValue(
                    entity_name=name,
                    final_value=None,
                    numeric_value=None,
                    extraction_mode=ExtractionMode.EXPRESSION,
                    status=ExtractionStatus.ERROR,
                    confidence=0.0,
                    source_pages=[],
                    audit_trail={
                        "expression_template": entity.expression_template,
                        "error_message": eval_result.error_message,
                        "variable_values": eval_result.variable_values,
                    },
                    requires_human_review=True,
                )

        return results
