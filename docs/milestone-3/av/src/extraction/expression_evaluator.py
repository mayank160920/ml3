from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum
import math
from simpleeval import EvalWithCompoundTypes, NameNotDefined


class EvalStatus(str, Enum):
    SUCCESS = "SUCCESS"
    MISSING_VARIABLES = "MISSING_VARIABLES"
    EVAL_ERROR = "EVAL_ERROR"


@dataclass
class VariableValue:
    name: str
    raw_value: str
    parsed_value: Optional[float]
    confidence: float


@dataclass
class ExpressionEvaluationResult:
    entity_name: str
    expression_template: str
    variable_values: Dict[str, Optional[float]]
    evaluated_result: Optional[float]
    status: EvalStatus
    error_message: Optional[str]
    confidence: float


class ExpressionEvaluator:

    def __init__(self):
        self._evaluator = EvalWithCompoundTypes(
            functions={
                "round": round,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "sqrt": math.sqrt,
                "ceil": math.ceil,
                "floor": math.floor,
            }
        )

    def _parse_numeric(self, value_str: str) -> Optional[float]:
        if value_str is None:
            return None
        value_str = str(value_str).strip()
        if value_str.endswith("%"):
            try:
                return float(value_str[:-1].replace(",", "")) / 100
            except ValueError:
                return None
        cleaned = value_str.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    def evaluate(self, entity_name: str, expression_template: str, expression_variables: dict, variable_extractions: dict) -> ExpressionEvaluationResult:
        collected: Dict[str, VariableValue] = {}

        for var_name, var_config in expression_variables.items():
            key = f"{entity_name}__VAR__{var_name}"
            extraction = variable_extractions.get(key)
            raw_value = extraction.get("extracted_value") if extraction else None
            confidence = extraction.get("confidence", 0.0) if extraction else 0.0
            parsed = self._parse_numeric(raw_value) if raw_value is not None else None
            collected[var_name] = VariableValue(
                name=var_name,
                raw_value=raw_value,
                parsed_value=parsed,
                confidence=confidence,
            )

        missing = [v.name for v in collected.values() if v.parsed_value is None]
        if missing:
            return ExpressionEvaluationResult(
                entity_name=entity_name,
                expression_template=expression_template,
                variable_values={k: v.parsed_value for k, v in collected.items()},
                evaluated_result=None,
                status=EvalStatus.MISSING_VARIABLES,
                error_message=f"Missing or unparseable variables: {missing}",
                confidence=0.0,
            )

        self._evaluator.names = {k: v.parsed_value for k, v in collected.items()}

        try:
            result = self._evaluator.eval(expression_template)
        except NameNotDefined as e:
            return ExpressionEvaluationResult(
                entity_name=entity_name,
                expression_template=expression_template,
                variable_values={k: v.parsed_value for k, v in collected.items()},
                evaluated_result=None,
                status=EvalStatus.EVAL_ERROR,
                error_message=str(e),
                confidence=0.0,
            )
        except Exception as e:
            return ExpressionEvaluationResult(
                entity_name=entity_name,
                expression_template=expression_template,
                variable_values={k: v.parsed_value for k, v in collected.items()},
                evaluated_result=None,
                status=EvalStatus.EVAL_ERROR,
                error_message=str(e),
                confidence=0.0,
            )

        confidences = [v.confidence for v in collected.values()]
        avg_confidence = sum(confidences) / len(confidences)
        if any(c < 0.7 for c in confidences):
            avg_confidence *= 0.85

        return ExpressionEvaluationResult(
            entity_name=entity_name,
            expression_template=expression_template,
            variable_values={k: v.parsed_value for k, v in collected.items()},
            evaluated_result=float(result),
            status=EvalStatus.SUCCESS,
            error_message=None,
            confidence=round(avg_confidence, 4),
        )
