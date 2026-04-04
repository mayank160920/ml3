import sys
import os
sys.path.insert(0, "/home/claude/src")
import pytest
from extraction.expression_evaluator import ExpressionEvaluator, EvalStatus


@pytest.fixture
def evaluator():
    return ExpressionEvaluator()


def make_extraction(value, confidence):
    return {"extracted_value": value, "confidence": confidence}


def test_basic_addition(evaluator):
    variables = {"tier1": {}, "tier2": {}}
    extractions = {
        "Total_Family_Deductible__VAR__tier1": make_extraction("$1,500", 0.99),
        "Total_Family_Deductible__VAR__tier2": make_extraction("$2,000", 0.99),
    }
    result = evaluator.evaluate("Total_Family_Deductible", "tier1 + tier2", variables, extractions)
    assert result.status == EvalStatus.SUCCESS
    assert result.evaluated_result == 3500.0


def test_expression_with_round(evaluator):
    variables = {"rate": {}}
    extractions = {
        "Coinsurance_Calc__VAR__rate": make_extraction("20%", 0.95),
    }
    result = evaluator.evaluate("Coinsurance_Calc", "round((1 - rate) * 100, 2)", variables, extractions)
    assert result.status == EvalStatus.SUCCESS
    assert result.evaluated_result == 80.0


def test_missing_variable_returns_missing_status(evaluator):
    variables = {"tier1": {}, "tier2": {}}
    extractions = {
        "Total_Family_Deductible__VAR__tier1": make_extraction("$1,500", 0.99),
    }
    result = evaluator.evaluate("Total_Family_Deductible", "tier1 + tier2", variables, extractions)
    assert result.status == EvalStatus.MISSING_VARIABLES
    assert result.evaluated_result is None


def test_invalid_expression_returns_eval_error(evaluator):
    variables = {"tier1": {}}
    extractions = {
        "Bad_Entity__VAR__tier1": make_extraction("$1,000", 0.99),
    }
    result = evaluator.evaluate("Bad_Entity", "tier1 + unknown_var", variables, extractions)
    assert result.status == EvalStatus.EVAL_ERROR
    assert result.evaluated_result is None


def test_parse_dollar_with_commas(evaluator):
    assert evaluator._parse_numeric("$1,500") == 1500.0


def test_parse_percentage(evaluator):
    assert evaluator._parse_numeric("20%") == 0.20


def test_parse_decimal(evaluator):
    assert evaluator._parse_numeric("0.20") == 0.20


def test_parse_float_with_commas(evaluator):
    assert evaluator._parse_numeric("1,500.00") == 1500.0


def test_parse_unparseable_returns_none(evaluator):
    assert evaluator._parse_numeric("N/A") is None


def test_confidence_average(evaluator):
    variables = {"tier1": {}, "tier2": {}}
    extractions = {
        "Test__VAR__tier1": make_extraction("$1,000", 0.90),
        "Test__VAR__tier2": make_extraction("$2,000", 0.80),
    }
    result = evaluator.evaluate("Test", "tier1 + tier2", variables, extractions)
    assert result.status == EvalStatus.SUCCESS
    assert result.confidence == round(0.85, 4)


def test_confidence_penalized_when_low(evaluator):
    variables = {"tier1": {}, "tier2": {}}
    extractions = {
        "Test__VAR__tier1": make_extraction("$1,000", 0.60),
        "Test__VAR__tier2": make_extraction("$2,000", 0.80),
    }
    result = evaluator.evaluate("Test", "tier1 + tier2", variables, extractions)
    assert result.status == EvalStatus.SUCCESS
    expected = round(((0.60 + 0.80) / 2) * 0.85, 4)
    assert result.confidence == expected
