"""
----------------------------------------
SimpleEval-based sandboxed expression evaluator for EXPRESSION entities.

Security rationale
------------------
Python's built-in eval() can execute arbitrary code — unacceptable when
configuration files are authored by domain experts without security review.
SimpleEval enforces a strict whitelist of permitted operators and functions,
preventing all code execution outside safe arithmetic and mathematical
operations.

Supported operations
---------------------
- Arithmetic      : +  -  *  /  **  %
- Comparison      : >  <  >=  <=  ==  !=
- Safe functions  : round, abs, min, max, sum, sqrt, ceil, floor
- Conditionals    : value_if_true if condition else value_if_false

Numeric parsing
---------------
Handles currency symbols, commas, and percentage notation:
  "$1,500"   → 1500.0
  "20%"      → 20.0
  "1,500.00" → 1500.0
"""
from __future__ import annotations

import math
from typing import Any

from simpleeval import EvalWithCompoundTypes, FeatureNotAvailable


# ── Permitted functions ────────────────────────────────────────────────────────

_SAFE_FUNCTIONS: dict[str, Any] = {
    "round": round,
    "abs":   abs,
    "min":   min,
    "max":   max,
    "sum":   sum,
    "sqrt":  math.sqrt,
    "ceil":  math.ceil,
    "floor": math.floor,
}


class ExpressionEvaluator:
    """
    Evaluates arithmetic expression templates using extracted variable values.

    Usage
    -----
    >>> evaluator = ExpressionEvaluator()
    >>> result = evaluator.evaluate(
    ...     template="var_a + var_b",
    ...     variable_values={"var_a": "$1,500", "var_b": "$3,000"},
    ...     data_type="monetary",
    ... )
    """

    def evaluate(
        self,
        template: str,
        variable_values: dict[str, str | None],
        data_type: str = "monetary",
    ) -> dict:
        """
        Parse variable values and evaluate the expression template.

        Parameters
        ----------
        template         : arithmetic expression string using variable names
        variable_values  : mapping of variable_name → raw extracted string
        data_type        : output formatting type (monetary | percentage | text)

        Returns
        -------
        dict with keys:
          status          : "SUCCESS" | "ERROR"
          computed_value  : formatted string result (e.g., "$1,500.00")
          numeric_result  : float result value
          template_used   : original template string
          variable_values : audit trail of raw → parsed values
          confidence      : average confidence of component variables
          data_type       : the data_type parameter
          error           : error message if status == "ERROR"
        """
        # ── Step 1: Parse all variable values to numeric ──────────────────────
        parsed_vars: dict[str, float] = {}
        audit_trail: dict[str, dict] = {}
        confidences: list[float] = []

        for var_name, raw_value in variable_values.items():
            if raw_value is None:
                return self._error_result(
                    template=template,
                    error=f"Variable '{var_name}' has null value — "
                           "cannot evaluate expression.",
                    data_type=data_type,
                )
            try:
                numeric = self._parse_numeric(raw_value)
                parsed_vars[var_name] = numeric
                audit_trail[var_name] = {
                    "raw": raw_value,
                    "parsed": numeric,
                }
                confidences.append(1.0)   # confidence from extraction layer
            except (ValueError, AttributeError) as exc:
                return self._error_result(
                    template=template,
                    error=f"Cannot parse variable '{var_name}' = "
                           f"'{raw_value}': {exc}",
                    data_type=data_type,
                )

        # ── Step 2: Evaluate expression safely ───────────────────────────────
        evaluator = EvalWithCompoundTypes(
            names=parsed_vars,
            functions=_SAFE_FUNCTIONS,
        )
        try:
            numeric_result: float = evaluator.eval(template)
        except (FeatureNotAvailable, Exception) as exc:
            return self._error_result(
                template=template,
                error=f"Expression evaluation failed: {exc}",
                data_type=data_type,
            )

        # ── Step 3: Format result by data_type ────────────────────────────────
        computed_value = self._format_result(numeric_result, data_type)
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return {
            "status":          "SUCCESS",
            "computed_value":  computed_value,
            "numeric_result":  numeric_result,
            "template_used":   template,
            "variable_values": audit_trail,
            "confidence":      avg_confidence,
            "data_type":       data_type,
        }

    def evaluate_with_confidences(
        self,
        template: str,
        variable_values: dict[str, str | None],
        variable_confidences: dict[str, float],
        data_type: str = "monetary",
    ) -> dict:
        """
        Like evaluate(), but also incorporates per-variable confidence scores
        from the MLLM extraction layer into the audit trail.

        Parameters
        ----------
        template               : arithmetic expression string
        variable_values        : mapping of variable_name → raw string
        variable_confidences   : mapping of variable_name → confidence float
        data_type              : output formatting type

        Returns
        -------
        dict (same schema as evaluate())
        """
        result = self.evaluate(template, variable_values, data_type)
        if result["status"] == "SUCCESS":
            # Enrich audit trail with actual extraction confidences
            for var_name in result["variable_values"]:
                conf = variable_confidences.get(var_name, 1.0)
                result["variable_values"][var_name]["confidence"] = conf

            # Recompute average confidence using actual values
            confs = [
                variable_confidences.get(v, 1.0)
                for v in result["variable_values"]
            ]
            result["confidence"] = sum(confs) / len(confs) if confs else 0.0
        return result

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_numeric(raw: str) -> float:
        """
        Parse a raw extracted string to a float.

        Handles:
          - Currency symbols: "$1,500" → 1500.0
          - Commas in numbers: "1,500.00" → 1500.0
          - Percentage: "20%" → 20.0 (raw percentage, not 0.20)
          - Plain numbers: "150" → 150.0
        """
        cleaned = (
            raw.strip()
               .replace("$", "")
               .replace(",", "")
               .replace("%", "")
               .strip()
        )
        return float(cleaned)

    @staticmethod
    def _format_result(value: float, data_type: str) -> str:
        """Format a numeric result according to its data_type."""
        if data_type == "monetary":
            return f"${value:,.2f}"
        if data_type == "percentage":
            return f"{value:.2f}%"
        return str(value)

    @staticmethod
    def _error_result(template: str, error: str, data_type: str) -> dict:
        """Build a standardised ERROR result dict."""
        return {
            "status":          "ERROR",
            "computed_value":  None,
            "numeric_result":  None,
            "template_used":   template,
            "variable_values": {},
            "confidence":      0.0,
            "data_type":       data_type,
            "error":           error,
        }
