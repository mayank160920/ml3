"""
-----------------------------------------
Rule-based value normaliser for the fast-path validation pre-processor.

Runs BEFORE any MLLM validation call. Entity pairs where both normalised
values are identical are immediately recorded as MATCH at zero additional
API cost — handling the large fraction of format-only differences between
SBC and Benefit Grid representations.

Normalisation rules
-------------------
Monetary  : "$1,500" / "1500" / "$1,500.00" → "1500.00 USD"
Percentage: "20%" / "0.20" / "20 percent"  → "20.0%"
Coverage  : "No charge" / "Covered in full" → "0.00 USD"
            "Not covered" / "Member pays 100%" → "MEMBER_PAYS_100_PERCENT"
Text      : lowercase + whitespace collapse
"""
from __future__ import annotations

import re


# ── Coverage synonym mappings ─────────────────────────────────────────────────

_ZERO_COST_PHRASES: frozenset[str] = frozenset({
    "no charge",
    "covered in full",
    "fully covered",
    "free",
    "no cost",
    "$0",
    "$0.00",
    "0",
    "0.00",
    "zero",
    "no copay",
    "no cost sharing",
    "not applicable",
    "n/a",
})

_MEMBER_PAYS_ALL_PHRASES: frozenset[str] = frozenset({
    "not covered",
    "member pays 100%",
    "member pays 100 percent",
    "not a covered benefit",
    "excluded",
    "not a benefit",
    "0% covered",
    "100% member responsibility",
})


class ValueNormalizer:
    """
    Normalises extracted entity values into canonical forms for comparison.

    The normaliser is purely rule-based — no LLM involved. It is designed
    to be fast, deterministic, and auditable.

    Usage
    -----
    >>> normalizer = ValueNormalizer()
    >>> normalizer.normalize("$1,500", data_type="monetary")
    '1500.00 USD'
    >>> normalizer.normalize("20 percent", data_type="percentage")
    '20.0%'
    >>> normalizer.normalize("Covered in full", data_type="coverage_classification")
    '0.00 USD'
    """

    def normalize(
        self,
        value: str | None,
        data_type: str = "text",
    ) -> str | None:
        """
        Normalise a single extracted value.

        Parameters
        ----------
        value     : raw extracted value string (may be None)
        data_type : entity data type controlling normalisation logic
                    ('monetary' | 'percentage' | 'coverage_classification' | 'text')

        Returns
        -------
        Normalised string, or None if input is None
        """
        if value is None:
            return None

        stripped = value.strip()
        if not stripped:
            return None

        if data_type == "monetary":
            return self._normalize_monetary(stripped)
        if data_type == "percentage":
            return self._normalize_percentage(stripped)
        if data_type == "coverage_classification":
            return self._normalize_coverage(stripped)
        return self._normalize_text(stripped)

    def normalize_pair(
        self,
        value_a: str | None,
        value_b: str | None,
        data_type: str = "text",
    ) -> tuple[str | None, str | None]:
        """
        Normalise a pair of values using the same data_type.

        Parameters
        ----------
        value_a   : value from Document A
        value_b   : value from Document B
        data_type : entity data type

        Returns
        -------
        (normalised_a, normalised_b) tuple
        """
        return (
            self.normalize(value_a, data_type),
            self.normalize(value_b, data_type),
        )

    def is_fast_path_match(
        self,
        value_a: str | None,
        value_b: str | None,
        data_type: str = "text",
    ) -> bool:
        """
        Return True if both values normalise to the same canonical form.

        Fast-path matches skip the MLLM validation call entirely.

        Parameters
        ----------
        value_a   : value from Document A
        value_b   : value from Document B
        data_type : entity data type

        Returns
        -------
        bool
        """
        # Both null → treat as INELIGIBLE, not a match
        if value_a is None and value_b is None:
            return False

        norm_a, norm_b = self.normalize_pair(value_a, value_b, data_type)
        if norm_a is None or norm_b is None:
            return False

        return norm_a == norm_b

    # ── private normalisation methods ─────────────────────────────────────────

    def _normalize_monetary(self, value: str) -> str:
        """
        Normalise a monetary value to 'NNNN.NN USD' canonical form.

        Handles:
          "$1,500"     → "1500.00 USD"
          "1500"       → "1500.00 USD"
          "$1,500.00"  → "1500.00 USD"
          "1,500 USD"  → "1500.00 USD"
          "No charge"  → "0.00 USD"

        Returns original value (lowercased) if parsing fails.
        """
        lower = value.lower()

        # Coverage synonym → zero cost
        if lower in _ZERO_COST_PHRASES:
            return "0.00 USD"

        # Strip currency indicators and formatting
        cleaned = (
            value
            .replace("$", "")
            .replace(",", "")
            .replace("USD", "")
            .replace("usd", "")
            .strip()
        )

        try:
            amount = float(cleaned)
            return f"{amount:.2f} USD"
        except ValueError:
            return lower

    def _normalize_percentage(self, value: str) -> str:
        """
        Normalise a percentage value to 'NN.N%' canonical form.

        Handles:
          "20%"         → "20.0%"
          "20 percent"  → "20.0%"
          "0.20"        → "20.0%"   (decimal fraction detected)
          "20.0"        → "20.0%"

        Returns original value (lowercased) if parsing fails.
        """
        lower = value.lower().strip()
        cleaned = (
            lower
            .replace("percent", "")
            .replace("%", "")
            .strip()
        )

        try:
            num = float(cleaned)
            # Convert decimal fractions (0.20 → 20.0%)
            if 0 < num <= 1.0 and "." in cleaned:
                num = num * 100
            return f"{num:.1f}%"
        except ValueError:
            return lower

    def _normalize_coverage(self, value: str) -> str:
        """
        Normalise coverage classification values.

        Maps known zero-cost and full-cost phrases to canonical forms.
        Falls back to lowercased text for unrecognised values.
        """
        lower = value.lower().strip()

        if lower in _ZERO_COST_PHRASES:
            return "0.00 USD"

        if lower in _MEMBER_PAYS_ALL_PHRASES:
            return "MEMBER_PAYS_100_PERCENT"

        # Try monetary normalisation first (e.g., "$25 copay")
        if "$" in lower or re.search(r"\d+\.\d+", lower):
            return self._normalize_monetary(value)

        # Try percentage normalisation (e.g., "20% coinsurance")
        if "%" in lower or "percent" in lower:
            return self._normalize_percentage(value)

        return self._normalize_text(value)

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Lowercase and collapse whitespace for text comparison."""
        return " ".join(value.lower().split())
