"""Tests for RiskCategory enum."""

import pytest
from models.risk_category import RiskCategory


class TestRiskCategory:
    """Test cases for RiskCategory enum."""
    
    def test_enum_values(self):
        """Test enum values are correct."""
        assert RiskCategory.SEVERELY_DEGRADED.value == "severely_degraded"
        assert RiskCategory.DEGRADED.value == "degraded"
        assert RiskCategory.STABLE.value == "stable"
        assert RiskCategory.IMPROVED.value == "improved"
        
    def test_display_names(self):
        """Test display names are formatted correctly."""
        assert RiskCategory.SEVERELY_DEGRADED.display_name == "Severely Degraded"
        assert RiskCategory.DEGRADED.display_name == "Degraded"
        assert RiskCategory.STABLE.display_name == "Stable"
        assert RiskCategory.IMPROVED.display_name == "Improved"
        
    def test_from_score_severely_degraded(self):
        """Test severely degraded categorization (> 0.2)."""
        assert RiskCategory.from_score(0.3) == RiskCategory.SEVERELY_DEGRADED
        assert RiskCategory.from_score(0.25) == RiskCategory.SEVERELY_DEGRADED
        assert RiskCategory.from_score(1.0) == RiskCategory.SEVERELY_DEGRADED
        
    def test_from_score_degraded(self):
        """Test degraded categorization (0.1 < score <= 0.2)."""
        assert RiskCategory.from_score(0.15) == RiskCategory.DEGRADED
        assert RiskCategory.from_score(0.11) == RiskCategory.DEGRADED
        assert RiskCategory.from_score(0.2) == RiskCategory.DEGRADED
        
    def test_from_score_stable(self):
        """Test stable categorization (0.0 <= score <= 0.1)."""
        assert RiskCategory.from_score(0.05) == RiskCategory.STABLE
        assert RiskCategory.from_score(0.0) == RiskCategory.STABLE
        assert RiskCategory.from_score(0.09) == RiskCategory.STABLE
        assert RiskCategory.from_score(0.1) == RiskCategory.STABLE
        
    def test_from_score_improved(self):
        """Test improved categorization (< 0.0)."""
        assert RiskCategory.from_score(-0.1) == RiskCategory.IMPROVED
        assert RiskCategory.from_score(-0.01) == RiskCategory.IMPROVED
        assert RiskCategory.from_score(-1.0) == RiskCategory.IMPROVED
        
    def test_string_representation(self):
        """Test string conversion."""
        assert str(RiskCategory.IMPROVED) == "improved"
        assert str(RiskCategory.STABLE) == "stable"