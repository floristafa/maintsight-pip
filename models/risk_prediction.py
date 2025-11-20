"""Risk prediction result model."""

from dataclasses import dataclass
from models.risk_category import RiskCategory


@dataclass
class RiskPrediction:
    """Result of maintenance risk prediction for a file."""
    
    module: str
    risk_category: RiskCategory
    degradation_score: float
    raw_prediction: float
    
    @property
    def is_degraded(self) -> bool:
        """Check if file is in a degraded state (degraded or severely degraded)."""
        return self.risk_category in [RiskCategory.DEGRADED, RiskCategory.SEVERELY_DEGRADED]
        
    @property
    def needs_attention(self) -> bool:
        """Alias for is_degraded for readability."""
        return self.is_degraded
        
    def __str__(self) -> str:
        """String representation of the prediction."""
        return (
            f"{self.module}: {self.risk_category.display_name} "
            f"(score: {self.degradation_score:.4f})"
        )