"""Risk prediction result model."""

from dataclasses import dataclass
from models.risk_category import RiskCategory


@dataclass
class RiskPrediction:
    """Result of maintenance risk prediction for a file."""
    
    module: str
    risk_category: RiskCategory
    normalized_score: float
    raw_prediction: float
    
    @property
    def is_risky(self) -> bool:
        """Check if file is in a risky state (high or critical)."""
        return self.risk_category in [RiskCategory.HIGH, RiskCategory.CRITICAL]
        
    @property
    def needs_attention(self) -> bool:
        """Alias for is_risky for readability."""
        return self.is_risky
        
    def __str__(self) -> str:
        """String representation of the prediction."""
        return (
            f"{self.module}: {self.risk_category.display_name} "
            f"(score: {self.normalized_score:.4f})"
        )