"""Risk categories for maintenance degradation classification."""

from enum import Enum


class RiskCategory(Enum):
    """Risk categories based on degradation score thresholds."""
    
    SEVERELY_DEGRADED = "severely_degraded"
    DEGRADED = "degraded" 
    STABLE = "stable"
    IMPROVED = "improved"
    
    def __str__(self) -> str:
        return self.value
        
    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        names = {
            self.SEVERELY_DEGRADED: "Severely Degraded",
            self.DEGRADED: "Degraded",
            self.STABLE: "Stable", 
            self.IMPROVED: "Improved",
        }
        return names[self]
        
    @classmethod
    def from_score(cls, score: float) -> "RiskCategory":
        """Categorize a degradation score into risk levels.
        
        Args:
            score: Degradation score (typically between -0.5 and 0.5)
            
        Returns:
            RiskCategory based on score thresholds
            
        Thresholds based on training data distribution:
        - < 0.0: Improved (code quality improving)
        - 0.0-0.1: Stable (code quality stable)
        - 0.1-0.2: Degraded (code quality declining)
        - > 0.2: Severely Degraded (rapid quality decline)
        """
        if score < 0.0:
            return cls.IMPROVED
        elif score <= 0.1:
            return cls.STABLE
        elif score <= 0.2:
            return cls.DEGRADED
        else:
            return cls.SEVERELY_DEGRADED