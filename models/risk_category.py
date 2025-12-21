"""Risk categories for maintenance degradation classification."""

from enum import Enum


class RiskCategory(Enum):
    """Risk categories based on degradation score thresholds."""
    
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    
    def __str__(self) -> str:
        return self.value
        
    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        names = {
            self.CRITICAL: "Critical",
            self.HIGH: "High",
            self.MEDIUM: "Medium", 
            self.LOW: "Low",
        }
        return names[self]
        
    @classmethod
    def from_score(cls, score: float) -> "RiskCategory":
        """Categorize a risk score into risk levels.
        
        Args:
            score: Risk score (typically between 0 and 100)
            
        Returns:
            RiskCategory based on score thresholds
            
        Thresholds based on training data distribution:
        - < 50: Low (code quality improving)
        - 50-70: Medium (code quality stable)
        - 70-90: High (code quality declining)
        - > 90: Critical (rapid quality decline)
        """
        if score < 50:
            return cls.LOW
        elif score < 70:
            return cls.MEDIUM
        elif score < 90:
            return cls.HIGH
        else:
            return cls.CRITICAL