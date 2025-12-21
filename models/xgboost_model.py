"""XGBoost model data structures."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass 
class XGBoostTree:
    """Single XGBoost tree structure."""
    
    left_children: List[int]
    right_children: List[int]
    split_indices: List[int] 
    split_conditions: List[float]
    base_weights: List[float]
    
    def predict(self, features: List[float]) -> float:
        """Traverse tree to get prediction for given features."""
        node_id = 0
        
        while True:
            # Check if leaf node
            if self.left_children[node_id] == -1:
                return self.base_weights[node_id]
                
            # Get feature value and split condition
            feature_idx = self.split_indices[node_id]
            feature_value = features[feature_idx]
            split_condition = self.split_conditions[node_id]
            
            # Decide which child to visit
            if feature_value < split_condition:
                node_id = self.left_children[node_id]
            else:
                node_id = self.right_children[node_id]


@dataclass
class XGBoostModel:
    """XGBoost model structure containing trees and metadata."""
    
    feature_names: List[str]
    feature_count: int
    trees: List[XGBoostTree]
    base_score: float = 0.5
    model_data: Optional[Dict[str, Any]] = None
    
    def predict(self, features: List[float]) -> float:
        """Make prediction using all trees in the ensemble.
        
        Args:
            features: Input feature vector
            
        Returns:
            Raw prediction score
        """
        if len(features) != self.feature_count:
            raise ValueError(
                f"Expected {self.feature_count} features, got {len(features)}"
            )
            
        # Start with base score
        score = self.base_score
        
        # Add predictions from all trees
        for tree in self.trees:
            score += tree.predict(features)
            
        return score