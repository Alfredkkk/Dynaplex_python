"""
Policy implementations for Zero-shot Lost Sales Inventory Control
"""

import numpy as np
from scipy import special
from typing import Dict, Any

from dynaplex.core.policy import Policy


class BaseStockPolicy(Policy):
    """
    Base stock policy for inventory control.
    
    Orders up to a target level, considering both on-hand inventory
    and pipeline inventory.
    """
    
    def __init__(self, mdp, config: Dict[str, Any]):
        """
        Initialize a base stock policy.
        
        Args:
            mdp: MDP to act on
            config: Configuration parameters including 'target_level'
        """
        super().__init__(mdp, config)
        
        # Get target level from config or calculate default
        self.target_level = self._config.get("target_level", None)
        if self.target_level is None:
            # Default heuristic: target = mean demand * (leadtime + 1) + safety stock
            # We compute this based on the mean demand and leadtime from the MDP
            mean_demand = np.mean(mdp.mean_demand)
            mean_leadtime = np.sum([i * p for i, p in enumerate(mdp.leadtime_probs)])
            safety_factor = 1.0  # Z-score for service level
            safety_stock = safety_factor * np.sqrt(mean_leadtime + 1) * np.mean(mdp.std_demand)
            self.target_level = int(mean_demand * (mean_leadtime + 1) + safety_stock)
    
    def get_action(self, state: Dict[str, Any]) -> int:
        """
        Get an action based on the base stock policy.
        
        Args:
            state: Current state
            
        Returns:
            Action to take (order quantity)
        """
        # Calculate inventory position
        inventory_position = state["inventory"] + state["pipeline"].sum()
        
        # Order up to target
        desired_order = max(0, self.target_level - inventory_position)
        
        # Ensure order is valid
        order = min(desired_order, state["order_constraint"])
        
        # Check system inventory constraint
        total_after_order = inventory_position + order
        if total_after_order > state["max_system_inv"]:
            order = max(0, state["max_system_inv"] - inventory_position)
        
        return int(order)


class MyopicPolicy(Policy):
    """
    Myopic policy for inventory control.
    
    Orders based on a single-period newsvendor solution,
    ignoring pipeline inventory and future impacts.
    """
    
    def __init__(self, mdp, config: Dict[str, Any]):
        """
        Initialize a myopic policy.
        
        Args:
            mdp: MDP to act on
            config: Configuration parameters
        """
        super().__init__(mdp, config)
    
    def get_action(self, state: Dict[str, Any]) -> int:
        """
        Get an action based on the myopic policy.
        
        Args:
            state: Current state
            
        Returns:
            Action to take (order quantity)
        """
        # Get current cycle index
        cycle_idx = state["period"] % state["cycle_length"]
        
        # Get demand distribution parameters
        mean_demand = state["mean_cycle_demand"][cycle_idx]
        std_demand = state["std_cycle_demand"][cycle_idx]
        
        # Critical ratio for newsvendor
        critical_ratio = state["p"] / (state["p"] + self._mdp.h)
        
        # Calculate order quantity using normal approximation
        z_score = np.sqrt(2) * special.erfinv(2 * critical_ratio - 1)  # Inverse CDF of standard normal
        target = mean_demand + z_score * std_demand
        
        # Adjust for current inventory
        desired_order = max(0, target - state["inventory"])
        
        # Ensure order is valid
        order = min(desired_order, state["order_constraint"])
        
        # Check system inventory constraint
        total_after_order = state["inventory"] + state["pipeline"].sum() + order
        if total_after_order > state["max_system_inv"]:
            order = max(0, state["max_system_inv"] - (state["inventory"] + state["pipeline"].sum()))
        
        return int(order)


def create_policy(mdp, id: str, **kwargs) -> Policy:
    """
    Create a policy for the lost sales inventory control MDP.
    
    Args:
        mdp: MDP to create policy for
        id: Policy identifier
        **kwargs: Additional configuration parameters
    
    Returns:
        Policy instance
    """
    if id.lower() == "base_stock":
        return BaseStockPolicy(mdp, kwargs)
    elif id.lower() == "myopic":
        return MyopicPolicy(mdp, kwargs)
    elif id.lower() == "random":
        from dynaplex.policies.random_policy import RandomPolicy
        return RandomPolicy(mdp, kwargs)
    else:
        raise ValueError(f"Unknown policy ID: {id}") 