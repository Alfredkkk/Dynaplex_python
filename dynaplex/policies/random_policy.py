"""
Random policy implementation
"""

import numpy as np
from typing import Dict, Any

from dynaplex.core.policy import Policy


class RandomPolicy(Policy):
    """
    Random policy that selects random valid actions.
    """
    
    def __init__(self, mdp, config: Dict[str, Any] = None):
        """
        Initialize a random policy.
        
        Args:
            mdp: MDP to act on
            config: Configuration parameters
        """
        super().__init__(mdp, config or {})
        self._seed = self._config.get("seed", None)
        
        # Set random seed if provided
        if self._seed is not None:
            np.random.seed(self._seed)
    
    def get_action(self, state: Dict[str, Any]) -> int:
        """
        Get a random valid action.
        
        Args:
            state: Current state
            
        Returns:
            Random valid action
        """
        # Find valid actions, but limit to the state's order_constraint
        max_order = min(state.get("order_constraint", 0), self._mdp.num_valid_actions() - 1)
        valid_actions = []
        
        for action in range(max_order + 1):
            if self._mdp.is_action_valid(state, action):
                valid_actions.append(action)
        
        # If no valid actions, return 0 as a fallback
        if not valid_actions:
            print(f"Warning: No valid actions found, using 0 as fallback. max_order={max_order}")
            return 0
        
        # Return random valid action
        chosen_action = np.random.choice(valid_actions)
        
        # Double-check validity (debugging)
        if not self._mdp.is_action_valid(state, chosen_action):
            print(f"Warning: Chosen action {chosen_action} from {valid_actions} is not valid!")
            # Force return a safe action
            return 0
            
        # Explicitly ensure integer return
        return int(chosen_action) 