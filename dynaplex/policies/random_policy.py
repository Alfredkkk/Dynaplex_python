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
        # Find valid actions
        valid_actions = []
        for action in range(self._mdp.num_valid_actions()):
            if self._mdp.is_action_valid(state, action):
                valid_actions.append(action)
        
        # If no valid actions, return 0 as a fallback
        if not valid_actions:
            return 0
        
        # Return random valid action
        return np.random.choice(valid_actions) 