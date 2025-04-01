"""
Markov Decision Process (MDP) base class
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional, Union


class MDP(ABC):
    """
    Base class for all Markov Decision Processes (MDPs).
    
    This abstract class defines the interface that all MDPs must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MDP with configuration parameters.
        
        Args:
            config: Dictionary containing MDP parameters
        """
        self._config = config
        self._discount_factor = config.get("discount_factor", 0.99)
        self._identifier = config.get("id", self.__class__.__name__)
        self._type_identifier = self.__class__.__name__
    
    @property
    def identifier(self) -> str:
        """Returns the identifier for this MDP."""
        return self._identifier
    
    @property
    def type_identifier(self) -> str:
        """Returns the type identifier for this MDP."""
        return self._type_identifier
    
    @property
    def discount_factor(self) -> float:
        """Returns the discount factor for this MDP."""
        return self._discount_factor
    
    def get_static_info(self) -> Dict[str, Any]:
        """
        Gets dictionary representing static information for this MDP.
        
        Returns:
            Dictionary containing static properties of the MDP
        """
        return self._config.copy()
    
    def is_infinite_horizon(self) -> bool:
        """
        Indicates whether the MDP is infinite or finite horizon.
        
        Returns:
            True if the MDP is infinite horizon, False otherwise
        """
        return True
    
    def num_valid_actions(self) -> int:
        """
        Returns the number of valid actions for this MDP.
        
        If action space is continuous, returns -1.
        """
        raise NotImplementedError("Subclasses must implement num_valid_actions")
    
    def provides_flat_features(self) -> bool:
        """
        Indicates whether this MDP provides flattened feature vectors.
        
        Returns:
            True if the MDP implements get_features method
        """
        return hasattr(self, "get_features") and callable(getattr(self, "get_features"))
    
    def num_flat_features(self) -> int:
        """
        Returns the number of features in the flattened feature vector.
        
        If the MDP doesn't provide flat features, returns 0.
        """
        return 0
    
    def get_policy(self, id: str = None, **kwargs) -> "Policy":
        """
        Get a policy for this MDP.
        
        Args:
            id: Optional identifier for a built-in policy
            **kwargs: Configuration parameters for the policy
        
        Returns:
            A policy instance
        """
        from dynaplex.core.policy import create_policy
        return create_policy(self, id=id, **kwargs)
    
    def list_policies(self) -> Dict[str, str]:
        """
        Lists key-value pairs (id, description) of available built-in policies.
        
        Returns:
            Dictionary mapping policy IDs to descriptions
        """
        return {}
    
    @abstractmethod
    def get_initial_state(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Get an initial state for the MDP.
        
        Args:
            seed: Optional random seed
        
        Returns:
            Dictionary representing the initial state
        """
        pass
    
    @abstractmethod
    def is_action_valid(self, state: Dict[str, Any], action: int) -> bool:
        """
        Check if an action is valid in the given state.
        
        Args:
            state: Current state
            action: Action to check
            
        Returns:
            True if the action is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_next_state_reward(
        self, 
        state: Dict[str, Any], 
        action: int,
        seed: Optional[int] = None
    ) -> Tuple[Dict[str, Any], float, bool]:
        """
        Get the next state and reward after taking an action.
        
        Args:
            state: Current state
            action: Action to take
            seed: Optional random seed
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        pass
    
    def get_features(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get a feature vector representation of the state.
        
        Args:
            state: State to get features for
            
        Returns:
            Numpy array of features
        """
        raise NotImplementedError("Feature extraction not implemented for this MDP") 