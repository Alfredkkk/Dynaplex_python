"""
Policy base classes and utilities
"""

import os
import json
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List

class Policy(ABC):
    """
    Base class for all policies.
    
    This abstract class defines the interface that all policies must implement.
    """
    
    def __init__(self, mdp, config: Dict[str, Any]):
        """
        Initialize a policy for the given MDP.
        
        Args:
            mdp: The MDP this policy operates on
            config: Configuration parameters
        """
        self._mdp = mdp
        self._config = config
        self._type_identifier = self.__class__.__name__
    
    @property
    def type_identifier(self) -> str:
        """Returns the type identifier for this policy."""
        return self._type_identifier
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration parameters for this policy.
        
        Returns:
            Dictionary of configuration parameters
        """
        return self._config.copy()
    
    @abstractmethod
    def get_action(self, state: Dict[str, Any]) -> int:
        """
        Get the action to take in the given state.
        
        Args:
            state: Current state
            
        Returns:
            Action to take
        """
        pass
    
    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the policy to disk.
        
        Args:
            path: Path to save the policy to
            metadata: Optional metadata to save with the policy
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save metadata
        meta = metadata or {}
        meta.update({
            "type_identifier": self.type_identifier,
            "config": self.get_config()
        })
        
        with open(f"{path}.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        # Subclasses should override this to save additional data if needed


class NeuralNetworkPolicy(Policy):
    """Base class for neural network policies."""
    
    def __init__(self, mdp, model, config: Dict[str, Any]):
        """
        Initialize a neural network policy.
        
        Args:
            mdp: The MDP this policy operates on
            model: PyTorch model
            config: Configuration parameters
        """
        super().__init__(mdp, config)
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_action(self, state: Dict[str, Any]) -> int:
        """
        Get the action to take in the given state.
        
        Args:
            state: Current state
            
        Returns:
            Action to take
        """
        # Get features from state
        features = self._mdp.get_features(state)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Run forward pass
        with torch.no_grad():
            action_values = self.model(features_tensor)
            action = int(action_values.argmax(dim=1).item())
        
        # Ensure action is valid
        valid_action = action
        if not self._mdp.is_action_valid(state, action):
            # Find first valid action as fallback
            for a in range(self._mdp.num_valid_actions()):
                if self._mdp.is_action_valid(state, a):
                    valid_action = a
                    break
        
        return valid_action
    
    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the policy to disk.
        
        Args:
            path: Path to save the policy to
            metadata: Optional metadata to save with the policy
        """
        super().save(path, metadata)
        
        # Save model
        self.model.to("cpu")
        torch.jit.script(self.model).save(f"{path}.pth")
        self.model.to(self.device)


def create_policy(mdp, id: Optional[str] = None, **kwargs) -> Policy:
    """
    Factory function to create a policy for the given MDP.
    
    Args:
        mdp: MDP instance
        id: Optional identifier for built-in policy
        **kwargs: Configuration parameters
    
    Returns:
        Policy instance
    """
    if id:
        # If ID is provided, try to get built-in policy
        available_policies = mdp.list_policies()
        if id not in available_policies:
            raise ValueError(f"Policy '{id}' not found for MDP '{mdp.identifier}'")
        
        # Import the policy module for the MDP
        module_path = f"dynaplex.models.{mdp.type_identifier.lower()}.policies"
        try:
            module = __import__(module_path, fromlist=["create_policy"])
            return module.create_policy(mdp, id, **kwargs)
        except (ImportError, AttributeError):
            raise ValueError(f"Failed to load policy '{id}' for MDP '{mdp.identifier}'")
    
    # If no ID provided, create a default policy
    from dynaplex.policies.random_policy import RandomPolicy
    return RandomPolicy(mdp, kwargs)


def load_policy_from_file(mdp, path: str) -> Policy:
    """
    Load a policy from a file.
    
    Args:
        mdp: MDP instance
        path: Path to the policy file (without extension)
    
    Returns:
        Policy instance
    """
    # Load metadata
    with open(f"{path}.json", "r") as f:
        metadata = json.load(f)
    
    # Check if it's a neural network policy
    if os.path.exists(f"{path}.pth"):
        # Load the model
        model = torch.jit.load(f"{path}.pth")
        
        # Create a NeuralNetworkPolicy
        return NeuralNetworkPolicy(mdp, model, metadata.get("config", {}))
    
    # Otherwise, use the type_identifier to create the appropriate policy
    type_id = metadata.get("type_identifier")
    if not type_id:
        raise ValueError(f"Invalid policy file: missing type_identifier in {path}.json")
    
    # Import the policy module based on type_identifier
    module_parts = type_id.split(".")
    if len(module_parts) == 1:
        # Try to find in policies package
        try:
            module_path = f"dynaplex.policies.{type_id.lower()}"
            module = __import__(module_path, fromlist=[type_id])
            policy_class = getattr(module, type_id)
            return policy_class(mdp, metadata.get("config", {}))
        except (ImportError, AttributeError):
            # Try MDP-specific policies
            module_path = f"dynaplex.models.{mdp.type_identifier.lower()}.policies"
            try:
                module = __import__(module_path, fromlist=[type_id])
                policy_class = getattr(module, type_id)
                return policy_class(mdp, metadata.get("config", {}))
            except (ImportError, AttributeError):
                raise ValueError(f"Could not find policy class '{type_id}'")
    else:
        # Full module path specified
        module_path = ".".join(module_parts[:-1])
        class_name = module_parts[-1]
        try:
            module = __import__(module_path, fromlist=[class_name])
            policy_class = getattr(module, class_name)
            return policy_class(mdp, metadata.get("config", {}))
        except (ImportError, AttributeError):
            raise ValueError(f"Could not find policy class '{type_id}'") 