"""
Feature Adapter for DynaPlex.

This module provides adapter classes to ensure compatibility between
different feature dimensions in models and MDPs.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Union


class FeatureAdapter:
    """
    Adapter class to handle feature dimension mismatches.
    
    This class transforms features from one dimension to another,
    allowing models trained with a specific feature dimension to
    work with MDPs that provide features of a different dimension.
    """
    
    @staticmethod
    def adapt_15d_to_38d(features: Union[List[float], np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Adapt 15-dimensional features to 38-dimensional features.
        
        This method pads the 15-dimensional feature vector with zeros
        to match the 38-dimensional input expected by the GC-LSN model.
        
        Args:
            features: The original 15-dimensional features
            
        Returns:
            A 38-dimensional feature vector
        """
        # Convert input to numpy array if it's a list or tensor
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)
        elif isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        # Create a 38-dimensional vector filled with zeros
        adapted_features = np.zeros(38, dtype=np.float32)
        
        # Copy the original features into the first 15 positions
        adapted_features[:min(15, len(features))] = features[:min(15, len(features))]
        
        return adapted_features


class FeatureAdapterPolicy:
    """
    A policy wrapper that adapts features before passing them to the model.
    
    This policy wrapper transforms the features from the MDP to match
    the dimensions expected by the model.
    
    Attributes:
        policy: The original policy
        adapter_func: The function to adapt features
    """
    
    def __init__(self, policy, adapter_func=None):
        """
        Initialize a feature adapter policy.
        
        Args:
            policy: The policy to wrap
            adapter_func: The function to adapt features
        """
        self.policy = policy
        self.adapter_func = adapter_func or FeatureAdapter.adapt_15d_to_38d
        self.type_identifier = "feature_adapter_" + getattr(policy, "type_identifier", "nn_policy")
        self.mdp = policy.mdp
        
    def get_action(self, state: Dict[str, Any]) -> int:
        """
        Get the action for a given state, adapting features first.
        
        Args:
            state: The current state
            
        Returns:
            The action to take
        """
        # Get original features
        mdp = self.policy.mdp
        original_features = mdp.get_features(state)
        
        # Store the original get_features method
        original_get_features = mdp.get_features
        
        # Override get_features method temporarily
        mdp.get_features = lambda s: self.adapter_func(original_get_features(s))
        
        # Get action using the adapted features
        action = self.policy.get_action(state)
        
        # Restore original get_features method
        mdp.get_features = original_get_features
        
        # Ensure the action is valid for the MDP
        if hasattr(mdp, 'num_valid_actions'):
            # Clip action to valid range
            action = max(0, min(action, mdp.num_valid_actions() - 1))
        
        # Additional validation if the MDP has an is_action_valid method
        if hasattr(mdp, 'is_action_valid'):
            # Make sure the action is valid, or find the closest valid action
            if not mdp.is_action_valid(state, action):
                # Try to find a valid action
                valid_action = self._find_closest_valid_action(mdp, state, action)
                if valid_action is not None:
                    action = valid_action
        
        return action
    
    def _find_closest_valid_action(self, mdp, state, action):
        """
        Find the closest valid action to the given action.
        
        Args:
            mdp: The MDP
            state: The current state
            action: The desired action
            
        Returns:
            The closest valid action, or None if no valid action is found
        """
        # Try actions in decreasing order, starting from the original action
        for offset in range(action + 1):
            # Try action - offset
            if action - offset >= 0 and mdp.is_action_valid(state, action - offset):
                return action - offset
            
            # Try action + offset
            if action + offset < mdp.num_valid_actions() and mdp.is_action_valid(state, action + offset):
                return action + offset
        
        # If no valid action is found, return 0 (do nothing) as a fallback
        if mdp.is_action_valid(state, 0):
            return 0
        
        return None
        
    def get_actions(self, states: List[Dict[str, Any]]) -> List[int]:
        """
        Get actions for a batch of states, adapting features first.
        
        Args:
            states: A list of states
            
        Returns:
            A list of actions
        """
        # Get original get_features method
        mdp = self.policy.mdp
        original_get_features = mdp.get_features
        
        # Override get_features method temporarily
        mdp.get_features = lambda s: self.adapter_func(original_get_features(s))
        
        # Get actions using the adapted features
        actions = self.policy.get_actions(states)
        
        # Restore original get_features method
        mdp.get_features = original_get_features
        
        # Validate actions
        validated_actions = []
        for state, action in zip(states, actions):
            # Ensure the action is valid for the MDP
            if hasattr(mdp, 'num_valid_actions'):
                # Clip action to valid range
                action = max(0, min(action, mdp.num_valid_actions() - 1))
            
            # Additional validation if the MDP has an is_action_valid method
            if hasattr(mdp, 'is_action_valid'):
                # Make sure the action is valid, or find the closest valid action
                if not mdp.is_action_valid(state, action):
                    # Try to find a valid action
                    valid_action = self._find_closest_valid_action(mdp, state, action)
                    if valid_action is not None:
                        action = valid_action
            
            validated_actions.append(action)
        
        return validated_actions 