"""
Neural Network Policy implementation for DynaPlex.

This module provides a policy implementation that uses a neural network
to make decisions in an MDP environment.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from dynaplex.nn.mlp import MLP


class NeuralNetworkPolicy:
    """
    A policy that uses a neural network to make decisions.
    
    This policy uses a trained neural network to choose actions in an MDP.
    It supports both discrete and continuous action spaces.
    
    Attributes:
        mdp: The MDP this policy is designed for
        model: The neural network model
        device: The device to run the model on
    """
    
    def __init__(self, mdp, model: Optional[Union[MLP, torch.nn.Module]] = None):
        """
        Initialize a neural network policy.
        
        Args:
            mdp: The MDP this policy is designed for
            model: A pre-trained neural network model
        """
        self.mdp = mdp
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model is not None:
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        
        # If no model provided, create a default one
        if self.model is None:
            state_dim = self.mdp.get_state_dimension()
            action_dim = self.mdp.get_action_dimension()
            
            self.model = MLP(
                input_dim=state_dim,
                output_dim=action_dim,
                hidden_sizes=[256, 128, 128, 128],
                activation="relu"
            ).to(self.device)
    
    def get_action(self, state: Dict[str, Any]) -> int:
        """
        Get the action for a given state.
        
        Args:
            state: The current state
            
        Returns:
            The action to take
        """
        # Convert state to features
        features = self.mdp.get_features(state)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Forward pass through the model
        with torch.no_grad():
            # 检查是否是TorchScript模型
            if isinstance(self.model, torch.jit.ScriptModule) or hasattr(self.model, 'scripted_model'):
                # 直接调用TorchScript模型
                if hasattr(self.model, 'scripted_model'):
                    # 使用TorchScriptWrapper中的scripted_model
                    logits = self.model.scripted_model(features_tensor)
                else:
                    # 直接调用TorchScript模型
                    logits = self.model(features_tensor)
            else:
                # 正常调用标准PyTorch模型
                logits = self.model(features_tensor)
            
            # For discrete action space, take the argmax
            action = torch.argmax(logits, dim=-1).item()
        
        return action
    
    def get_actions(self, states: List[Dict[str, Any]]) -> List[int]:
        """
        Get actions for a batch of states.
        
        Args:
            states: A list of states
            
        Returns:
            A list of actions
        """
        # Convert states to features
        features = [self.mdp.get_features(state) for state in states]
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Forward pass through the model
        with torch.no_grad():
            # 检查是否是TorchScript模型
            if isinstance(self.model, torch.jit.ScriptModule) or hasattr(self.model, 'scripted_model'):
                # 直接调用TorchScript模型
                if hasattr(self.model, 'scripted_model'):
                    # 使用TorchScriptWrapper中的scripted_model
                    logits = self.model.scripted_model(features_tensor)
                else:
                    # 直接调用TorchScript模型
                    logits = self.model(features_tensor)
            else:
                # 正常调用标准PyTorch模型
                logits = self.model(features_tensor)
            
            # For discrete action space, take the argmax
            actions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        return actions.tolist()
    
    def save(self, path: str) -> None:
        """
        Save the policy model to a file.
        
        Args:
            path: The path to save the model to
        """
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """
        Load the policy model from a file.
        
        Args:
            path: The path to load the model from
        """
        self.model.load(path, self.device)
        self.model.eval()  # Set to evaluation mode
    
    def to_json(self) -> Dict:
        """
        Convert policy to JSON representation.
        
        Returns:
            A dictionary containing the policy configuration
        """
        return {
            "type": "neural_network",
            "model": self.model.to_json() if hasattr(self.model, "to_json") else None
        }
    
    @classmethod
    def from_json(cls, mdp, config: Dict) -> 'NeuralNetworkPolicy':
        """
        Create a policy from a JSON configuration.
        
        Args:
            mdp: The MDP this policy is designed for
            config: A dictionary containing the policy configuration
            
        Returns:
            A new NeuralNetworkPolicy instance
        """
        model_config = config.get("model", {})
        model = MLP.from_json(model_config)
        
        return cls(mdp, model) 