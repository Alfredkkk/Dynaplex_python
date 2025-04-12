"""
Multi-Layer Perceptron (MLP) implementation for DynaPlex.

This module provides an implementation of a standard MLP neural network
that can be used for policy networks in the context of reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union


class MLP(nn.Module):
    """
    Multi-Layer Perceptron neural network.
    
    Implements a standard MLP with customizable hidden layer sizes and activations.
    Used for policy networks in DynaPlex.
    
    Attributes:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        hidden_sizes (List[int]): List of hidden layer sizes
        activation (str): Activation function name
        output_activation (str): Output activation function name
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int] = [256, 128, 128, 128],
        activation: str = "relu",
        output_activation: Optional[str] = None
    ):
        """
        Initialize the MLP.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output (usually action space size)
            hidden_sizes: List of hidden layer sizes
            activation: Activation function for hidden layers ('relu', 'tanh', 'sigmoid', etc.)
            output_activation: Activation function for output layer
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation
        self.output_activation_name = output_activation
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input to first hidden layer
        self.layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_dim)
        
        # Activation function
        self.activation = self._get_activation(activation)
        self.output_activation = self._get_activation(output_activation) if output_activation else None
    
    def _get_activation(self, name: Optional[str]) -> Optional[nn.Module]:
        """Get activation function by name."""
        if name is None:
            return None
        
        name = name.lower()
        if name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'leaky_relu':
            return nn.LeakyReLU()
        elif name == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Forward through hidden layers with activation
        for layer in self.layers:
            x = self.activation(layer(x))
        
        # Output layer
        x = self.output_layer(x)
        
        # Apply output activation if specified
        if self.output_activation:
            x = self.output_activation(x)
        
        return x
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action probabilities and values for a given state.
        
        Args:
            state: State tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (action_probs, value) where action_probs has shape (batch_size, output_dim)
        """
        logits = self.forward(state)
        
        # For discrete action spaces
        action_probs = F.softmax(logits, dim=-1)
        
        return action_probs, logits
    
    def save(self, path: str) -> None:
        """
        Save model weights to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path: str, device: Optional[torch.device] = None) -> None:
        """
        Load model weights from a file.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
    
    def to_json(self) -> Dict:
        """
        Convert model architecture to JSON format.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            'type': 'mlp',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_layers': self.hidden_sizes,
            'activation': self.activation_name,
            'output_activation': self.output_activation_name
        }
    
    @classmethod
    def from_json(cls, config: Dict, device: Optional[torch.device] = None) -> 'MLP':
        """
        Create model from JSON configuration.
        
        Args:
            config: Dictionary with model configuration
            device: Device to load the model to
            
        Returns:
            New MLP instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        hidden_sizes = config.get('hidden_layers', [256, 128, 128, 128])
        if isinstance(hidden_sizes, dict) and 'hidden_layers' in hidden_sizes:
            hidden_sizes = hidden_sizes['hidden_layers']
        
        model = cls(
            input_dim=config.get('input_dim', config.get('num_inputs', 38)),
            output_dim=config.get('output_dim', config.get('num_outputs', 130)),
            hidden_sizes=hidden_sizes,
            activation=config.get('activation', 'relu'),
            output_activation=config.get('output_activation', None)
        )
        
        model.to(device)
        return model 