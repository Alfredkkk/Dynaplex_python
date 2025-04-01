"""
Neural network models for DynaPlex
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union


class MLP(nn.Module):
    """
    Multi-layer perceptron with customizable architecture.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [128, 128],
        activation: str = "relu",
        output_activation: Optional[str] = None,
        dropout: float = 0.0
    ):
        """
        Initialize MLP network.
        
        Args:
            input_size: Size of input features
            output_size: Size of output features
            hidden_sizes: List of hidden layer sizes
            activation: Activation function to use
            output_activation: Activation function to use on output layer
            dropout: Dropout probability
        """
        super().__init__()
        
        # Set activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Set output activation function
        if output_activation == "relu":
            self.output_activation = F.relu
        elif output_activation == "tanh":
            self.output_activation = torch.tanh
        elif output_activation == "sigmoid":
            self.output_activation = torch.sigmoid
        elif output_activation == "softmax":
            self.output_activation = lambda x: F.softmax(x, dim=-1)
        elif output_activation is None:
            self.output_activation = lambda x: x  # Identity
        else:
            raise ValueError(f"Unsupported output activation function: {output_activation}")
        
        # Create layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights of linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
    
    def forward(self, x):
        """Forward pass through the network."""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.layers[-1](x)
        x = self.output_activation(x)
        
        return x


class DuelingMLP(nn.Module):
    """
    Dueling network architecture for Q-learning.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [128, 128],
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Initialize Dueling network.
        
        Args:
            input_size: Size of input features
            output_size: Size of output features (action space)
            hidden_sizes: List of hidden layer sizes
            activation: Activation function to use
            dropout: Dropout probability
        """
        super().__init__()
        
        # Set activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Feature extraction layers
        self.feature_layers = nn.ModuleList()
        self.feature_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        for i in range(len(hidden_sizes) - 2):
            self.feature_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Value stream
        self.value_hidden = nn.Linear(hidden_sizes[-2], hidden_sizes[-1])
        self.value = nn.Linear(hidden_sizes[-1], 1)
        
        # Advantage stream
        self.advantage_hidden = nn.Linear(hidden_sizes[-2], hidden_sizes[-1])
        self.advantage = nn.Linear(hidden_sizes[-1], output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights of linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Feature extraction
        for layer in self.feature_layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Value stream
        value = self.value_hidden(x)
        value = self.activation(value)
        value = self.value(value)
        
        # Advantage stream
        advantage = self.advantage_hidden(x)
        advantage = self.activation(advantage)
        advantage = self.advantage(advantage)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def create_network(
    input_size: int,
    output_size: int,
    config: Dict[str, Any] = None
) -> nn.Module:
    """
    Factory function to create a neural network.
    
    Args:
        input_size: Size of input features
        output_size: Size of output features
        config: Configuration parameters
    
    Returns:
        Neural network module
    """
    config = config or {}
    network_type = config.get("type", "mlp").lower()
    
    if network_type == "mlp":
        return MLP(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=config.get("hidden_sizes", [128, 128]),
            activation=config.get("activation", "relu"),
            output_activation=config.get("output_activation", None),
            dropout=config.get("dropout", 0.0)
        )
    elif network_type == "dueling":
        return DuelingMLP(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=config.get("hidden_sizes", [128, 128, 128]),
            activation=config.get("activation", "relu"),
            dropout=config.get("dropout", 0.0)
        )
    else:
        raise ValueError(f"Unsupported network type: {network_type}") 