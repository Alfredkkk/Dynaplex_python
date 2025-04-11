from typing import Dict, List, Union

import torch
from torch.nn import Linear, ReLU, Sequential, LayerNorm, Module


class ActorMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Union[int, List[int]], 
                 min_val=torch.finfo(torch.float).min, activation=ReLU):
        """
        Neural network for policy implementation with flexible hidden layers.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output actions
            hidden_dim: Either a single integer for one hidden layer or a list of integers for multiple hidden layers
            min_val: Minimum value to use for masking invalid actions
            activation: Activation function to use
        """
        super(ActorMLP, self).__init__()
        self.min_val = min_val
        self.output_dim = output_dim
        
        # Build layers dynamically based on hidden_dim
        layers = []
        
        # Convert single int to list for consistent processing
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = hidden_dim
            
        # First layer from input_dim to first hidden dim
        layers.append(Linear(input_dim, hidden_dims[0]))
        layers.append(LayerNorm(hidden_dims[0]))
        layers.append(activation())
        
        # Create additional hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(activation())
        
        # Final layer to output
        layers.append(Linear(hidden_dims[-1], output_dim))
        
        self.actor = Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network - this version is TorchScript compatible
        and assumes tensor input for inference
        
        Args:
            observations: Tensor with observation features
            
        Returns:
            Tensor of action logits
        """
        return self.actor(observations)
        
    def training_forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with explicit observation and mask tensors for training
        
        Args:
            obs: Tensor with observation features
            mask: Boolean tensor for action masking
            
        Returns:
            Tensor of masked action logits
        """
        x = self.actor(obs)
        
        # Apply mask
        if mask is not None:
            x = torch.masked_fill(x, ~mask, self.min_val)
            
        return x 