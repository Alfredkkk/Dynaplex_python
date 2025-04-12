"""
Policy implementations for DynaPlex.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from dynaplex.core.policy import Policy
from dynaplex.policies.neural_network_policy import NeuralNetworkPolicy
from dynaplex.policies.random_policy import RandomPolicy
from dynaplex.nn.mlp import MLP

__all__ = [
    'NeuralNetworkPolicy',
    'RandomPolicy',
    'load_policy_from_file'
]

def load_policy_from_file(mdp, path: str) -> Policy:
    """
    Load a policy from a file.
    
    Args:
        mdp: MDP instance
        path: Path to load the policy from (without extension)
    
    Returns:
        Policy: The loaded policy
    """
    # Check if paths exist
    config_path = f"{path}.json"
    weights_path = f"{path}.pth"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Policy config file not found: {config_path}")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Policy weights file not found: {weights_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model based on type
    policy_type = config.get("type", "mlp")
    
    if policy_type == "mlp":
        # Load MLP model
        model = MLP.from_json(config, device)
        model.load(weights_path, device)
        
        # Create policy
        return NeuralNetworkPolicy(mdp, model)
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}") 