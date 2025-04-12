"""
DynaPlex - Python-only implementation of DynaPlex for inventory control
"""

import os
import json
from pathlib import Path

from dynaplex.core.mdp import MDP
from dynaplex.core.policy import Policy
from dynaplex.core.simulator import Simulator
from dynaplex.core.trainer import Trainer
from dynaplex.utils.io import filepath, io_path

# Explictly import the nn and policies modules to make them available
import dynaplex.nn
import dynaplex.policies

__version__ = "0.1.0"

# Factory functions
def get_mdp(**kwargs):
    """
    Gets MDP based on keyword arguments.
    
    Args:
        **kwargs: Configuration parameters including at least 'id'
    
    Returns:
        MDP: An instance of the specified MDP
    """
    from dynaplex.models import create_mdp
    return create_mdp(**kwargs)

def get_simulator(mdp, **kwargs):
    """
    Gets simulator based on the MDP.
    
    Args:
        mdp: MDP instance
        **kwargs: Configuration parameters
    
    Returns:
        Simulator: An instance configured for the specified MDP
    """
    return Simulator(mdp, **kwargs)

def get_trainer(mdp, algorithm="dcl", **kwargs):
    """
    Gets a trainer instance for the specified algorithm and MDP.
    
    Args:
        mdp: MDP instance
        algorithm: String identifier for algorithm (default: "dcl")
        **kwargs: Configuration parameters
    
    Returns:
        Trainer: A configured trainer
    """
    return Trainer(mdp, algorithm=algorithm, **kwargs)

def list_mdps():
    """
    Lists available MDPs
    
    Returns:
        dict: Dictionary mapping MDP ids to descriptions
    """
    from dynaplex.models import list_available_mdps
    return list_available_mdps()

def load_policy(mdp, path):
    """
    Loads policy for mdp from path
    
    Args:
        mdp: MDP instance
        path: Path to the saved policy file
    
    Returns:
        Policy: The loaded policy
    """
    from dynaplex.policies import load_policy_from_file
    return load_policy_from_file(mdp, path)

def save_policy(policy, path, metadata=None):
    """
    Saves a policy to the specified path
    
    Args:
        policy: Policy instance to save
        path: Path to save the policy to
        metadata: Optional metadata to save with policy
    """
    policy.save(path, metadata) 