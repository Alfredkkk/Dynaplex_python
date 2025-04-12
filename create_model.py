#!/usr/bin/env python3
"""
Create a new MLP model with the same architecture as the pre-trained GC-LSN model.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(str(script_dir))

from dynaplex.nn.mlp import MLP

def create_model_from_config(config_path, output_path=None):
    """
    Create a new MLP model with the same architecture as specified in the config file.
    
    Args:
        config_path: Path to model config file (.json)
        output_path: Path to save the new model (without extension)
    
    Returns:
        New PyTorch model
    """
    # Load model config
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    print(f"Loaded model config: {model_config}")
    
    # Get model architecture from config
    hidden_layers = model_config.get("nn_architecture", {}).get("hidden_layers", [256, 128, 128, 128])
    num_inputs = model_config.get("num_inputs", 38)
    num_outputs = model_config.get("num_outputs", 130)
    
    # Create device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with the specified architecture
    model = MLP(
        input_dim=num_inputs,
        output_dim=num_outputs,
        hidden_sizes=hidden_layers,
        activation="relu"
    ).to(device)
    
    print(f"Created model with architecture: {hidden_layers}")
    print(f"Input dimension: {num_inputs}, Output dimension: {num_outputs}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Save model if output path provided
    if output_path:
        # Save state dict
        torch.save(model.state_dict(), f"{output_path}.pth")
        
        # Save config
        with open(f"{output_path}.json", 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"Saved model to {output_path}.pth")
        print(f"Saved model config to {output_path}.json")
    
    return model

if __name__ == "__main__":
    # Get config path from command line
    config_path = sys.argv[1] if len(sys.argv) > 1 else "policies/gc_lsn/GC-LSN.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "policies/gc_lsn/GC-LSN-py"
    
    # Check if output files already exist
    if os.path.exists(f"{output_path}.pth") and os.path.exists(f"{output_path}.json"):
        print(f"Model already exists at {output_path}.pth")
        print(f"To recreate the model, delete these files first.")
    else:
        # Create model
        model = create_model_from_config(config_path, output_path) 