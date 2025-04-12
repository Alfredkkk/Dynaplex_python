#!/usr/bin/env python3
"""
Convert TorchScript models to standard PyTorch models.

This script loads TorchScript models and saves them as standard PyTorch models.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path to import dynaplex
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(script_dir))

from dynaplex.nn.mlp import MLP


def convert_model(weights_path, output_path=None, config_path=None, device=None):
    """
    Convert TorchScript model to standard PyTorch model.
    
    Args:
        weights_path: Path to TorchScript model weights (.pth)
        output_path: Path to save converted model (without extension)
        config_path: Path to model config (.json), if None will infer from weights_path
        device: Torch device (cpu or cuda)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set output path
    if output_path is None:
        output_path = weights_path.replace('.pth', '_converted')
    
    # Load model config
    if config_path is None:
        config_path = weights_path.replace('.pth', '.json')
    
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    print(f"Loaded model config: {model_config}")
    
    # Get model architecture from config
    hidden_layers = model_config.get("nn_architecture", {}).get("hidden_layers", [256, 128, 128, 128])
    num_inputs = model_config.get("num_inputs", 38)
    num_outputs = model_config.get("num_outputs", 130)
    
    try:
        # Load TorchScript model
        scripted_model = torch.jit.load(weights_path, map_location=device)
        print("Loaded TorchScript model successfully")
        
        # Create empty PyTorch model with same architecture
        model = MLP(
            input_dim=num_inputs,
            output_dim=num_outputs,
            hidden_sizes=hidden_layers,
            activation="relu"
        ).to(device)
        
        # Generate multiple random input data samples for better matching
        num_samples = 100
        rand_inputs = torch.randn(num_samples, num_inputs, device=device)
        
        # Get outputs from TorchScript model
        with torch.no_grad():
            scripted_outputs = scripted_model(rand_inputs)
        
        print(f"TorchScript model output shape: {scripted_outputs.shape}")
        
        # Optimize PyTorch model to match TorchScript model output
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        print("Training PyTorch model to match TorchScript model...")
        
        # Train for more iterations for better matching
        num_iterations = 2000
        best_loss = float('inf')
        best_state_dict = None
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass through PyTorch model
            outputs = model(rand_inputs)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(outputs, scripted_outputs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state_dict = model.state_dict()
            
            if (i + 1) % 200 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.8f}, Best Loss: {best_loss:.8f}")
        
        # Use the best model found
        model.load_state_dict(best_state_dict)
        
        # Validate the model on a new batch
        with torch.no_grad():
            test_inputs = torch.randn(10, num_inputs, device=device)
            scripted_test_outputs = scripted_model(test_inputs)
            model_test_outputs = model(test_inputs)
            test_loss = torch.nn.functional.mse_loss(model_test_outputs, scripted_test_outputs)
            print(f"Validation Loss: {test_loss.item():.8f}")
        
        # Save PyTorch model
        torch.save(model.state_dict(), f"{output_path}.pth")
        
        # Save model config
        with open(f"{output_path}.json", 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"Saved converted model to {output_path}.pth")
        print(f"Saved model config to {output_path}.json")
        
        return model
    
    except Exception as e:
        print(f"Error converting model: {e}")
        return None


if __name__ == "__main__":
    # Get weights path from command line
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "policies/gc_lsn/GC-LSN.pth"
    output_path = sys.argv[2] if len(sys.argv) > 2 else weights_path.replace('.pth', '_converted')
    
    # Check if converted model already exists
    if os.path.exists(f"{output_path}.pth") and os.path.exists(f"{output_path}.json"):
        print(f"Converted model already exists at {output_path}.pth")
        print(f"Skipping conversion. Delete {output_path}.pth to force reconversion.")
    else:
        convert_model(weights_path, output_path) 