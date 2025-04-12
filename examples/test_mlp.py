#!/usr/bin/env python3
"""
Test script for the MLP implementation.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import dynaplex
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

import torch
from dynaplex.nn.mlp import MLP

def test_mlp():
    """Test MLP implementation."""
    
    print("Testing MLP implementation...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a small MLP
    input_dim = 5
    output_dim = 3
    hidden_sizes = [10, 8]
    
    model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        activation="relu"
    ).to(device)
    
    print(f"Created MLP with architecture: {hidden_sizes}")
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    
    # Create a random input
    x = torch.randn(2, input_dim).to(device)  # Batch size of 2
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Test get_action
    action_probs, logits = model.get_action(x)
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Test to_json and from_json
    config = model.to_json()
    print(f"Model config: {config}")
    
    new_model = MLP.from_json(config, device)
    print(f"Recreated model from config")
    
    # Verify the models produce the same output
    with torch.no_grad():
        output1 = model(x)
        output2 = new_model(x)
        is_same = torch.allclose(output1, output2)
        print(f"Original and recreated models produce the same output: {is_same}")
    
    # Test saving and loading
    try:
        temp_path = "/tmp/test_mlp.pth"
        model.save(temp_path)
        print(f"Model saved to {temp_path}")
        
        loaded_model = MLP(input_dim, output_dim, hidden_sizes, "relu").to(device)
        loaded_model.load(temp_path)
        print(f"Model loaded from {temp_path}")
        
        # Check if loaded model produces the same output
        with torch.no_grad():
            output3 = loaded_model(x)
            is_same = torch.allclose(output1, output3)
            print(f"Original and loaded models produce the same output: {is_same}")
            
        # Clean up
        os.remove(temp_path)
    except Exception as e:
        print(f"Error during save/load test: {e}")
    
    print("MLP test completed successfully!")

if __name__ == "__main__":
    test_mlp() 