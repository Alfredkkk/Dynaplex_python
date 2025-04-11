import json
import sys
import torch
import os
import numpy as np

from dp import dynaplex
from scripts.networks.lost_sales_dcl_mlp import ActorMLP

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_simple_policy():
    """Train a policy on fixed parameters with fewer epochs"""
    params = {
        "leadtime": 3,
        "p": 9.0,
        "h": 1.0,
        "demand_mean": 4.0
    }
    
    print(f"Training policy with parameters: {params}")
    
    # Create MDP
    mdp_config = {
        "id": "lost_sales",
        "p": params["p"],
        "h": params["h"],
        "leadtime": params["leadtime"],
        "discount_factor": 1.0,
        "demand_dist": {
            "type": "poisson",
            "mean": params["demand_mean"]
        }
    }
    
    mdp = dynaplex.get_mdp(**mdp_config)
    
    # Get base policy
    base_policy = mdp.get_policy("base_stock")
    
    # Generate samples - using smaller values
    N = 1000  # Number of trajectories - reduced
    M = 500   # Steps per trajectory - reduced
    sample_generator = dynaplex.get_sample_generator(mdp, N=N, M=M)
    
    # Generate and save samples
    sample_path = dynaplex.filepath(mdp.identifier(), "simple_samples.json")
    sample_generator.generate_samples(base_policy, sample_path)
    
    # Train policy using the generated samples
    with open(sample_path, 'r') as json_file:
        sample_data = json.load(json_file)['samples']
        
        # Prepare tensors
        tensor_y = torch.LongTensor([sample['action_label'] for sample in sample_data])
        tensor_mask = torch.BoolTensor([sample['allowed_actions'] for sample in sample_data])
        tensor_x = torch.FloatTensor([sample['features'] for sample in sample_data])
        
        # Create model
        model = ActorMLP(
            input_dim=mdp.num_flat_features(), 
            hidden_dim=64, 
            output_dim=mdp.num_valid_actions(),
            min_val=torch.finfo(torch.float).min
        )
        
        # Move to device if needed
        if device != torch.device('cpu'):
            tensor_mask = tensor_mask.to(device)
            tensor_x = tensor_x.to(device)
            tensor_y = tensor_y.to(device)
            model.to(device)
        
        # Training with fewer epochs
        train_model(model, tensor_x, tensor_y, tensor_mask, epochs=10, batch_size=32)
        
        # Save the trained policy
        policy_path = dynaplex.filepath(mdp.identifier(), "simple_policy")
        
        json_info = {
            'input_type': 'dict', 
            'num_inputs': mdp.num_flat_features(), 
            'num_outputs': mdp.num_valid_actions()
        }
        
        dynaplex.save_policy(model, json_info, policy_path, device)
        print(f"Saved model at {policy_path}")
        
        return policy_path

def train_model(model, features, targets, masks, epochs=10, batch_size=32):
    """Train the model using supervised learning (simplified)"""
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0)
    
    # Define loss function
    loss_function = torch.nn.NLLLoss()
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(features, targets, masks)
    
    # Split train and validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # Training loop
    min_val = torch.finfo(torch.float).min
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets, masks in train_loader:
            optimizer.zero_grad()
            
            observations = {'obs': inputs, 'mask': masks}
            outputs = model(observations)
            
            # Apply mask
            masked_outputs = torch.masked_fill(outputs, ~masks, min_val)
            log_outputs = log_softmax(masked_outputs)
            
            # Compute loss
            loss = loss_function(log_outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                observations = {'obs': inputs, 'mask': masks}
                outputs = model(observations)
                
                # Apply mask
                masked_outputs = torch.masked_fill(outputs, ~masks, min_val)
                log_outputs = log_softmax(masked_outputs)
                
                # Compute loss
                loss = loss_function(log_outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def test_zero_shot_generalization(trained_policy_path):
    """Test the trained policy on a few different parameter settings"""
    print("\nTesting zero-shot generalization...")
    
    # Base parameters
    base_params = {
        "leadtime": 3,
        "p": 9.0,
        "h": 1.0,
        "demand_mean": 4.0
    }
    
    # Test cases - one for each parameter change
    test_cases = [
        {"leadtime": 4, "p": 9.0, "h": 1.0, "demand_mean": 4.0, "name": "Leadtime=4"},
        {"leadtime": 3, "p": 19.0, "h": 1.0, "demand_mean": 4.0, "name": "Penalty=19"},
        {"leadtime": 3, "p": 9.0, "h": 1.0, "demand_mean": 5.0, "name": "Demand=5"}
    ]
    
    # Run evaluation for each test case
    for test_case in test_cases:
        name = test_case.pop("name")
        print(f"\nTesting on {name}")
        
        # Create test MDP
        test_mdp_config = {
            "id": "lost_sales",
            "p": test_case["p"],
            "h": test_case["h"],
            "leadtime": test_case["leadtime"],
            "discount_factor": 1.0,
            "demand_dist": {
                "type": "poisson",
                "mean": test_case["demand_mean"]
            }
        }
        
        test_mdp = dynaplex.get_mdp(**test_mdp_config)
        
        # Load trained policy
        trained_policy = dynaplex.load_policy(test_mdp, trained_policy_path)
        
        # Get benchmark policies
        base_stock_policy = test_mdp.get_policy("base_stock")
        
        # Compare policies
        policies = [trained_policy, base_stock_policy]
        policy_names = ["Trained DNN", "Base Stock"]
        
        # Set up policy comparer with fewer trajectories/periods for faster evaluation
        comparer = dynaplex.get_comparer(
            test_mdp, 
            number_of_trajectories=20,  # Reduced 
            periods_per_trajectory=200, # Reduced
            rng_seed=42
        )
        
        # Run comparison
        comparison = comparer.compare(policies)
        
        # Print results
        print(f"\nResults for {name}:")
        for i, item in enumerate(comparison):
            print(f"  {policy_names[i]}: Mean Cost = {item['mean']:.4f}, Std Dev = {item['std_dev']:.4f}")

if __name__ == "__main__":
    # Run the simplified experiment
    policy_path = train_simple_policy()
    
    # Test zero-shot generalization
    test_zero_shot_generalization(policy_path) 