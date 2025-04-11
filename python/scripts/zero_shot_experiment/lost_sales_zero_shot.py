import json
import sys
import torch
import os
import numpy as np
import pandas as pd

from dp import dynaplex
from scripts.networks.lost_sales_dcl_mlp import ActorMLP

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the parameter ranges for the experiments
# These can be adjusted based on the specific scenarios in the zero-shot paper
leadtimes = [1, 2, 3, 4, 5]
p_values = [4.0, 9.0, 19.0, 29.0, 39.0, 49.0, 99.0]
demand_means = [2.0, 3.0, 4.0, 5.0, 6.0]

# Define the training parameters
# You might need to adjust these based on the specific paper experiments
training_params = {
    "leadtime": 3,
    "p": 9.0,
    "h": 1.0,
    "demand_mean": 4.0
}

def train_policy(params):
    """Train a policy on the given parameters"""
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
    
    # Generate samples
    N = 4000  # Number of trajectories
    M = 1000  # Steps per trajectory
    sample_generator = dynaplex.get_sample_generator(mdp, N=N, M=M)
    
    # Generate and save samples
    sample_path = dynaplex.filepath(mdp.identifier(), "zero_shot_samples.json")
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
        
        # Training logic (simplified DCL approach)
        # This could be expanded based on the specific training approach in the paper
        train_model(model, tensor_x, tensor_y, tensor_mask)
        
        # Save the trained policy
        policy_path = dynaplex.filepath(
            mdp.identifier(), 
            f"zero_shot_policy_L{params['leadtime']}_p{params['p']}_m{params['demand_mean']}"
        )
        
        json_info = {
            'input_type': 'dict', 
            'num_inputs': mdp.num_flat_features(), 
            'num_outputs': mdp.num_valid_actions()
        }
        
        dynaplex.save_policy(model, json_info, policy_path, device)
        print(f"Saved model at {policy_path}")
        
        return policy_path

def train_model(model, features, targets, masks, epochs=100, batch_size=32):
    """Train the model using supervised learning"""
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0)
    
    # Define loss function
    loss_function = torch.nn.NLLLoss()
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(features, targets, masks)
    
    # Split train and validation
    train_size = int(0.95 * len(dataset))
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
    patience = 15
    patience_counter = 0
    
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
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

def evaluate_policy(trained_policy_path, test_params):
    """Evaluate a trained policy on different parameter settings"""
    # Create test MDP
    test_mdp_config = {
        "id": "lost_sales",
        "p": test_params["p"],
        "h": test_params["h"],
        "leadtime": test_params["leadtime"],
        "discount_factor": 1.0,
        "demand_dist": {
            "type": "poisson",
            "mean": test_params["demand_mean"]
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
    
    # Set up policy comparer
    comparer = dynaplex.get_comparer(
        test_mdp, 
        number_of_trajectories=100, 
        periods_per_trajectory=1000,
        rng_seed=42
    )
    
    # Run comparison
    comparison = comparer.compare(policies)
    
    # Extract results
    results = []
    for i, item in enumerate(comparison):
        result = {
            "policy": policy_names[i],
            "mean_cost": item['mean'],
            "std_dev": item['std_dev'],
            "p": test_params["p"],
            "leadtime": test_params["leadtime"],
            "demand_mean": test_params["demand_mean"]
        }
        results.append(result)
        
    return results

def run_zero_shot_experiments():
    """Run the full zero-shot experiments"""
    # Train on the base parameters
    trained_policy_path = train_policy(training_params)
    
    # Run evaluations across all parameter combinations
    all_results = []
    
    # Test across different leadtimes
    for lt in leadtimes:
        if lt == training_params["leadtime"]:
            continue  # Skip the training case
            
        test_params = training_params.copy()
        test_params["leadtime"] = lt
        results = evaluate_policy(trained_policy_path, test_params)
        all_results.extend(results)
    
    # Test across different p values
    for p in p_values:
        if p == training_params["p"]:
            continue  # Skip the training case
            
        test_params = training_params.copy()
        test_params["p"] = p
        results = evaluate_policy(trained_policy_path, test_params)
        all_results.extend(results)
    
    # Test across different demand means
    for mean in demand_means:
        if mean == training_params["demand_mean"]:
            continue  # Skip the training case
            
        test_params = training_params.copy()
        test_params["demand_mean"] = mean
        results = evaluate_policy(trained_policy_path, test_params)
        all_results.extend(results)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_path = dynaplex.filepath("lost_sales", "zero_shot_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
    
    return results_df

if __name__ == "__main__":
    # Run the experiments
    results = run_zero_shot_experiments()
    
    # Print summary
    print("\nExperiment Results Summary:")
    print(results.groupby(['policy', 'leadtime', 'p', 'demand_mean'])['mean_cost'].mean()) 