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

def train_policy_on_base_parameters():
    """Train a DNN policy on base parameter settings"""
    # Base training parameters
    base_params = {
        "leadtime": 3,
        "p": 9.0,
        "h": 1.0,
        "demand_mean": 4.0
    }
    
    print(f"\n=== Training policy on base parameters: {base_params} ===\n")
    
    # Create MDP for training
    mdp_config = {
        "id": "lost_sales",
        "p": base_params["p"],
        "h": base_params["h"],
        "leadtime": base_params["leadtime"],
        "discount_factor": 1.0,
        "demand_dist": {
            "type": "poisson",
            "mean": base_params["demand_mean"]
        }
    }
    
    mdp = dynaplex.get_mdp(**mdp_config)
    
    # Get base policy to generate training data
    base_policy = mdp.get_policy("base_stock")
    
    # Generate samples
    N = 2000  # Number of trajectories - can be increased for better training
    M = 500   # Steps per trajectory
    sample_generator = dynaplex.get_sample_generator(mdp, N=N, M=M)
    
    # Generate and save samples
    sample_path = dynaplex.filepath(mdp.identifier(), "zeroshot_samples.json")
    print("Generating training samples...")
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
        
        # Training
        print("\nTraining DNN policy...")
        train_model(model, tensor_x, tensor_y, tensor_mask, epochs=20)
        
        # Save the trained policy
        policy_path = dynaplex.filepath(mdp.identifier(), "zeroshot_trained_policy")
        
        json_info = {
            'input_type': 'dict', 
            'num_inputs': mdp.num_flat_features(), 
            'num_outputs': mdp.num_valid_actions()
        }
        
        dynaplex.save_policy(model, json_info, policy_path, device)
        print(f"Saved trained model at {policy_path}")
        
        return policy_path, base_params

def train_model(model, features, targets, masks, epochs=20, batch_size=32):
    """Train the model using supervised learning"""
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0)
    
    # Define loss function
    loss_function = torch.nn.NLLLoss()
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(features, targets, masks)
    
    # Split train and validation
    train_size = int(0.9 * len(dataset))
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
    patience = 5
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

def create_custom_policy_for_test(trained_policy_path, test_params):
    """Special function to adjust policy for zero-shot evaluation with different parameter settings"""
    print(f"\n=== Creating test MDP with parameters: {test_params} ===\n")
    
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
    
    # Get the base stock policy for reference
    base_stock_policy = test_mdp.get_policy("base_stock")
    
    try:
        # Try to directly use the trained policy for the test MDP
        trained_policy = dynaplex.load_policy(test_mdp, trained_policy_path)
        return test_mdp, [trained_policy, base_stock_policy], ["Trained DNN", "Base Stock"]
    except RuntimeError as e:
        print(f"Error loading trained policy: {e}")
        # If it fails, we can only test the base stock policy
        print("Falling back to testing only the Base Stock policy")
        return test_mdp, [base_stock_policy], ["Base Stock"]

def evaluate_policies(mdp, policies, policy_names):
    """Evaluate policies on the given MDP"""
    print("\n=== Evaluating policies ===\n")
    
    # Set up policy comparer
    comparer = dynaplex.get_comparer(
        mdp, 
        number_of_trajectories=30,
        periods_per_trajectory=300,
        rng_seed=42
    )
    
    # Run comparison
    comparison = comparer.compare(policies)
    
    # Print results
    print("\nResults:")
    for i, item in enumerate(comparison):
        mean_cost = item['mean']
        error = item.get('error', 0.0)
        print(f"  {policy_names[i]}: Mean Cost = {mean_cost:.2f}, Error = {error:.2f}")
    
    return comparison

def run_zeroshot_experiment():
    """Run a zero-shot experiment by training on one set of parameters and testing on another"""
    # 1. Train policy on base parameters
    policy_path, base_params = train_policy_on_base_parameters()
    
    # 2. Define test scenarios
    test_scenarios = [
        # Vary lead time
        {"leadtime": 1, "p": base_params["p"], "h": base_params["h"], "demand_mean": base_params["demand_mean"], "name": "Leadtime=1"},
        {"leadtime": 5, "p": base_params["p"], "h": base_params["h"], "demand_mean": base_params["demand_mean"], "name": "Leadtime=5"},
        
        # Vary penalty cost
        {"leadtime": base_params["leadtime"], "p": 4.0, "h": base_params["h"], "demand_mean": base_params["demand_mean"], "name": "Penalty=4"},
        {"leadtime": base_params["leadtime"], "p": 19.0, "h": base_params["h"], "demand_mean": base_params["demand_mean"], "name": "Penalty=19"},
        
        # Vary demand mean
        {"leadtime": base_params["leadtime"], "p": base_params["p"], "h": base_params["h"], "demand_mean": 2.0, "name": "Demand=2"},
        {"leadtime": base_params["leadtime"], "p": base_params["p"], "h": base_params["h"], "demand_mean": 6.0, "name": "Demand=6"},
    ]
    
    # 3. Test on each scenario
    results = []
    for scenario in test_scenarios:
        name = scenario.pop("name")
        print(f"\n\n===== Testing on {name} =====")
        
        # Create test MDP and policies
        test_mdp, policies, policy_names = create_custom_policy_for_test(policy_path, scenario)
        
        # Evaluate policies
        comparison = evaluate_policies(test_mdp, policies, policy_names)
        
        # Store results
        for i, item in enumerate(comparison):
            results.append({
                "scenario": name,
                "policy": policy_names[i],
                "mean_cost": item['mean'],
                "error": item.get('error', 0.0)
            })
    
    # 4. Print summary
    print("\n\n===== EXPERIMENT SUMMARY =====")
    
    # Group results by scenario
    scenarios = sorted(set(r["scenario"] for r in results))
    policies = sorted(set(r["policy"] for r in results))
    
    print(f"\n{'Scenario':<15} | " + " | ".join(f"{policy:<15}" for policy in policies))
    print("-" * (15 + 3 + sum(16 for _ in policies)))
    
    for scenario in scenarios:
        scenario_results = {r["policy"]: r["mean_cost"] for r in results if r["scenario"] == scenario}
        print(f"{scenario:<15} | " + " | ".join(f"{scenario_results.get(policy, 'N/A'):<15.2f}" for policy in policies))

if __name__ == "__main__":
    run_zeroshot_experiment() 