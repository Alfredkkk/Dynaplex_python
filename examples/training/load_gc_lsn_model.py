#!/usr/bin/env python3
"""
Load and use the pre-trained GC-LSN model from the weights file.

This script demonstrates how to:
1. Load the pre-trained model
2. Evaluate it on the standard inventory control problem
3. Visualize state-action mappings to understand the learned policy
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dynaplex.nn.mlp import MLP
from dynaplex.utils.feature_adapter import FeatureAdapter, FeatureAdapterPolicy

# Add parent directory to path to import dynaplex
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = script_dir.parent.parent
sys.path.append(str(parent_dir))

import dynaplex as dp


def load_gc_lsn_model(weights_path, config_path=None, device=None):
    """
    Load a pre-trained GC-LSN model.
    
    Args:
        weights_path: Path to model weights
        config_path: Path to model config (if None, inferred from weights_path)
        device: Device to load model to
    
    Returns:
        PyTorch model
    """
    # Load model config
    if config_path is None:
        config_path = weights_path.replace('.pth', '.json')
    
    print("Loading GC-LSN model...")
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model config
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    print(f"Loaded model config: {model_config}")
    
    # Get model architecture from config
    hidden_layers = model_config.get("nn_architecture", {}).get("hidden_layers", [256, 128, 128, 128])
    if isinstance(hidden_layers, dict) and 'hidden_layers' in hidden_layers:
        hidden_layers = hidden_layers['hidden_layers']
    
    num_inputs = model_config.get("num_inputs", 38)
    num_outputs = model_config.get("num_outputs", 130)
    
    # Create model with the specified architecture
    model = MLP(
        input_dim=num_inputs,
        output_dim=num_outputs,
        hidden_sizes=hidden_layers,
        activation="relu"
    ).to(device)
    
    print(f"Created model with architecture: {hidden_layers}")
    print(f"Input dimension: {num_inputs}, Output dimension: {num_outputs}")
    
    # Try to load Python-compatible weights
    py_weights_path = weights_path
    try:
        # Load model weights
        model.load_state_dict(torch.load(py_weights_path, map_location=device))
        print(f"Successfully loaded weights from {py_weights_path}")
    except Exception as e:
        print(f"Error loading weights from {py_weights_path}: {e}")
        print("Using model with random weights")
    else:
        print("Python-compatible model not found, using model with random weights")
        # If you want to create and save a Python model for future use
        py_output_path = weights_path.replace('.pth', '-py')
        torch.save(model.state_dict(), f"{py_output_path}.pth")
        print(f"Saved Python-compatible model to {py_output_path}.pth for future use")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    return model


def evaluate_model(model, num_episodes=100, max_steps=100):
    """Evaluate the loaded model on the standard inventory control problem
    
    Args:
        model: Pre-trained PyTorch model
        num_episodes: Number of episodes for evaluation
        max_steps: Maximum steps per episode
        
    Returns:
        Evaluation results
    """
    # Create MDP with standard parameters
    mdp = dp.get_mdp(
        id="zero_shot_lost_sales_inventory_control",
        discount_factor=0.99,
        p=10.0,          # Penalty cost
        h=1.0,           # Holding cost
        max_leadtime=3,
        mean_demand=[5.0],
        std_demand=[2.0],
        max_order_size=10,
        max_system_inv=20,
        train_stochastic_leadtimes=True,
        leadtime_probs=[0.2, 0.5, 0.3, 0.0]
    )
    
    # Create simulator
    simulator = dp.get_simulator(mdp, config={"num_episodes": num_episodes, "max_steps": max_steps})
    
    # Create policy using the loaded model
    from dynaplex.policies.neural_network_policy import NeuralNetworkPolicy
    base_nn_policy = NeuralNetworkPolicy(mdp, model)
    
    # Wrap the policy with the feature adapter
    nn_policy = FeatureAdapterPolicy(base_nn_policy, FeatureAdapter.adapt_15d_to_38d)
    
    # Get benchmark policies
    base_stock_policy = mdp.get_policy(id="base_stock")
    myopic_policy = mdp.get_policy(id="myopic")
    
    # Evaluate policies
    print("\nEvaluating policies...")
    
    nn_results = simulator.evaluate(nn_policy)
    base_stock_results = simulator.evaluate(base_stock_policy)
    myopic_results = simulator.evaluate(myopic_policy)
    
    # Print results
    print(f"GC-LSN Neural Network Policy:")
    print(f"  Average Cost: {-nn_results['average_reward']:.2f} ± {nn_results['std_reward']:.2f}")
    print(f"  Average Discounted Return: {nn_results['average_discounted_return']:.2f}")
    
    print(f"\nBase Stock Policy:")
    print(f"  Average Cost: {-base_stock_results['average_reward']:.2f} ± {base_stock_results['std_reward']:.2f}")
    print(f"  Average Discounted Return: {base_stock_results['average_discounted_return']:.2f}")
    
    print(f"\nMyopic Policy:")
    print(f"  Average Cost: {-myopic_results['average_reward']:.2f} ± {myopic_results['std_reward']:.2f}")
    print(f"  Average Discounted Return: {myopic_results['average_discounted_return']:.2f}")
    
    # Calculate improvements
    nn_vs_bs = (-nn_results['average_reward'] / -base_stock_results['average_reward'] - 1) * 100
    nn_vs_myopic = (-nn_results['average_reward'] / -myopic_results['average_reward'] - 1) * 100
    
    print(f"\nGC-LSN Policy vs Base Stock: {nn_vs_bs:.2f}%")
    print(f"GC-LSN Policy vs Myopic: {nn_vs_myopic:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    policy_names = ["GC-LSN", "Base Stock", "Myopic"]
    avg_costs = [
        -nn_results['average_reward'],
        -base_stock_results['average_reward'],
        -myopic_results['average_reward']
    ]
    std_costs = [
        nn_results['std_reward'],
        base_stock_results['std_reward'],
        myopic_results['std_reward']
    ]
    
    x = np.arange(len(policy_names))
    width = 0.6
    
    plt.bar(x, avg_costs, width, yerr=std_costs, capsize=10, alpha=0.7)
    plt.ylabel('Average Cost per Episode')
    plt.xlabel('Policy')
    plt.title('Policy Performance Comparison')
    plt.xticks(x, policy_names)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, cost in enumerate(avg_costs):
        plt.text(i, cost + 0.5, f"{cost:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('gc_lsn_comparison.png')
    print(f"Saved comparison plot to gc_lsn_comparison.png")
    
    return {
        "nn": nn_results,
        "base_stock": base_stock_results,
        "myopic": myopic_results
    }


def visualize_policy_behavior(model):
    """Visualize the behavior of the GC-LSN policy with a trace
    
    Args:
        model: Pre-trained PyTorch model
    """
    # Create MDP
    mdp = dp.get_mdp(
        id="zero_shot_lost_sales_inventory_control",
        discount_factor=0.99,
        p=10.0,
        h=1.0,
        max_leadtime=3,
        mean_demand=[5.0],
        std_demand=[2.0],
        max_order_size=10,
        max_system_inv=20,
        train_stochastic_leadtimes=True,
        leadtime_probs=[0.2, 0.5, 0.3, 0.0]
    )
    
    # Create simulator
    simulator = dp.get_simulator(mdp, config={"max_steps": 50})
    
    # Create policy using the loaded model
    from dynaplex.policies.neural_network_policy import NeuralNetworkPolicy
    base_nn_policy = NeuralNetworkPolicy(mdp, model)
    
    # Wrap the policy with the feature adapter
    nn_policy = FeatureAdapterPolicy(base_nn_policy, FeatureAdapter.adapt_15d_to_38d)
    
    # Get base stock policy for comparison
    base_stock_policy = mdp.get_policy(id="base_stock")
    
    # Get trace for both policies (using same random seed for fair comparison)
    seed = 42
    nn_trace = simulator.get_trace(nn_policy, max_steps=50, seed=seed)
    bs_trace = simulator.get_trace(base_stock_policy, max_steps=50, seed=seed)
    
    # Extract data for plotting
    periods = [entry["step"] for entry in nn_trace]
    
    # Neural Network policy data
    nn_inventory = [entry["state"]["inventory"] for entry in nn_trace]
    nn_pipeline = [entry["state"]["pipeline"].sum() for entry in nn_trace]
    nn_actions = [entry.get("action", 0) for entry in nn_trace]
    nn_demands = [entry.get("demand", 0) for entry in nn_trace]
    nn_total_inventory = [i + p for i, p in zip(nn_inventory, nn_pipeline)]
    
    # Base Stock policy data
    bs_inventory = [entry["state"]["inventory"] for entry in bs_trace]
    bs_pipeline = [entry["state"]["pipeline"].sum() for entry in bs_trace]
    bs_actions = [entry.get("action", 0) for entry in bs_trace]
    bs_total_inventory = [i + p for i, p in zip(bs_inventory, bs_pipeline)]
    
    # Calculate costs for each period
    nn_period_costs = []
    bs_period_costs = []
    
    for i in range(len(periods)):
        # Holding cost = h * max(inventory, 0)
        # Penalty cost = p * max(-inventory, 0)
        nn_holding = 1.0 * max(nn_inventory[i], 0)
        nn_penalty = 10.0 * max(-nn_inventory[i], 0)
        nn_period_costs.append(nn_holding + nn_penalty)
        
        bs_holding = 1.0 * max(bs_inventory[i], 0)
        bs_penalty = 10.0 * max(-bs_inventory[i], 0)
        bs_period_costs.append(bs_holding + bs_penalty)
    
    # Plot
    plt.figure(figsize=(14, 12))
    
    # Inventory levels - GC-LSN
    plt.subplot(3, 2, 1)
    plt.plot(periods, nn_inventory, 'b-', label='Inventory')
    plt.plot(periods, nn_pipeline, 'g--', label='Pipeline Inventory')
    plt.plot(periods, nn_total_inventory, 'r:', label='Total Inventory')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Period')
    plt.ylabel('Units')
    plt.title('GC-LSN Policy - Inventory Evolution')
    plt.legend()
    plt.grid(True)
    
    # Inventory levels - Base Stock
    plt.subplot(3, 2, 2)
    plt.plot(periods, bs_inventory, 'b-', label='Inventory')
    plt.plot(periods, bs_pipeline, 'g--', label='Pipeline Inventory')
    plt.plot(periods, bs_total_inventory, 'r:', label='Total Inventory')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Period')
    plt.ylabel('Units')
    plt.title('Base Stock Policy - Inventory Evolution')
    plt.legend()
    plt.grid(True)
    
    # Order quantities - GC-LSN
    plt.subplot(3, 2, 3)
    plt.bar(periods[:-1], nn_actions[:-1], alpha=0.7, color='blue')
    plt.xlabel('Period')
    plt.ylabel('Order Quantity')
    plt.title('GC-LSN Policy - Order Decisions')
    plt.grid(True)
    
    # Order quantities - Base Stock
    plt.subplot(3, 2, 4)
    plt.bar(periods[:-1], bs_actions[:-1], alpha=0.7, color='green')
    plt.xlabel('Period')
    plt.ylabel('Order Quantity')
    plt.title('Base Stock Policy - Order Decisions')
    plt.grid(True)
    
    # Period costs - GC-LSN
    plt.subplot(3, 2, 5)
    plt.bar(periods, nn_period_costs, alpha=0.7, color='blue')
    plt.xlabel('Period')
    plt.ylabel('Cost')
    plt.title('GC-LSN Policy - Period Costs')
    plt.grid(True)
    
    # Period costs - Base Stock
    plt.subplot(3, 2, 6)
    plt.bar(periods, bs_period_costs, alpha=0.7, color='green')
    plt.xlabel('Period')
    plt.ylabel('Cost')
    plt.title('Base Stock Policy - Period Costs')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('gc_lsn_policy_behavior.png')
    print(f"Saved policy behavior plot to gc_lsn_policy_behavior.png")
    
    # Calculate and print summary statistics
    nn_total_cost = sum(nn_period_costs)
    bs_total_cost = sum(bs_period_costs)
    
    print(f"\nTrace Summary:")
    print(f"GC-LSN Policy - Total Cost: {nn_total_cost:.2f}")
    print(f"Base Stock Policy - Total Cost: {bs_total_cost:.2f}")
    print(f"Cost Difference: {(nn_total_cost - bs_total_cost):.2f}")
    print(f"Improvement: {((bs_total_cost - nn_total_cost) / bs_total_cost * 100):.2f}%")
    
    # Additional analysis of order patterns
    nn_avg_order = np.mean(nn_actions[:-1])
    bs_avg_order = np.mean(bs_actions[:-1])
    
    nn_stockouts = sum(1 for inv in nn_inventory if inv < 0)
    bs_stockouts = sum(1 for inv in bs_inventory if inv < 0)
    
    print(f"\nOrder Pattern Analysis:")
    print(f"GC-LSN - Average Order Size: {nn_avg_order:.2f}")
    print(f"Base Stock - Average Order Size: {bs_avg_order:.2f}")
    
    print(f"\nStockout Analysis:")
    print(f"GC-LSN - Number of Stockouts: {nn_stockouts}")
    print(f"Base Stock - Number of Stockouts: {bs_stockouts}")


def visualize_state_action_mapping(model, device=None):
    """Visualize the state-action mapping learned by the GC-LSN model
    
    Args:
        model: Pre-trained PyTorch model
        device: Torch device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate various inventory and pipeline inventory levels to visualize policy decisions
    inventory_levels = np.arange(-5, 16)  # Inventory from -5 to 15
    pipeline_levels = np.arange(0, 16)    # Pipeline from 0 to 15
    
    # Grid for heatmap
    order_decisions = np.zeros((len(inventory_levels), len(pipeline_levels)))
    
    # Create a standard set of features for the model
    # We'll vary only inventory and pipeline[0] levels
    standard_state = {
        "inventory": 0,
        "pipeline": np.array([0, 0, 0]),
        "demand": 5.0,
        "leadtime_probs": np.array([0.2, 0.5, 0.3, 0.0]),
        "period": 0
    }
    
    # Function to convert state to features
    def state_to_features(state):
        """
        Convert state to feature vector expected by the model.
        This implementation is based on the GC-LSN model from the paper.
        
        Args:
            state: State dictionary containing inventory, pipeline, etc.
            
        Returns:
            Feature tensor of shape (1, num_features)
        """
        # Initialize empty feature list
        features = []
        
        # 1. Inventory level
        features.append(float(state["inventory"]))
        
        # 2. Pipeline inventory (ordered but not yet received)
        # For each position in the pipeline, add the quantity
        pipeline = state["pipeline"]
        for i in range(3):  # GC-LSN model has max_leadtime=3
            pipe_value = pipeline[i] if i < len(pipeline) else 0.0
            features.append(float(pipe_value))
        
        # 3. Period information (in case of cyclic demand)
        features.append(float(state.get("period", 0)))
        
        # 4. Demand information
        features.append(float(state.get("demand", 5.0)))
        
        # 5. Leadtime probabilities
        leadtime_probs = state.get("leadtime_probs", [0.2, 0.5, 0.3, 0.0])
        features.extend([float(p) for p in leadtime_probs])
        
        # 6. Add padding to ensure we have the correct input dimension
        # The GC-LSN model expects 38 features as per the config
        while len(features) < 38:
            features.append(0.0)
        
        # Ensure we have exactly the right number of features
        features = features[:38]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # For each combination of inventory and pipeline level
    for i, inv in enumerate(inventory_levels):
        for j, pipe in enumerate(pipeline_levels):
            # Create state with these levels
            state = standard_state.copy()
            state["inventory"] = inv
            state["pipeline"] = np.array([pipe, 0, 0])  # Only vary the first pipeline level
            
            # Convert to features
            features = state_to_features(state)
            
            # Get model prediction
            with torch.no_grad():
                output = model(features)
                
                # Get the action with highest score
                action = torch.argmax(output, dim=1).item()
            
            order_decisions[i, j] = action
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    
    plt.subplot(1, 1, 1)
    im = plt.imshow(order_decisions, cmap='viridis', aspect='auto', 
                    extent=[min(pipeline_levels), max(pipeline_levels), 
                            min(inventory_levels), max(inventory_levels)])
    
    plt.colorbar(im, label='Order Quantity')
    plt.xlabel('Pipeline Inventory')
    plt.ylabel('On-hand Inventory')
    plt.title('GC-LSN Policy State-Action Mapping')
    
    # Add grid
    plt.grid(which='both', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add contour lines
    contour_levels = np.arange(0, 11, 2)
    CS = plt.contour(pipeline_levels, inventory_levels, order_decisions, 
                     levels=contour_levels, colors='white', alpha=0.7)
    plt.clabel(CS, inline=True, fontsize=8)
    
    # Add a line for inventory + pipeline = target_level
    target_level = 15  # Approximate base stock target level
    x = np.array([0, 15])
    y = np.array([target_level, target_level - 15])
    plt.plot(x, y, 'r--', label=f'Total Inventory = {target_level}')
    
    # Add a line for inventory = 0 (stockout threshold)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.5, label='Stockout Threshold')
    
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('gc_lsn_state_action_map.png')
    print(f"Saved state-action mapping to gc_lsn_state_action_map.png")


if __name__ == "__main__":
    print("Loading GC-LSN model...")
    
    # Get model path from command line
    model_path = sys.argv[1] if len(sys.argv) > 1 else "policies/gc_lsn/GC-LSN.pth"
    
    # Load model
    model = load_gc_lsn_model(model_path)
    
    # Evaluate model performance
    results = evaluate_model(model)
    
    # Visualize policy behavior
    visualize_policy_behavior(model)
    
    # Visualize state-action mapping
    visualize_state_action_mapping(model) 