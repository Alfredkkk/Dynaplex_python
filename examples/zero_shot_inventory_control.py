#!/usr/bin/env python3
"""
Example script for using the Zero-shot Lost Sales Inventory Control MDP
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import dynaplex
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

import dynaplex as dp


def evaluate_policies():
    """
    Evaluate different policies on the Zero-shot Lost Sales Inventory Control MDP.
    """
    # Create MDP with custom parameters
    mdp = dp.get_mdp(
        id="zero_shot_lost_sales_inventory_control",
        discount_factor=0.99,
        p=10.0,  # Penalty cost
        h=1.0,   # Holding cost
        max_leadtime=3,
        mean_demand=[5.0],
        std_demand=[2.0],
        max_order_size=10,
        max_system_inv=20,
        train_stochastic_leadtimes=True,
        leadtime_probs=[0.2, 0.5, 0.3, 0.0]
    )
    
    # Create simulator for evaluation
    simulator = dp.get_simulator(mdp, num_episodes=100, max_steps=100)
    
    # Get different policies
    random_policy = mdp.get_policy(id="random")
    base_stock_policy = mdp.get_policy(id="base_stock", target_level=15)
    myopic_policy = mdp.get_policy(id="myopic")
    
    # Evaluate policies
    print("Evaluating Random Policy...")
    random_results = simulator.evaluate(random_policy)
    print(f"Average Cost: {-random_results['average_reward']:.2f}")
    
    print("\nEvaluating Base Stock Policy...")
    base_stock_results = simulator.evaluate(base_stock_policy)
    print(f"Average Cost: {-base_stock_results['average_reward']:.2f}")
    
    print("\nEvaluating Myopic Policy...")
    myopic_results = simulator.evaluate(myopic_policy)
    print(f"Average Cost: {-myopic_results['average_reward']:.2f}")
    
    # Return results for plotting
    return {
        "Random": random_results,
        "Base Stock": base_stock_results,
        "Myopic": myopic_results
    }


def plot_results(results):
    """
    Plot the results of the policy evaluation.
    
    Args:
        results: Dictionary mapping policy names to evaluation results
    """
    # Plot average costs
    plt.figure(figsize=(10, 6))
    
    policy_names = list(results.keys())
    avg_costs = [-results[name]['average_reward'] for name in policy_names]
    std_costs = [results[name]['std_reward'] for name in policy_names]
    
    plt.bar(policy_names, avg_costs, yerr=std_costs, capsize=10, alpha=0.7)
    plt.ylabel('Average Cost per Episode')
    plt.title('Policy Comparison - Zero-shot Lost Sales Inventory Control')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, cost in enumerate(avg_costs):
        plt.text(i, cost + 1, f"{cost:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('policy_comparison.png')
    plt.show()


def train_and_evaluate_neural_network():
    """
    Train a neural network policy using DCL and evaluate it.
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
    
    # Create a trainer for DCL algorithm
    trainer = dp.get_trainer(
        mdp, 
        algorithm="dcl",
        num_episodes=1000,
        learning_rate=0.001,
        network={
            "type": "mlp",
            "hidden_sizes": [64, 64],
            "activation": "relu"
        },
        save_path="policies/zero_shot"
    )
    
    # Progress callback
    def progress_callback(info):
        if info["episode"] % 100 == 0:
            print(f"Episode {info['episode']}, "
                  f"Loss: {info['loss']:.4f if info['loss'] is not None else 'N/A'}, "
                  f"Return: {info['results']['average_discounted_return']:.2f}")
    
    # Train policy
    print("Training neural network policy...")
    nn_policy = trainer.train(callback=progress_callback)
    
    # Create simulator for evaluation
    simulator = dp.get_simulator(mdp, num_episodes=100, max_steps=100)
    
    # Evaluate neural network policy
    print("\nEvaluating Neural Network Policy...")
    nn_results = simulator.evaluate(nn_policy)
    print(f"Average Cost: {-nn_results['average_reward']:.2f}")
    
    # Get base stock policy for comparison
    base_stock_policy = mdp.get_policy(id="base_stock")
    base_stock_results = simulator.evaluate(base_stock_policy)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    policy_names = ["Neural Network", "Base Stock"]
    avg_costs = [-nn_results['average_reward'], -base_stock_results['average_reward']]
    std_costs = [nn_results['std_reward'], base_stock_results['std_reward']]
    
    plt.bar(policy_names, avg_costs, yerr=std_costs, capsize=10, alpha=0.7)
    plt.ylabel('Average Cost per Episode')
    plt.title('Neural Network vs Base Stock Policy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, cost in enumerate(avg_costs):
        plt.text(i, cost + 1, f"{cost:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('nn_comparison.png')
    plt.show()


def visualize_policy_behavior():
    """
    Visualize the behavior of a policy on a single episode.
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
    simulator = dp.get_simulator(mdp, max_steps=50)
    
    # Get base stock policy
    policy = mdp.get_policy(id="base_stock", target_level=15)
    
    # Get trace
    trace = simulator.get_trace(policy, max_steps=50)
    
    # Extract data for plotting
    periods = [entry["step"] for entry in trace]
    inventory = [entry["state"]["inventory"] for entry in trace]
    pipeline = [entry["state"]["pipeline"].sum() for entry in trace]
    actions = [entry.get("action", 0) for entry in trace]
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Inventory and pipeline
    plt.subplot(2, 1, 1)
    plt.plot(periods, inventory, 'b-', label='Inventory')
    plt.plot(periods, pipeline, 'g--', label='Pipeline Inventory')
    plt.plot(periods, [i + p for i, p in zip(inventory, pipeline)], 'r:', label='Total Inventory')
    plt.xlabel('Period')
    plt.ylabel('Units')
    plt.title('Inventory Evolution')
    plt.legend()
    plt.grid(True)
    
    # Actions
    plt.subplot(2, 1, 2)
    plt.bar(periods[:-1], actions[:-1], alpha=0.7)
    plt.xlabel('Period')
    plt.ylabel('Order Quantity')
    plt.title('Order Decisions')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('policy_behavior.png')
    plt.show()


if __name__ == "__main__":
    print("Zero-shot Lost Sales Inventory Control Example")
    print("=============================================")
    
    # Create directory for output
    os.makedirs("policies/zero_shot", exist_ok=True)
    
    # Choose which examples to run
    run_evaluation = True
    run_training = False
    run_visualization = True
    
    if run_evaluation:
        print("\nEvaluating Policies...")
        results = evaluate_policies()
        plot_results(results)
    
    if run_training:
        print("\nTraining Neural Network Policy...")
        train_and_evaluate_neural_network()
    
    if run_visualization:
        print("\nVisualizing Policy Behavior...")
        visualize_policy_behavior() 