import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from dp import dynaplex
from scripts.zero_shot_experiment.lost_sales_zero_shot import train_policy, evaluate_policy

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Run zero-shot inventory control experiments')
    parser.add_argument('--experiment', type=str, choices=['leadtime', 'penalty', 'demand_mean'], 
                        default='leadtime', help='Which parameter to vary in the experiment')
    parser.add_argument('--train_leadtime', type=int, default=3, help='Leadtime for training')
    parser.add_argument('--train_penalty', type=float, default=9.0, help='Penalty cost (p) for training')
    parser.add_argument('--train_demand_mean', type=float, default=4.0, help='Demand mean for training')
    parser.add_argument('--holding_cost', type=float, default=1.0, help='Holding cost (h)')
    parser.add_argument('--trajectories', type=int, default=100, help='Number of trajectories for evaluation')
    parser.add_argument('--periods', type=int, default=1000, help='Periods per trajectory for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def run_experiment(args):
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Define training parameters
    training_params = {
        "leadtime": args.train_leadtime,
        "p": args.train_penalty,
        "h": args.holding_cost,
        "demand_mean": args.train_demand_mean
    }
    
    # Train a policy with the base parameters
    print(f"\n=== Training policy with parameters: leadtime={training_params['leadtime']}, "
          f"p={training_params['p']}, demand_mean={training_params['demand_mean']} ===\n")
    
    trained_policy_path = train_policy(training_params)
    
    # Define parameter ranges based on the experiment type
    if args.experiment == 'leadtime':
        param_values = [1, 2, 3, 4, 5]
        param_name = 'leadtime'
        x_label = 'Lead Time'
    elif args.experiment == 'penalty':
        param_values = [4.0, 9.0, 19.0, 29.0, 39.0, 49.0, 99.0]
        param_name = 'p'
        x_label = 'Penalty Cost (p)'
    else:  # demand_mean
        param_values = [2.0, 3.0, 4.0, 5.0, 6.0]
        param_name = 'demand_mean'
        x_label = 'Demand Mean'
    
    # Run evaluations across parameter values
    all_results = []
    
    for param_value in param_values:
        # Create test parameters (copy training params and update the varied parameter)
        test_params = training_params.copy()
        test_params[param_name] = param_value
        
        print(f"\n=== Evaluating on {param_name}={param_value} ===\n")
        
        # Evaluate policy
        results = evaluate_policy(trained_policy_path, test_params)
        
        # Add parameter value to results
        for result in results:
            result['varied_param'] = param_name
            result['param_value'] = param_value
            
        all_results.extend(results)
    
    # Create DataFrame with results
    results_df = pd.DataFrame(all_results)
    
    # Save results
    Path(os.path.dirname(dynaplex.filepath("results", ""))).mkdir(parents=True, exist_ok=True)
    results_path = dynaplex.filepath("results", f"zero_shot_{args.experiment}_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")
    
    # Plot results
    plot_results(results_df, param_name, x_label, args.experiment)
    
    return results_df

def plot_results(results_df, param_name, x_label, experiment_name):
    """Plot the results of the experiment"""
    plt.figure(figsize=(10, 6))
    
    # Get unique policy names
    policies = results_df['policy'].unique()
    colors = ['b', 'r', 'g', 'c', 'm']
    
    for i, policy in enumerate(policies):
        policy_data = results_df[results_df['policy'] == policy]
        
        # Group by parameter value and calculate mean
        grouped = policy_data.groupby('param_value')['mean_cost'].mean()
        
        # Plot
        plt.plot(grouped.index, grouped.values, marker='o', label=policy, color=colors[i % len(colors)])
    
    plt.xlabel(x_label)
    plt.ylabel('Average Cost')
    plt.title(f'Zero-Shot Generalization: Varying {x_label}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    plot_path = dynaplex.filepath("results", f"zero_shot_{experiment_name}_plot.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    
    # Highlight the training parameter
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    results = run_experiment(args)
    
    # Print summary
    print("\nExperiment Results Summary:")
    print(results.groupby(['policy', 'param_value'])['mean_cost'].mean()) 