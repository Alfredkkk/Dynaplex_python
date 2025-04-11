"""
Script to demonstrate how to load and use a trained GC-LSN policy.
"""

import json
import numpy as np
from dp import dynaplex

# Define the MDP configuration to match training
# Adjust these parameters to match your specific MDP
vars = {
    "id": "lost_sales",
    "p": 4.0,        # Penalty cost parameter
    "h": 1.0,        # Holding cost parameter
    "leadtime": 3,   # Leadtime for orders
    "discount_factor": 1.0,
    "demand_dist": {
        "type": "poisson",
        "mean": 4.0
    }
}

# Create the MDP
mdp = dynaplex.get_mdp(**vars)

# Print MDP info
print(f"MDP identifier: {mdp.identifier()}")
print(f"Number of features: {mdp.num_flat_features()}")
print(f"Number of valid actions: {mdp.num_valid_actions()}")

# Load GC-LSN policy
# Note: You can provide either the saved policy path from training or use the GC-LSN.pth file directly
policy_path = dynaplex.filepath("", "GC-LSN")  # Path to GC-LSN.pth and GC-LSN.json
policy = dynaplex.load_policy(mdp, policy_path)

print(f"Loaded policy: {policy.identifier()}")

# Compare with baseline policy
base_policy = mdp.get_policy("base_stock")  # Or another appropriate baseline policy
print(f"Baseline policy: {base_policy.identifier()}")

# Set up a policy comparer
comparer = dynaplex.get_comparer(mdp, number_of_trajectories=100, periods_per_trajectory=100)
comparison = comparer.compare([base_policy, policy])

# Print comparison results
print("\n策略比较结果:")
for i, result in enumerate(comparison):
    policy_name = "Baseline" if i == 0 else "GC-LSN"
    if isinstance(result, dict) and 'mean' in result:
        if 'std' in result:
            print(f"{policy_name}: Mean reward = {result['mean']:.2f}, Std = {result['std']:.2f}")
        else:
            print(f"{policy_name}: Mean reward = {result['mean']:.2f}")
    else:
        print(f"{policy_name}: Result = {result}")

# Demonstrate policy in action
print("\nDemonstrating policy in action:")
state = mdp.sample_initial_state()
print(f"Initial state features: {state.features()}")

# Get valid actions
valid_actions = state.valid_actions()
print(f"Valid actions: {valid_actions}")

# Get action from GC-LSN policy
action = policy.get_action(state)
print(f"GC-LSN policy action: {action}")

# Get action from baseline
baseline_action = base_policy.get_action(state)
print(f"Baseline policy action: {baseline_action}")

# Simulate a few steps
print("\nSimulating 5 steps:")
for step in range(5):
    # Get action from GC-LSN policy
    action = policy.get_action(state)
    
    # Apply action and get next state
    next_state, reward = mdp.step(state, action)
    
    print(f"Step {step+1}:")
    print(f"  Action: {action}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Next state features: {next_state.features()}")
    
    # Update state
    state = next_state 