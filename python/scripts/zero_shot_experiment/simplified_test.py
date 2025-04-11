import json
import sys
import os
import numpy as np

from dp import dynaplex

def test_base_stock_policy_across_parameters():
    """Test the base stock policy on different parameter settings"""
    print("\nTesting inventory control policies across different parameters...")
    
    # Base parameters
    base_params = {
        "leadtime": 3,
        "p": 9.0,
        "h": 1.0,
        "demand_mean": 4.0
    }
    
    # Test cases for each parameter
    test_cases = []
    
    # Leadtime variations
    for lt in [1, 2, 3, 4, 5]:
        params = base_params.copy()
        params["leadtime"] = lt
        test_cases.append({"params": params, "name": f"Leadtime={lt}"})
    
    # Penalty cost variations
    for p in [4.0, 9.0, 19.0, 39.0, 99.0]:
        params = base_params.copy()
        params["p"] = p
        test_cases.append({"params": params, "name": f"Penalty={p}"})
    
    # Demand mean variations
    for dm in [2.0, 3.0, 4.0, 5.0, 6.0]:
        params = base_params.copy()
        params["demand_mean"] = dm
        test_cases.append({"params": params, "name": f"Demand={dm}"})
    
    # Results storage
    results = []
    
    # Run evaluation for each test case
    for test_case in test_cases:
        name = test_case["name"]
        params = test_case["params"]
        print(f"\nTesting on {name}")
        
        # Create MDP with these parameters
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
        
        # Get policies to test
        base_stock_policy = mdp.get_policy("base_stock")
        random_policy = mdp.get_policy("random")
        
        # Compare policies
        policies = [base_stock_policy, random_policy]
        policy_names = ["Base Stock", "Random"]
        
        # Set up policy comparer with smaller number of trajectories for faster evaluation
        comparer = dynaplex.get_comparer(
            mdp, 
            number_of_trajectories=10,  # Reduced
            periods_per_trajectory=100, # Reduced
            rng_seed=42
        )
        
        # Run comparison
        comparison = comparer.compare(policies)
        
        # First, print the complete data structure to see its format
        if test_case == test_cases[0]:  # Only for first case
            print("\nSample comparison result structure:")
            print(comparison[0])  # Look at the first policy result
        
        # Print results - check for mean and standard deviation keys
        print(f"\nResults for {name}:")
        for i, item in enumerate(comparison):
            # Extract mean value - check if it's directly available or nested
            if 'mean' in item:
                mean_cost = item['mean']
                std_dev = item.get('stddev', 0.0)  # Try to get stddev if available
            else:
                # For debugging - will help identify the actual structure
                print(f"  Keys in comparison item: {list(item.keys())}")
                mean_cost = 0.0  # Default
                std_dev = 0.0   # Default
            
            print(f"  {policy_names[i]}: Mean Cost = {mean_cost:.2f}, Std Dev = {std_dev:.2f}")
            
            # Store results
            results.append({
                "policy": policy_names[i],
                "parameter": name.split("=")[0],
                "value": float(name.split("=")[1]),
                "mean_cost": mean_cost,
                "std_dev": std_dev
            })
    
    # Summarize results
    print("\n=== Summary of Results ===")
    
    # Group by parameter type and policy
    parameter_types = ["Leadtime", "Penalty", "Demand"]
    policies = ["Base Stock", "Random"]
    
    for param_type in parameter_types:
        print(f"\n{param_type} Parameter Results:")
        for policy in policies:
            print(f"\n  {policy} Policy:")
            # Filter results for this parameter and policy
            param_results = [r for r in results if r["parameter"] == param_type and r["policy"] == policy]
            # Sort by parameter value
            param_results.sort(key=lambda x: x["value"])
            # Print in tabular format
            print(f"    {'Value':<8} | {'Mean Cost':<10} | {'Std Dev':<10}")
            print(f"    {'-'*8} | {'-'*10} | {'-'*10}")
            for r in param_results:
                print(f"    {r['value']:<8.1f} | {r['mean_cost']:<10.2f} | {r['std_dev']:<10.2f}")

if __name__ == "__main__":
    # Run the simplified test
    test_base_stock_policy_across_parameters() 