"""
Simulator for evaluating policies on MDPs
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import time
import os
import json


class Simulator:
    """
    Simulator for evaluating policies on MDPs.
    """
    
    def __init__(self, mdp, config: Dict[str, Any] = None):
        """
        Initialize a simulator for the given MDP.
        
        Args:
            mdp: MDP instance
            config: Configuration parameters
        """
        self.mdp = mdp
        self.config = config or {}
        self.max_steps = self.config.get("max_steps", 1000)
        self.num_episodes = self.config.get("num_episodes", 100)
        self.rng = np.random.RandomState(self.config.get("seed", None))
    
    def run_episode(self, policy, seed=None):
        """
        Run a single episode using the given policy.
        
        Args:
            policy: Policy to use for action selection
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with episode results
        """
        # Initialize state
        state = self.mdp.get_initial_state(seed)
        
        total_reward = 0.0
        discounted_return = 0.0
        discount_factor = self.mdp.discount_factor
        discount_cumulative = 1.0
        
        state_history = [state]
        action_history = []
        reward_history = []
        
        # Run episode
        for step in range(self.config.get("max_steps", 100)):
            # Get action using policy
            action = policy.get_action(state)
            
            # Apply action to environment
            next_state, reward, done = self.mdp.get_next_state_reward(state, action)
            
            # Update histories
            action_history.append(action)
            reward_history.append(reward)
            
            # Update running totals
            total_reward += reward
            discounted_return += discount_cumulative * reward
            discount_cumulative *= discount_factor
            
            # Update state
            state = next_state
            state_history.append(state)
            
            # Check if episode is done
            if done:
                break
        
        # Return results
        return {
            "total_reward": total_reward,
            "discounted_return": discounted_return,
            "num_steps": len(action_history),
            "state_history": state_history,
            "action_history": action_history,
            "reward_history": reward_history
        }
    
    def evaluate(self, policy, num_episodes=None, render=False):
        """
        Evaluate a policy over multiple episodes.
        
        Args:
            policy: Policy to evaluate
            num_episodes: Number of episodes to run (overrides config)
            render: Whether to render the environment
            
        Returns:
            Dictionary with evaluation results
        """
        if num_episodes is None:
            num_episodes = self.config.get("num_episodes", 10)
        
        rewards = []
        discounted_returns = []
        steps = []
        
        # Run episodes
        for i in range(num_episodes):
            # Generate episode seed
            episode_seed = self.rng.randint(0, 2**32 - 1)
            
            # Run episode
            result = self.run_episode(policy, episode_seed)
            
            # Update statistics
            rewards.append(result["total_reward"])
            discounted_returns.append(result["discounted_return"])
            steps.append(result["num_steps"])
        
        # Compute statistics
        average_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        average_discounted_return = np.mean(discounted_returns)
        average_steps = np.mean(steps)
        
        # Get policy type
        policy_type = "unknown"
        if hasattr(policy, "type_identifier"):
            policy_type = policy.type_identifier
        elif hasattr(policy, "policy") and hasattr(policy.policy, "type_identifier"):
            # For wrapped policies like FeatureAdapterPolicy
            policy_type = f"adapter_{policy.policy.type_identifier}"
        
        # Return results
        return {
            "average_reward": average_reward,
            "std_reward": std_reward,
            "average_discounted_return": average_discounted_return,
            "average_steps": average_steps,
            "policy_type": policy_type
        }
    
    def get_trace(self, policy, max_steps: int = 100, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get a trace of states, actions, and rewards for visualization.
        
        Args:
            policy: Policy to use
            max_steps: Maximum number of steps to trace
            seed: Optional random seed
            
        Returns:
            List of dictionaries with state, action, and reward information
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Get initial state
        state = self.mdp.get_initial_state(seed)
        
        # Run episode
        trace = []
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Get state features if available
            features = None
            if self.mdp.provides_flat_features():
                features = self.mdp.get_features(state).tolist()
            
            # Add state to trace
            trace_entry = {
                "state": state,
                "features": features,
                "step": step
            }
            
            # Choose action
            action = policy.get_action(state)
            trace_entry["action"] = action
            
            # Take action
            next_state, reward, done = self.mdp.get_next_state_reward(state, action)
            trace_entry["reward"] = reward
            trace_entry["done"] = done
            
            # Add to trace
            trace.append(trace_entry)
            
            # Update state
            state = next_state
            step += 1
            
            # Add final state if done
            if done:
                features = None
                if self.mdp.provides_flat_features():
                    features = self.mdp.get_features(state).tolist()
                    
                trace.append({
                    "state": state,
                    "features": features,
                    "step": step,
                    "done": done
                })
        
        return trace
    
    def _compute_discounted_return(self, rewards: List[float]) -> float:
        """
        Compute the discounted return from a list of rewards.
        
        Args:
            rewards: List of rewards
            
        Returns:
            Discounted return
        """
        discount_factor = self.mdp.discount_factor
        discounted_return = 0.0
        
        for t, reward in enumerate(rewards):
            discounted_return += reward * (discount_factor ** t)
            
        return discounted_return
    
    def save_results(self, results: Dict[str, Any], path: str) -> None:
        """
        Save evaluation results to a file.
        
        Args:
            results: Evaluation results
            path: Path to save results to
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save results
        with open(path, "w") as f:
            json.dump(results, f, indent=2) 