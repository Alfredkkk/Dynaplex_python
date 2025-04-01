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
    
    def run_episode(self, policy, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a single episode using the given policy.
        
        Args:
            policy: Policy to evaluate
            seed: Optional random seed
            
        Returns:
            Dictionary containing episode results
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Get initial state
        state = self.mdp.get_initial_state(seed)
        
        # Run episode
        total_reward = 0.0
        step = 0
        done = False
        states = [state]
        actions = []
        rewards = []
        
        while not done and step < self.max_steps:
            # Choose action
            action = policy.get_action(state)
            actions.append(action)
            
            # Take action
            next_state, reward, done = self.mdp.get_next_state_reward(state, action)
            states.append(next_state)
            rewards.append(reward)
            
            # Update total reward
            total_reward += reward
            
            # Update state
            state = next_state
            step += 1
        
        # Return results
        return {
            "total_reward": total_reward,
            "steps": step,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "discounted_return": self._compute_discounted_return(rewards)
        }
    
    def evaluate(self, policy, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a policy on multiple episodes.
        
        Args:
            policy: Policy to evaluate
            seed: Optional random seed
            
        Returns:
            Dictionary containing evaluation results
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            episode_seeds = [np.random.randint(0, 2**32) for _ in range(self.num_episodes)]
        else:
            episode_seeds = [None] * self.num_episodes
        
        # Run episodes
        start_time = time.time()
        episode_results = []
        
        for i, episode_seed in enumerate(episode_seeds):
            result = self.run_episode(policy, episode_seed)
            episode_results.append(result)
        
        end_time = time.time()
        
        # Compute statistics
        total_rewards = [result["total_reward"] for result in episode_results]
        discounted_returns = [result["discounted_return"] for result in episode_results]
        steps = [result["steps"] for result in episode_results]
        
        # Return overall results
        return {
            "num_episodes": self.num_episodes,
            "average_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "min_reward": np.min(total_rewards),
            "max_reward": np.max(total_rewards),
            "average_discounted_return": np.mean(discounted_returns),
            "average_steps": np.mean(steps),
            "total_steps": np.sum(steps),
            "execution_time": end_time - start_time,
            "mdp_identifier": self.mdp.identifier,
            "policy_type": policy.type_identifier
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