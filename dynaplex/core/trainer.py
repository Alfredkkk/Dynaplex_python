"""
Trainer for training policies on MDPs
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from dynaplex.core.policy import Policy, NeuralNetworkPolicy
from dynaplex.nn.models import create_network


class Trainer:
    """
    Trainer for training policies on MDPs.
    """
    
    def __init__(self, mdp, algorithm: str = "dcl", config: Dict[str, Any] = None):
        """
        Initialize a trainer for the given MDP.
        
        Args:
            mdp: MDP instance
            algorithm: Algorithm to use ("dcl", "ppo", "dqn", etc.)
            config: Configuration parameters
        """
        self.mdp = mdp
        self.algorithm = algorithm.lower()
        self.config = config or {}
        
        # Set default parameters
        self.num_episodes = self.config.get("num_episodes", 1000)
        self.batch_size = self.config.get("batch_size", 64)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.max_steps = self.config.get("max_steps", 1000)
        self.gamma = self.config.get("gamma", self.mdp.discount_factor)
        self.epsilon_start = self.config.get("epsilon_start", 1.0)
        self.epsilon_end = self.config.get("epsilon_end", 0.01)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.target_update = self.config.get("target_update", 10)
        self.save_path = self.config.get("save_path", "policies")
        self.save_frequency = self.config.get("save_frequency", 100)
        self.clip_ratio = self.config.get("clip_ratio", 0.2)  # For PPO
        self.network_config = self.config.get("network", {})
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize attributes to be set in train method
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.replay_buffer = None
        self.epsilon = self.epsilon_start
        self.current_episode = 0
    
    def train(self, callback: Optional[Callable] = None) -> Policy:
        """
        Train a policy on the MDP.
        
        Args:
            callback: Optional callback function to call after each episode
            
        Returns:
            Trained policy
        """
        # Initialize model and optimizer based on algorithm
        if self.algorithm == "dcl":
            return self._train_dcl(callback)
        elif self.algorithm == "dqn":
            return self._train_dqn(callback)
        elif self.algorithm == "ppo":
            return self._train_ppo(callback)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _train_dcl(self, callback: Optional[Callable] = None) -> Policy:
        """
        Train a policy using Deep Controlled Learning.
        
        Args:
            callback: Optional callback function
            
        Returns:
            Trained policy
        """
        # Initialize model
        input_size = self.mdp.num_flat_features()
        output_size = self.mdp.num_valid_actions()
        
        if input_size == 0:
            raise ValueError("MDP must provide flat features for DCL algorithm")
        
        # Create network
        self.model = create_network(
            input_size=input_size,
            output_size=output_size,
            config=self.network_config
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = []
        
        # Training loop
        start_time = time.time()
        best_return = float("-inf")
        best_model = None
        
        for episode in range(self.num_episodes):
            self.current_episode = episode
            
            # Run episode and collect samples
            samples = self._collect_samples()
            
            # Update replay buffer
            self.replay_buffer.extend(samples)
            
            # Keep buffer size limited
            max_buffer_size = self.config.get("max_buffer_size", 10000)
            if len(self.replay_buffer) > max_buffer_size:
                self.replay_buffer = self.replay_buffer[-max_buffer_size:]
            
            # Train model
            if len(self.replay_buffer) >= self.batch_size:
                loss = self._update_model()
            else:
                loss = None
            
            # Evaluate policy
            if episode % self.config.get("eval_frequency", 10) == 0:
                policy = NeuralNetworkPolicy(self.mdp, self.model, {
                    "algorithm": self.algorithm,
                    "episode": episode
                })
                
                from dynaplex.core.simulator import Simulator
                simulator = Simulator(self.mdp, {"num_episodes": 5, "max_steps": self.max_steps})
                results = simulator.evaluate(policy)
                
                avg_return = results["average_discounted_return"]
                
                if avg_return > best_return:
                    best_return = avg_return
                    best_model = self.model.state_dict().copy()
                
                # Call callback if provided
                if callback:
                    callback_info = {
                        "episode": episode,
                        "loss": loss,
                        "results": results,
                        "best_return": best_return,
                        "elapsed_time": time.time() - start_time
                    }
                    callback(callback_info)
            
            # Save policy
            if episode % self.save_frequency == 0 and episode > 0:
                policy = NeuralNetworkPolicy(self.mdp, self.model, {
                    "algorithm": self.algorithm,
                    "episode": episode
                })
                
                os.makedirs(self.save_path, exist_ok=True)
                save_file = os.path.join(self.save_path, f"{self.mdp.identifier}_{self.algorithm}_ep{episode}")
                
                policy.save(save_file, {
                    "algorithm": self.algorithm,
                    "episode": episode,
                    "mdp_identifier": self.mdp.identifier,
                    "config": self.config
                })
        
        # Create final policy using best model
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        policy = NeuralNetworkPolicy(self.mdp, self.model, {
            "algorithm": self.algorithm,
            "episode": self.num_episodes,
            "training_time": time.time() - start_time
        })
        
        # Save final policy
        os.makedirs(self.save_path, exist_ok=True)
        save_file = os.path.join(self.save_path, f"{self.mdp.identifier}_{self.algorithm}_final")
        
        policy.save(save_file, {
            "algorithm": self.algorithm,
            "episode": self.num_episodes,
            "mdp_identifier": self.mdp.identifier,
            "config": self.config,
            "best_return": best_return,
            "training_time": time.time() - start_time
        })
        
        return policy
    
    def _train_dqn(self, callback: Optional[Callable] = None) -> Policy:
        """
        Train a policy using Deep Q-Network (DQN).
        
        Args:
            callback: Optional callback function
            
        Returns:
            Trained policy
        """
        # DQN implementation would go here
        # For brevity, we'll use similar structure to DCL
        # The main differences would be:
        # - Using target network
        # - Different update rules
        # - Epsilon-greedy exploration
        
        # For now, return a DCL policy as placeholder
        return self._train_dcl(callback)
    
    def _train_ppo(self, callback: Optional[Callable] = None) -> Policy:
        """
        Train a policy using Proximal Policy Optimization (PPO).
        
        Args:
            callback: Optional callback function
            
        Returns:
            Trained policy
        """
        # PPO implementation would go here
        # For brevity, we'll use DCL as placeholder
        return self._train_dcl(callback)
    
    def _collect_samples(self) -> List[Dict[str, Any]]:
        """
        Collect samples by running an episode.
        
        Returns:
            List of state, action, reward, next_state, done tuples
        """
        # Initialize environment
        state = self.mdp.get_initial_state()
        
        # Run episode
        samples = []
        step = 0
        done = False
        
        while not done and step < self.max_steps:
            # Get features
            features = self.mdp.get_features(state)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                # Random action
                valid_actions = []
                for a in range(self.mdp.num_valid_actions()):
                    if self.mdp.is_action_valid(state, a):
                        valid_actions.append(a)
                
                if not valid_actions:
                    valid_actions = [0]  # Fallback
                
                action = np.random.choice(valid_actions)
            else:
                # Model action
                with torch.no_grad():
                    q_values = self.model(features_tensor)
                    action = q_values.argmax(dim=1).item()
                    
                    # Ensure action is valid
                    if not self.mdp.is_action_valid(state, action):
                        # Find first valid action
                        for a in range(self.mdp.num_valid_actions()):
                            if self.mdp.is_action_valid(state, a):
                                action = a
                                break
            
            # Take action
            next_state, reward, done = self.mdp.get_next_state_reward(state, action)
            
            # Store sample
            samples.append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "features": features,
                "next_features": self.mdp.get_features(next_state) if not done else None
            })
            
            # Update state
            state = next_state
            step += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return samples
    
    def _update_model(self) -> float:
        """
        Update the model using a batch of samples.
        
        Returns:
            Loss value
        """
        # Sample random batch
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([sample["features"] for sample in batch])).to(self.device)
        actions = torch.LongTensor(np.array([sample["action"] for sample in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([sample["reward"] for sample in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([
            sample["next_features"] if sample["next_features"] is not None 
            else np.zeros_like(sample["features"]) 
            for sample in batch
        ])).to(self.device)
        dones = torch.FloatTensor(np.array([sample["done"] for sample in batch])).to(self.device)
        
        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item() 