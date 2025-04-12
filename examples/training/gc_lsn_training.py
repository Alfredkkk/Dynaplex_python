#!/usr/bin/env python3
"""
Training script for GC-LSN (Greedy Capped Lost Sales Network) model
Based on 'Deep Controlled Learning for Inventory Control' and 
'Zero-shot Generalization in Inventory Management: Train, then Estimate and Decide'
"""

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta

# Add parent directory to path to import dynaplex
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = script_dir.parent.parent
sys.path.append(str(parent_dir))

import dynaplex as dp
from dynaplex.nn.mlp import MLP


class GCLSNTrainer:
    """Trainer for GC-LSN (Greedy Capped Lost Sales Network) model"""
    
    def __init__(self, save_dir="policies/gc_lsn", device=None):
        """Initialize the trainer
        
        Args:
            save_dir: Directory to save model weights and logs
            device: Torch device (cpu or cuda)
        """
        self.save_dir = save_dir
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_generations = 5  # Set number of generations as in paper
        os.makedirs(save_dir, exist_ok=True)
        self.log_file = os.path.join(save_dir, "training_logs.txt")
        self.training_stats = []
        
        # Initialize log file
        with open(self.log_file, "w") as f:
            f.write(f"GC-LSN Training Log\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Starting training at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log(self, message):
        """Log a message to both console and file"""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
    
    def create_mdp(self, **kwargs):
        """Create an MDP with default or provided parameters"""
        # Default parameters based on paper
        params = {
            "id": "zero_shot_lost_sales_inventory_control",
            "discount_factor": 0.99,
            "p": 10.0,          # Penalty cost
            "h": 1.0,           # Holding cost
            "max_leadtime": 3,
            "mean_demand": [5.0],
            "std_demand": [2.0],
            "max_order_size": 10,
            "max_system_inv": 20,
            "train_stochastic_leadtimes": True,
            "leadtime_probs": [0.2, 0.5, 0.3, 0.0]
        }
        
        # Update with any provided kwargs
        params.update(kwargs)
        
        # Create and return the MDP
        return dp.get_mdp(**params)
    
    def create_model(self, state_dim, action_dim):
        """Create a neural network model based on GC-LSN architecture
        
        Architecture from GC-LSN.json: [256, 128, 128, 128]
        """
        # Create model according to GC-LSN.json specification
        model = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=[256, 128, 128, 128],
            activation="relu"
        ).to(self.device)
        
        return model
    
    def train_generation(self, mdp, model, policy_id, gen_number, num_samples=500000, 
                         batch_size=256, num_epochs=100, learning_rate=0.001):
        """Train a single generation of the GC-LSN model
        
        Args:
            mdp: MDP model
            model: Neural network model
            policy_id: ID of the policy to use for generating samples
            gen_number: Generation number
            num_samples: Number of samples to generate
            batch_size: Batch size for training
            num_epochs: Number of epochs to train
            learning_rate: Learning rate
            
        Returns:
            Trained model and training stats
        """
        start_time = time.time()
        
        # 1. Generate samples using the current policy
        self.log(f"Generation {gen_number}: Generating {num_samples} samples using policy {policy_id}")
        
        if policy_id == "greedy_capped_base_stock":
            # For first generation, use greedy_capped_base_stock
            policy = mdp.get_policy(id=policy_id)
        else:
            # For later generations, use the neural network policy
            policy = mdp.get_policy(id=policy_id)
        
        # Create simulator for sample generation
        simulator = dp.get_simulator(mdp, config={"num_episodes": num_samples // 100, "max_steps": 100})
        samples = simulator.generate_samples(policy, num_samples=num_samples)
        
        self.log(f"Generation {gen_number}: Generated {len(samples)} samples")
        self.log(f"Sample generation time: {timedelta(seconds=int(time.time() - start_time))}")
        
        # 2. Prepare training dataset
        X = torch.tensor([sample["state_features"] for sample in samples], dtype=torch.float32).to(self.device)
        Y = torch.tensor([sample["action"] for sample in samples], dtype=torch.long).to(self.device)
        
        # Split into training and validation sets (90% train, 10% validation)
        split_idx = int(0.9 * len(samples))
        X_train, Y_train = X[:split_idx], Y[:split_idx]
        X_val, Y_val = X[split_idx:], Y[split_idx:]
        
        # 3. Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        gen_stats = {"epoch_stats": []}
        
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            
            train_loss /= len(X_train)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, Y_val).item()
                
                # Calculate accuracy
                _, predicted = torch.max(val_outputs, 1)
                val_accuracy = (predicted == Y_val).sum().item() / len(Y_val)
            
            # Evaluate policy performance using simulator
            if epoch % 5 == 0:
                # Create a new policy using the current model state
                nn_policy = dp.policies.NeuralNetworkPolicy(mdp, model)
                eval_results = simulator.evaluate(nn_policy, num_episodes=100)
                cost_improvement = eval_results.get("cost_improvement", 0)
                
                # Log progress
                self.log(f"EPOCHS: {epoch} - Training Loss: {train_loss:.5f} ({val_accuracy*100:.0f}%) - "
                         f"Validation Loss: {val_loss:.5f} ({val_accuracy*100:.0f}%) - "
                         f"Cost Imp.: {cost_improvement:.4f} "
                         f"Time: {timedelta(seconds=int(time.time() - start_time))}")
                
                # Save epoch stats
                gen_stats["epoch_stats"].append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "cost_improvement": cost_improvement,
                    "time": int(time.time() - start_time)
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        nn_policy = dp.policies.NeuralNetworkPolicy(mdp, model)
        final_results = simulator.evaluate(nn_policy, num_episodes=100)
        cost_improvement = final_results.get("cost_improvement", 0)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Update generation stats
        gen_stats.update({
            "generation": gen_number,
            "final_train_loss": train_loss,
            "final_val_loss": best_val_loss,
            "final_val_accuracy": val_accuracy,
            "final_cost_improvement": cost_improvement,
            "training_time": training_time
        })
        
        # Save model
        model_path = os.path.join(self.save_dir, f"gc_lsn_gen{gen_number}.pth")
        torch.save(model.state_dict(), model_path)
        
        # Save model config
        model_config = {
            "gen": gen_number,
            "id": "NN_Policy",
            "nn_architecture": {
                "type": "mlp",
                "hidden_layers": [256, 128, 128, 128]
            },
            "num_inputs": model.input_dim,
            "num_outputs": model.output_dim
        }
        
        config_path = os.path.join(self.save_dir, f"gc_lsn_gen{gen_number}.json")
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=1)
        
        self.log(f"\nGeneration {gen_number} complete!")
        self.log(f"Training time: {timedelta(seconds=int(training_time))}")
        self.log(f"Final validation loss: {best_val_loss:.5f}")
        self.log(f"Final cost improvement: {cost_improvement:.4f}")
        self.log(f"Model saved to {model_path}")
        
        return model, gen_stats
    
    def train(self, custom_mdp_params=None):
        """Run the complete training process across multiple generations
        
        Args:
            custom_mdp_params: Custom parameters for MDP creation
        
        Returns:
            Final model and all training stats
        """
        start_time = time.time()
        
        # Create MDP
        mdp_params = custom_mdp_params or {}
        mdp = self.create_mdp(**mdp_params)
        
        # Get state and action dimensions
        state_dim = mdp.get_state_dimension()
        action_dim = mdp.get_action_dimension()
        
        self.log(f"Created MDP with state dimension {state_dim} and action dimension {action_dim}")
        
        # Create initial model
        model = self.create_model(state_dim, action_dim)
        self.log(f"Created model with architecture: {[256, 128, 128, 128]}")
        
        # Train for multiple generations
        for gen in range(self.max_generations):
            self.log(f"\n{'='*50}")
            self.log(f"Starting Generation {gen}")
            self.log(f"{'='*50}\n")
            
            # For the first generation, use greedy_capped_base_stock policy
            # For later generations, use the neural network policy
            policy_id = "greedy_capped_base_stock" if gen == 0 else "NN_Policy"
            
            model, gen_stats = self.train_generation(mdp, model, policy_id, gen)
            self.training_stats.append(gen_stats)
            
            # Save training stats
            stats_path = os.path.join(self.save_dir, "training_stats.json")
            with open(stats_path, "w") as f:
                json.dump(self.training_stats, f, indent=2)
        
        # Training complete
        total_time = time.time() - start_time
        self.log(f"\n{'='*50}")
        self.log(f"Training complete!")
        self.log(f"Total training time: {timedelta(seconds=int(total_time))}")
        self.log(f"{'='*50}\n")
        
        return model, self.training_stats
    
    def plot_training_progress(self):
        """Plot training progress across generations"""
        if not self.training_stats:
            self.log("No training stats available for plotting")
            return
        
        # Extract data for plotting
        generations = [stats["generation"] for stats in self.training_stats]
        val_losses = [stats["final_val_loss"] for stats in self.training_stats]
        cost_improvements = [stats["final_cost_improvement"] for stats in self.training_stats]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot validation loss
        ax1.plot(generations, val_losses, 'o-', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title('Validation Loss by Generation')
        ax1.grid(True)
        
        # Plot cost improvement
        ax2.plot(generations, cost_improvements, 'o-', linewidth=2, color='green')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Cost Improvement')
        ax2.set_title('Cost Improvement by Generation')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_progress.png"))
        plt.close()
        
        self.log(f"Training progress plot saved to {os.path.join(self.save_dir, 'training_progress.png')}")


if __name__ == "__main__":
    # Create trainer
    trainer = GCLSNTrainer(save_dir="policies/gc_lsn")
    
    # Train model
    model, stats = trainer.train()
    
    # Plot training progress
    trainer.plot_training_progress() 