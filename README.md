# DynaPlex Python

A pure Python implementation of DynaPlex, designed for training generally capable agents using Deep Reinforcement Learning in inventory control based on Super-MDP.

## Overview

This project is a Python-only reimplementation of the original [DynaPlex](https://github.com/DynaPlex/DynaPlex) C++ codebase. It provides all the same functionality but is implemented entirely in Python for improved accessibility and easier extension.

The library supports:

- Zero-shot Generalization in Inventory Management
- Deep Controlled Learning for Inventory Control
- Various inventory models (lost sales, random lead times, etc.)
- Training generally capable agents using Deep RL
- Super-MDP frameworks for inventory control

## New Features: GC-LSN and Zero-shot Experiments

The library now includes:

1. **GC-LSN (Greedy Capped Lost Sales Network)** implementation:
   - Training scripts for replicating the GC-LSN model from the paper
   - Model loading utilities for using pre-trained weights
   - Visualization tools for policy analysis

2. **Zero-shot Generalization Experiments**:
   - Evaluate model performance across varied demand patterns
   - Test robustness to different leadtime distributions
   - Analyze sensitivity to cost parameter changes
   - Compare against benchmark policies like base-stock and myopic

## Getting Started

To use this library, first install it:

```bash
# Clone the repository
git clone https://github.com/your-username/DynaPlex_Python.git
cd DynaPlex_Python

# Install the package
pip install -e .
```

Then try running the example script:

```bash
python examples/zero_shot_inventory_control.py
```

## Running GC-LSN Experiments

We provide a convenient script to run all experiments:

```bash
# Run the pre-trained model evaluation and zero-shot experiments
./run_experiments.sh

# To also train a new GC-LSN model (time-consuming)
./run_experiments.sh --train
```

The script will:
1. Load and evaluate the pre-trained GC-LSN model
2. Run zero-shot generalization experiments
3. Optionally train a new model from scratch

Results will be saved as:
- `gc_lsn_comparison.png` - Performance comparison against benchmark policies
- `gc_lsn_policy_behavior.png` - Visualization of policy behavior
- `gc_lsn_state_action_map.png` - Heatmap of state-action mappings
- Various CSV files and plots in the `results/zero_shot/` directory

## Implementation Details

This Python implementation includes:

1. **Core Framework**:
   - MDP base class with standardized interface
   - Policy interface with implementations for various strategies
   - Simulator for evaluation and visualization
   - Trainer for different RL algorithms

2. **Inventory Control Models**:
   - Zero-shot Lost Sales Inventory Control (from the original paper)
   - Support for stochastic leadtimes, cyclic demand, and random yield

3. **Policy Implementations**:
   - Base stock policies for inventory control
   - Myopic (newsvendor) policies
   - Neural network policies trained with various algorithms
   - Random policies for benchmarking

4. **Neural Network Components**:
   - MLP architectures for policy learning
   - Dueling network architectures for Q-learning
   - Configurable activation functions and hyperparameters

5. **Reinforcement Learning Algorithms**:
   - Deep Controlled Learning (DCL) as described in the paper
   - Framework for implementing DQN, PPO, and other algorithms

## Creating a Custom MDP

You can create your own MDP by extending the base `MDP` class:

```python
from dynaplex.core.mdp import MDP

class MyCustomMDP(MDP):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your MDP parameters
        
    def get_initial_state(self, seed=None):
        # Create and return initial state
        
    def is_action_valid(self, state, action):
        # Check if action is valid in state
        
    def get_next_state_reward(self, state, action, seed=None):
        # Apply action to state and return next_state, reward, done
        
    def get_features(self, state):
        # Extract feature vector from state
```

## Differences from C++ Version

This Python reimplementation differs from the original C++ version in these ways:

1. **Pure Python Implementation**: No C++ dependencies or bindings required
2. **Simplified API**: More Pythonic interface with better type hints and docstrings
3. **Modern ML Framework**: Direct integration with PyTorch for neural networks
4. **Extensibility**: Easier to extend with new models, policies, and algorithms
5. **Visualization**: Built-in plotting and analysis tools

## Structure

- `dynaplex/core` - Core functionality and interfaces
- `dynaplex/models` - MDP models for various inventory problems
- `dynaplex/algorithms` - RL algorithms and solvers
- `dynaplex/policies` - Policy implementations
- `dynaplex/nn` - Neural network architectures
- `dynaplex/utils` - Utility functions and helpers
- `examples/training` - Training and evaluation scripts for GC-LSN
- `examples/zero_shot_inventory_control.py` - Basic example for the zero-shot inventory control problem

## References

This implementation is based on the papers:
- [Deep Controlled Learning for Inventory Control](https://www.sciencedirect.com/science/article/pii/S0377221725000463)
- [Zero-shot Generalization in Inventory Management: Train, then Estimate and Decide](https://arxiv.org/abs/2411.00515) 