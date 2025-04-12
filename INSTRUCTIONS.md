# GC-LSN and Zero-shot Experiments Instructions

This document provides step-by-step instructions for reproducing the training pipeline and experiments from the papers "Deep Controlled Learning for Inventory Control" and "Zero-shot Generalization in Inventory Management".

## Prerequisites

1. Python 3.8 or higher
2. PyTorch 1.13.0 or higher
3. Required Python packages: numpy, matplotlib, pandas, scipy

## Overview of Available Scripts

We've implemented several scripts to facilitate reproduction of the papers' results:

1. **create_model.py**: Creates a Python-compatible model with the same architecture as the GC-LSN model
2. **load_gc_lsn_model.py**: Loads and evaluates the GC-LSN model
3. **zero_shot_experiments.py**: Runs zero-shot generalization experiments across different demand, leadtime, and cost configurations
4. **gc_lsn_training.py**: Implements the GC-LSN training pipeline from scratch (optional)
5. **run_experiments.sh**: A shell script that runs all experiments in sequence

## Step 1: Set Up the Environment

First, ensure you have all the necessary packages installed:

```bash
pip install torch numpy matplotlib pandas scipy
```

Then make sure you're in the project root directory:

```bash
cd /path/to/Dynaplex_python
```

## Step 2: File Preparation

Before running experiments, you need to ensure the pre-trained GC-LSN weights are accessible:

1. Verify the path to the pre-trained model weights:
   - Default path: `/root/autodl-tmp/Dynaplex_python/GC-LSN_weights/GC-LSN.pth`
   - Default config: `/root/autodl-tmp/Dynaplex_python/GC-LSN_weights/GC-LSN.json`

2. If the paths are different on your system, update the paths in the run_experiments.sh script.

## Step 3: Model Creation

Due to compatibility issues with TorchScript models, we need to create a Python-compatible model with the same architecture:

```bash
# Create a Python-compatible model with the same architecture
python create_model.py policies/gc_lsn/GC-LSN.json policies/gc_lsn/GC-LSN-py
```

This will create a new model using the parameters from the config file and save it as a Python-compatible model.

## Step 4: Running the Experiments

### Option 1: Run All Experiments at Once

The easiest way to run all experiments is to use the provided shell script:

```bash
# Make sure the script is executable
chmod +x run_experiments.sh

# Run pre-trained model evaluation and zero-shot experiments
./run_experiments.sh

# To also train a new GC-LSN model (time-consuming)
./run_experiments.sh --train
```

### Option 2: Run Individual Scripts

Alternatively, you can run each script individually:

```bash
# 1. Create a Python-compatible model
python create_model.py policies/gc_lsn/GC-LSN.json policies/gc_lsn/GC-LSN-py

# 2. Load and evaluate the GC-LSN model
python examples/training/load_gc_lsn_model.py policies/gc_lsn/GC-LSN-py.pth

# 3. Run zero-shot generalization experiments
python examples/training/zero_shot_experiments.py policies/gc_lsn/GC-LSN-py.pth

# 4. (Optional) Train a new GC-LSN model from scratch
python examples/training/gc_lsn_training.py
```

## Step 5: Examining Results

After running the experiments, you'll find the following results:

1. **Policy Comparison Plots**:
   - `gc_lsn_comparison.png`: Comparison of GC-LSN against benchmark policies
   - `gc_lsn_policy_behavior.png`: Visualization of policy behavior over time
   - `gc_lsn_state_action_map.png`: Heatmap of the state-action mapping

2. **Zero-shot Experiment Results** (in `results/zero_shot/`):
   - `demand_variation_results.csv` and `.png`: Results across different demand patterns
   - `leadtime_variation_results.csv` and `.png`: Results across different leadtime distributions
   - `cost_variation_results.csv` and `.png`: Results across different cost parameters

3. **Training Results** (if you ran training):
   - Model weights and configurations in `policies/gc_lsn/`
   - Training logs in `policies/gc_lsn/training_logs.txt`
   - Training progress plot in `policies/gc_lsn/training_progress.png`

## Understanding the Code Structure

### GC-LSN Training Pipeline

The training pipeline in `gc_lsn_training.py` follows these steps:
1. Initialize MDP with standard parameters
2. For Generation 0:
   - Use greedy_capped_base_stock policy to generate samples
   - Train neural network on these samples
3. For Generations 1-4:
   - Use the trained neural network from the previous generation to generate samples
   - Train a new neural network on these samples
4. Save the final model weights and training statistics

### Zero-shot Experiments

The zero-shot experiments in `zero_shot_experiments.py` evaluate:
1. **Demand Variation**: Testing across different demand mean and standard deviation values
2. **Leadtime Variation**: Testing across different leadtime distributions
3. **Cost Variation**: Testing across different holding and penalty cost configurations

Each experiment:
- Creates MDP instances with varied parameters
- Evaluates GC-LSN policy against base-stock and myopic policies
- Computes relative performance metrics
- Generates visualizations of the results

## Modifying the Experiments

### Changing MDP Parameters

To test with different MDP parameters, modify the `create_test_mdp()` function in `zero_shot_experiments.py` or the parameters in the experiment functions.

### Adding New Experiments

To add a new experiment type:
1. Create a new function in `zero_shot_experiments.py` following the pattern of existing ones
2. Add your new experiment to the `run_all_experiments()` function
3. Update the plotting functions to visualize your results

## Reference

For more details on the theoretical background and algorithms, please refer to the original papers:
- "Deep Controlled Learning for Inventory Control" (Temizöz et al., 2025)
- "Zero-shot Generalization in Inventory Management: Train, then Estimate and Decide" (Temizöz et al., 2024) 