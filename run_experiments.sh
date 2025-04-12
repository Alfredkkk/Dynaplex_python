#!/bin/bash

# Run experiments for DynaPlex GC-LSN training and zero-shot generalization
# Based on 'Deep Controlled Learning for Inventory Control' and 
# 'Zero-shot Generalization in Inventory Management: Train, then Estimate and Decide'

# Create required directories
mkdir -p policies/gc_lsn
mkdir -p results/zero_shot

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Copy the pre-trained model files to the policies directory
echo "====================================================="
echo "Copying pre-trained model files to policies directory..."
echo "====================================================="
cp -v /root/autodl-tmp/Dynaplex_python/GC-LSN_weights/GC-LSN.* policies/gc_lsn/

# Create a Python-compatible model with the same architecture
echo "====================================================="
echo "Creating a Python-compatible model with the same architecture..."
echo "====================================================="
python create_model.py policies/gc_lsn/GC-LSN.json policies/gc_lsn/GC-LSN-py

echo "====================================================="
echo "Starting DynaPlex GC-LSN Experiments (Pure Python Implementation)"
echo "====================================================="

# 1. Load and evaluate the pre-trained GC-LSN model
echo ""
echo "====================================================="
echo "Loading and evaluating GC-LSN model..."
echo "====================================================="
python examples/training/load_gc_lsn_model.py policies/gc_lsn/GC-LSN-py.pth

# 2. Run zero-shot generalization experiments
echo ""
echo "====================================================="
echo "Running zero-shot generalization experiments..."
echo "====================================================="
python examples/training/zero_shot_experiments.py policies/gc_lsn/GC-LSN-py.pth

# Optional: To train a new GC-LSN model (takes a long time)
if [[ "$1" == "--train" ]]; then
  echo ""
  echo "====================================================="
  echo "Training new GC-LSN model from scratch..."
  echo "====================================================="
  python examples/training/gc_lsn_training.py
fi

echo ""
echo "====================================================="
echo "All experiments completed!"
echo "====================================================="
echo "Results available in:"
echo "  - GC-LSN model comparison: gc_lsn_comparison.png"
echo "  - GC-LSN policy behavior: gc_lsn_policy_behavior.png"
echo "  - GC-LSN state-action map: gc_lsn_state_action_map.png"
echo "  - Zero-shot experiments: results/zero_shot/"
echo "=====================================================" 