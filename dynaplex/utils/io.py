"""
IO utility functions
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


def io_path() -> str:
    """
    Gets the path of the dynaplex IO directory.
    
    Returns:
        Path to the IO directory
    """
    # Default is ~/.dynaplex
    home_dir = os.path.expanduser("~")
    io_dir = os.path.join(home_dir, ".dynaplex")
    
    # Create if it doesn't exist
    os.makedirs(io_dir, exist_ok=True)
    
    return io_dir


def filepath(*args) -> str:
    """
    Constructs a file path from a list of subdirectories and a filename.
    
    Creates the directory if it doesn't exist.
    
    Args:
        *args: List of subdirectories and filename
    
    Returns:
        Complete file path
    """
    if not args:
        raise ValueError("No path components provided")
    
    # First check if it's an absolute path
    if os.path.isabs(args[0]):
        path = os.path.join(*args)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    # Otherwise, construct path relative to IO directory
    path = os.path.join(io_path(), *args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Save data as JSON to a file.
    
    Args:
        data: Data to save
        path: Path to save to
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Convert numpy arrays to lists
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    data = convert_numpy(data)
    
    # Save data
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.
    
    Args:
        path: Path to load from
    
    Returns:
        Loaded data
    """
    with open(path, "r") as f:
        return json.load(f)


def save_numpy(data: np.ndarray, path: str) -> None:
    """
    Save numpy array to a file.
    
    Args:
        data: Numpy array to save
        path: Path to save to
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Save data
    np.save(path, data)


def load_numpy(path: str) -> np.ndarray:
    """
    Load numpy array from a file.
    
    Args:
        path: Path to load from
    
    Returns:
        Loaded numpy array
    """
    return np.load(path) 