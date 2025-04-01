"""
MDP models for DynaPlex
"""

from typing import Dict, Any

# Register available models here
_AVAILABLE_MODELS = {
    "zero_shot_lost_sales_inventory_control": {
        "description": "Zero-shot generalization in lost sales inventory control",
        "module": "dynaplex.models.zero_shot_lost_sales_inventory_control.mdp",
        "class": "ZeroShotLostSalesInventoryControlMDP"
    },
    "lost_sales": {
        "description": "Lost sales inventory control",
        "module": "dynaplex.models.lost_sales.mdp",
        "class": "LostSalesMDP"
    }
}


def list_available_mdps() -> Dict[str, str]:
    """
    Lists available MDPs.
    
    Returns:
        Dictionary mapping MDP IDs to descriptions
    """
    return {k: v["description"] for k, v in _AVAILABLE_MODELS.items()}


def create_mdp(id: str = None, **kwargs) -> "MDP":
    """
    Create an MDP instance.
    
    Args:
        id: Identifier for the MDP to create
        **kwargs: Configuration parameters
    
    Returns:
        MDP instance
    """
    if id is None:
        raise ValueError("MDP id must be provided")
    
    id = id.lower()
    if id not in _AVAILABLE_MODELS:
        raise ValueError(f"Unknown MDP id: {id}")
    
    # Get model info
    model_info = _AVAILABLE_MODELS[id]
    
    # Import module
    module_name = model_info["module"]
    class_name = model_info["class"]
    
    try:
        module = __import__(module_name, fromlist=[class_name])
        mdp_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import MDP class {class_name} from {module_name}: {e}")
    
    # Create instance
    config = {"id": id, **kwargs}
    return mdp_class(config) 