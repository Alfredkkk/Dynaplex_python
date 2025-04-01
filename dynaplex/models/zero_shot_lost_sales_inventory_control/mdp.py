"""
Zero-shot Lost Sales Inventory Control MDP
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import deque

from dynaplex.core.mdp import MDP


class DiscreteDist:
    """
    Discrete distribution for sampling.
    """
    
    def __init__(self, probs: List[float], min_val: int = 0):
        """
        Initialize discrete distribution.
        
        Args:
            probs: List of probabilities
            min_val: Minimum value in support
        """
        self.probs = np.array(probs)
        self.cum_probs = np.cumsum(self.probs)
        self.min_val = min_val
    
    def sample(self, rng: np.random.RandomState = None) -> int:
        """
        Sample from the distribution.
        
        Args:
            rng: Optional random number generator
        
        Returns:
            Sampled value
        """
        if rng is None:
            u = np.random.random()
        else:
            u = rng.random()
        
        idx = np.searchsorted(self.cum_probs, u)
        return idx + self.min_val


class Queue:
    """
    Queue implementation for inventory state.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize queue.
        
        Args:
            max_size: Maximum size of the queue
        """
        self.data = deque(maxlen=max_size)
    
    def push(self, item):
        """Push item to the queue."""
        self.data.append(item)
    
    def pop(self):
        """Pop item from the queue."""
        if not self.data:
            return 0
        return self.data.popleft()
    
    def peek(self, idx: int = 0):
        """Peek at an item in the queue."""
        if not self.data or idx >= len(self.data):
            return 0
        return self.data[idx]
    
    def sum(self) -> int:
        """Sum of all items in the queue."""
        return sum(self.data)
    
    def size(self) -> int:
        """Size of the queue."""
        return len(self.data)
    
    def clear(self):
        """Clear the queue."""
        self.data.clear()
    
    def to_list(self) -> List:
        """Convert queue to list."""
        return list(self.data)


class ZeroShotLostSalesInventoryControlMDP(MDP):
    """
    Zero-shot Lost Sales Inventory Control MDP.
    
    This MDP implements the Super-MDP for lost sales inventory control
    as described in the paper "Zero-shot generalization in Inventory Management:
    Train, then Estimate and Decide".
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MDP with configuration parameters.
        
        Args:
            config: Configuration parameters
        """
        super().__init__(config)
        
        # Set default parameters
        self.discount_factor = config.get("discount_factor", 0.99)
        self.max_p = config.get("max_p", 10.0)
        self.min_p = config.get("min_p", 1.0)
        self.p = config.get("p", 5.0)
        self.h = config.get("h", 1.0)
        self.min_randomYield = config.get("min_randomYield", 0.5)
        self.max_leadtime = config.get("max_leadtime", 3)
        self.min_leadtime = config.get("min_leadtime", 0)
        self.max_num_cycles = config.get("max_num_cycles", 1)
        self.max_order_size = config.get("max_order_size", 10)
        self.max_system_inv = config.get("max_system_inv", 20)
        
        # Demand parameters
        self.demand_cycles = config.get("demand_cycles", [1])
        self.mean_demand = config.get("mean_demand", [5.0])
        self.std_demand = config.get("std_demand", [2.0])
        self.max_demand = config.get("max_demand", 15)
        self.min_demand = config.get("min_demand", 0)
        self.max_period_demand = config.get("max_period_demand", 15)
        
        # Distribution parameters
        self.leadtime_probs = config.get("leadtime_probs", [0.5, 0.5, 0.0, 0.0])
        self.non_crossing_leadtime_rv_probs = config.get("non_crossing_leadtime_rv_probs", [0.5, 0.5, 0.0, 0.0])
        
        # Feature flags
        self.evaluate = config.get("evaluate", False)
        self.train_stochastic_leadtimes = config.get("train_stochastic_leadtimes", True)
        self.train_cyclic_demand = config.get("train_cyclic_demand", True)
        self.train_random_yield = config.get("train_random_yield", True)
        self.censored_demand = config.get("censored_demand", False)
        self.censored_leadtime = config.get("censored_leadtime", False)
        self.order_crossover = config.get("order_crossover", False)
        self.random_yield = config.get("random_yield", False)
        self.censored_random_yield = config.get("censored_random_yield", False)
        self.yield_when_realized = config.get("yield_when_realized", False)
        
        # Random yield parameters
        self.random_yield_case = config.get("random_yield_case", 0)
        self.random_yield_probs_crossover = config.get("random_yield_probs_crossover", [])
        self.random_yield_probs = config.get("random_yield_probs", [])
        self.min_yield = config.get("min_yield", 0.0)
        
        # Initialize random yield distribution
        yield_probs = config.get("yield_probs", [0.0, 0.0, 0.0, 1.0, 0.0])
        self.random_yield_dist = DiscreteDist(yield_probs, min_val=0)
        
        # Variance parameters
        self.p_var = config.get("p_var", 0.0)
        self.alpha_var = config.get("alpha_var", 0.0)
        self.k_var = config.get("k_var", 0.0)
        
        # Feature control
        self.include_all_features = config.get("include_all_features", True)
        self.random_yield_features_size = config.get("random_yield_features_size", 5)
        
        # Set random seed if provided
        self.seed = config.get("seed", None)
        self.rng = np.random.RandomState(self.seed)
    
    def num_valid_actions(self) -> int:
        """
        Returns the number of valid actions.
        
        Returns:
            Number of valid actions (max order size + 1)
        """
        return self.max_order_size + 1
    
    def num_flat_features(self) -> int:
        """
        Returns the number of features in the flattened feature vector.
        
        Returns:
            Number of state features
        """
        # Base features:
        # - Inventory level
        # - Pipeline inventory (max_leadtime elements)
        # - Period 
        # - Period statistics (2 per cycle)
        # - Leadtime statistics (max_leadtime * 2)
        base_features = 1 + self.max_leadtime + 1 + (2 * self.max_num_cycles) + (self.max_leadtime * 2)
        
        # Random yield features (if enabled)
        if self.random_yield:
            base_features += self.random_yield_features_size
        
        return base_features
    
    def get_initial_state(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Get an initial state for the MDP.
        
        Args:
            seed: Optional random seed
        
        Returns:
            Initial state
        """
        # Set random seed if provided
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Create state
        state = {
            "period": 0,
            "cycle_length": len(self.demand_cycles),
            "demand_cycles": self.demand_cycles.copy(),
            "collect_demand_statistics": [True] * len(self.demand_cycles),
            
            # Inventory state
            "inventory": 0,
            "pipeline": Queue(max_size=self.max_leadtime),
            "total_inv": 0,
            
            # Demand statistics
            "past_demands": [[] for _ in range(len(self.demand_cycles))],
            "cumulative_demands": [[] for _ in range(len(self.demand_cycles))],
            "censor_indicator": [[] for _ in range(len(self.demand_cycles))],
            "cycle_probs": [[] for _ in range(len(self.demand_cycles))],
            "cycle_min_demand": [self.min_demand] * len(self.demand_cycles),
            "mean_cycle_demand": self.mean_demand.copy(),
            "std_cycle_demand": self.std_demand.copy(),
            "period_count": [0] * len(self.demand_cycles),
            
            # Flags
            "collect_statistics": True,
            "censored_demand": self.censored_demand,
            "stochastic_leadtimes": self.train_stochastic_leadtimes,
            "censored_leadtime": self.censored_leadtime,
            "order_crossover": self.order_crossover,
            "random_yield": self.random_yield,
            "censored_random_yield": self.censored_random_yield,
            
            # Demand distributions
            "cumulative_pmfs": [[] for _ in range(len(self.demand_cycles))],
            "min_true_demand": [self.min_demand] * len(self.demand_cycles),
            
            # Cost parameters
            "p": self.p,
            "max_order_size": self.max_order_size,
            "max_system_inv": self.max_system_inv,
            "max_order_size_limit": self.max_order_size,
            "order_constraint": self.max_order_size,
            
            # Cycle-specific parameters
            "cycle_max_order_size": [self.max_order_size] * len(self.demand_cycles),
            "cycle_max_system_inv": [self.max_system_inv] * len(self.demand_cycles),
            
            # Leadtime statistics
            "estimated_leadtime_probs": self.leadtime_probs.copy(),
            "cumulative_leadtime_probs": np.cumsum(self.leadtime_probs).tolist(),
            "min_leadtime": self.min_leadtime,
            "max_leadtime": self.max_leadtime,
            "estimated_min_leadtime": self.min_leadtime,
            "estimated_max_leadtime": self.max_leadtime,
            "past_leadtimes": [],
            "orders_received": 0,
            
            # Random yield
            "random_yield_probs": self.random_yield_probs,
            "random_yield_probs_crossover": self.random_yield_probs_crossover,
            "random_yield_statistics": [],
            "random_yield_features": [0.0] * self.random_yield_features_size,
            "yield_when_realized": self.yield_when_realized
        }
        
        return state
    
    def is_action_valid(self, state: Dict[str, Any], action: int) -> bool:
        """
        Check if an action is valid in the given state.
        
        Args:
            state: Current state
            action: Action to check
            
        Returns:
            True if the action is valid, False otherwise
        """
        # Action must be an integer between 0 and max_order_size
        if not isinstance(action, int) or action < 0:
            return False
        
        # Check if action exceeds order constraint
        if action > state["order_constraint"]:
            return False
        
        # Check if action would exceed system inventory constraint
        total_inventory = state["inventory"] + state["pipeline"].sum() + action
        if total_inventory > state["max_system_inv"]:
            return False
        
        return True
    
    def get_next_state_reward(
        self, 
        state: Dict[str, Any], 
        action: int,
        seed: Optional[int] = None
    ) -> Tuple[Dict[str, Any], float, bool]:
        """
        Get the next state and reward after taking an action.
        
        Args:
            state: Current state
            action: Action to take
            seed: Optional random seed
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        # Set random seed if provided
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Make a deep copy of the state
        next_state = self._copy_state(state)
        
        # Apply action (order placement)
        reward = self._apply_action(next_state, action)
        
        # Generate demand and update inventory
        reward += self._process_demand(next_state)
        
        # Process leadtime and pipeline
        self._process_pipeline(next_state)
        
        # Update statistics
        self._update_statistics(next_state)
        
        # Increment period
        next_state["period"] += 1
        
        # Check if done (for finite horizon MDPs, always False for infinite horizon)
        done = False
        
        return next_state, reward, done
    
    def get_features(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get a feature vector representation of the state.
        
        Args:
            state: State to get features for
            
        Returns:
            Numpy array of features
        """
        features = []
        
        # Inventory level
        features.append(state["inventory"])
        
        # Pipeline inventory
        pipeline_list = state["pipeline"].to_list()
        pipeline_list.extend([0] * (self.max_leadtime - len(pipeline_list)))
        features.extend(pipeline_list)
        
        # Period features
        features.append(state["period"] % state["cycle_length"])
        
        # Demand statistics features
        for cycle in range(len(state["demand_cycles"])):
            features.append(state["mean_cycle_demand"][cycle])
            features.append(state["std_cycle_demand"][cycle])
        
        # Leadtime features
        for lt in range(self.max_leadtime + 1):
            lt_prob = state["estimated_leadtime_probs"][lt] if lt < len(state["estimated_leadtime_probs"]) else 0.0
            features.append(lt_prob)
            features.append(lt if lt < len(state["past_leadtimes"]) and lt < len(state["past_leadtimes"]) else 0)
        
        # Random yield features
        if state["random_yield"]:
            features.extend(state["random_yield_features"])
        
        return np.array(features, dtype=np.float32)
    
    def list_policies(self) -> Dict[str, str]:
        """
        Lists available policies for this MDP.
        
        Returns:
            Dictionary mapping policy IDs to descriptions
        """
        return {
            "base_stock": "Base stock policy",
            "myopic": "Myopic policy",
            "random": "Random policy"
        }
    
    def _copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a deep copy of the state.
        
        Args:
            state: State to copy
            
        Returns:
            Deep copy of the state
        """
        # Create a new state dictionary
        new_state = {}
        
        # Copy primitive types
        for key, value in state.items():
            if key == "pipeline":
                # Create a new Queue with the same contents
                new_state[key] = Queue(max_size=self.max_leadtime)
                for item in state[key].to_list():
                    new_state[key].push(item)
            elif isinstance(value, list):
                # Deep copy lists
                if len(value) > 0 and isinstance(value[0], list):
                    new_state[key] = [sublist.copy() for sublist in value]
                else:
                    new_state[key] = value.copy()
            else:
                # Direct copy for primitive types
                new_state[key] = value
        
        return new_state
    
    def _apply_action(self, state: Dict[str, Any], action: int) -> float:
        """
        Apply an action (place order) to the state.
        
        Args:
            state: Current state
            action: Action to take (order quantity)
            
        Returns:
            Immediate reward (ordering cost)
        """
        # Check if action is valid
        if not self.is_action_valid(state, action):
            raise ValueError(f"Invalid action {action} for state")
        
        # Calculate ordering cost
        ordering_cost = 0.0
        
        # Add order to pipeline if non-zero
        if action > 0:
            if state["random_yield"] and not state["yield_when_realized"]:
                # Apply random yield immediately if not realized at arrival
                realized_quantity = self._apply_random_yield(state, action)
                state["pipeline"].push(realized_quantity)
            else:
                # Add full order to pipeline
                state["pipeline"].push(action)
            
            # Add ordering cost
            ordering_cost = -state["p"] * action
        
        return ordering_cost
    
    def _process_demand(self, state: Dict[str, Any]) -> float:
        """
        Process demand and update inventory.
        
        Args:
            state: Current state
            
        Returns:
            Immediate reward (holding and stockout costs)
        """
        # Get current cycle and demand distribution
        cycle_idx = state["period"] % state["cycle_length"]
        mean_demand = state["mean_cycle_demand"][cycle_idx]
        std_demand = state["std_cycle_demand"][cycle_idx]
        
        # Generate demand
        demand = self._generate_demand(mean_demand, std_demand)
        
        # Limit demand to max_period_demand
        demand = min(demand, self.max_period_demand)
        
        # Record demand observation
        cycle_min_demand = state["cycle_min_demand"][cycle_idx]
        self._update_demand_statistics(state, cycle_idx, demand, True)
        
        # Calculate fulfilled demand
        fulfilled_demand = min(demand, state["inventory"])
        
        # Update inventory
        state["inventory"] = max(0, state["inventory"] - demand)
        
        # Calculate costs
        # Holding cost for inventory at end of period
        holding_cost = -state["inventory"] * self.h
        
        # Stockout cost (penalty for lost sales)
        lost_sales = demand - fulfilled_demand
        stockout_cost = -lost_sales * state["p"]
        
        return holding_cost + stockout_cost
    
    def _process_pipeline(self, state: Dict[str, Any]) -> None:
        """
        Process pipeline and update inventory.
        
        Args:
            state: Current state
        """
        # Process orders in pipeline
        if state["pipeline"].size() > 0:
            # Determine how many orders to deliver
            if state["stochastic_leadtimes"]:
                # Sample from leadtime distribution
                leadtime_dist = state["estimated_leadtime_probs"]
                u = self.rng.random()
                num_arrivals = 0
                cum_prob = 0.0
                
                for lt, prob in enumerate(leadtime_dist):
                    cum_prob += prob
                    if u <= cum_prob:
                        num_arrivals = lt + 1
                        break
            else:
                # Deterministic leadtime (always deliver the first item)
                num_arrivals = 1 if state["pipeline"].size() > 0 else 0
            
            # Process arrivals
            for _ in range(min(num_arrivals, state["pipeline"].size())):
                # Get order from pipeline
                order = state["pipeline"].pop()
                
                # Apply random yield if realized at arrival
                if state["random_yield"] and state["yield_when_realized"]:
                    order = self._apply_random_yield(state, order)
                
                # Add to inventory
                state["inventory"] += order
                state["orders_received"] += 1
    
    def _update_statistics(self, state: Dict[str, Any]) -> None:
        """
        Update state statistics.
        
        Args:
            state: Current state
        """
        # Update total inventory
        state["total_inv"] = state["inventory"] + state["pipeline"].sum()
        
        # Update period count
        cycle_idx = state["period"] % state["cycle_length"]
        state["period_count"][cycle_idx] += 1
    
    def _generate_demand(self, mean: float, std: float) -> int:
        """
        Generate random demand.
        
        Args:
            mean: Mean demand
            std: Standard deviation of demand
            
        Returns:
            Random demand value
        """
        # Generate using truncated normal distribution
        demand = self.rng.normal(mean, std)
        demand = max(self.min_demand, min(self.max_demand, int(round(demand))))
        return demand
    
    def _update_demand_statistics(
        self, 
        state: Dict[str, Any], 
        cycle: int, 
        demand: int, 
        uncensored: bool
    ) -> None:
        """
        Update demand statistics using Kaplan-Meier estimator.
        
        Args:
            state: Current state
            cycle: Current cycle index
            demand: Observed demand
            uncensored: Whether the demand is uncensored
        """
        if not state["collect_statistics"]:
            return
        
        # Record the demand
        state["past_demands"][cycle].append(demand)
        
        # Record censoring indicator
        state["censor_indicator"][cycle].append(1 if uncensored else 0)
        
        # Update demand statistics
        if uncensored:
            # Simple update for uncensored demand
            demands = [d for d, c in zip(state["past_demands"][cycle], state["censor_indicator"][cycle]) 
                      if c == 1]
            if demands:
                state["mean_cycle_demand"][cycle] = np.mean(demands)
                state["std_cycle_demand"][cycle] = max(0.1, np.std(demands))
        else:
            # For censored demand, use a more complex estimator (simplified here)
            censored_demands = [d for d, c in zip(state["past_demands"][cycle], state["censor_indicator"][cycle]) 
                               if c == 0]
            uncensored_demands = [d for d, c in zip(state["past_demands"][cycle], state["censor_indicator"][cycle]) 
                                 if c == 1]
            
            if uncensored_demands:
                state["mean_cycle_demand"][cycle] = np.mean(uncensored_demands)
                state["std_cycle_demand"][cycle] = max(0.1, np.std(uncensored_demands))
    
    def _apply_random_yield(self, state: Dict[str, Any], quantity: int) -> int:
        """
        Apply random yield to an order quantity.
        
        Args:
            state: Current state
            quantity: Order quantity
            
        Returns:
            Realized quantity after yield
        """
        if not state["random_yield"] or quantity == 0:
            return quantity
        
        # Use binomial yield model
        yield_rate = self.rng.uniform(self.min_randomYield, 1.0)
        realized_quantity = self.rng.binomial(quantity, yield_rate)
        
        # Update yield statistics
        state["random_yield_statistics"].append((quantity, realized_quantity))
        
        # Update yield features
        if len(state["random_yield_statistics"]) > 0:
            # Calculate average yield rate
            total_ordered = sum(q for q, _ in state["random_yield_statistics"])
            total_received = sum(r for _, r in state["random_yield_statistics"])
            avg_yield = total_received / total_ordered if total_ordered > 0 else 1.0
            
            # Update features (simplified)
            state["random_yield_features"][0] = avg_yield
            state["random_yield_features"][1] = yield_rate
            state["random_yield_features"][2] = min(1.0, quantity / self.max_order_size)
            state["random_yield_features"][3] = min(1.0, realized_quantity / self.max_order_size)
            state["random_yield_features"][4] = min(1.0, state["inventory"] / self.max_system_inv)
        
        return realized_quantity 