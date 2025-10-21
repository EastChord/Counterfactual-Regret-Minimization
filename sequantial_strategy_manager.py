"""Sequential Strategy Manager for CFR Algorithm.

This module provides classes for managing game strategies in sequential games
using Counterfactual Regret Minimization (CFR). It tracks regrets, strategies,
and reach probabilities for each information set in the game tree.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


class SequentialStrategyManager:
    """Manages strategy and regret tracking for a single information set in CFR.
    
    Implements regret matching algorithm to compute strategies based on accumulated
    regrets. Tracks both current strategies and average strategies over time.
    
    Attributes:
        info (list): Information set identifier
        actions (list): Available actions at this information set
        num_actions (int): Number of available actions
        strategy (np.ndarray): Current strategy (probability distribution over actions)
        strategy_sum (np.ndarray): Accumulated strategy for computing average
        regrets (np.ndarray): Current regrets (unused, kept for compatibility)
        regret_sum (np.ndarray): Accumulated regrets for regret matching
        util (float): Expected utility at this information set
        my_reach (float): Reach probability for the current player
        opp_reach (float): Reach probability for the opponent
    """
    
    # Display formatting constants
    PERCENTAGE_DECIMALS = 2
    DEFAULT_NODE_TYPE = "node"
    
    def __init__(self, info: list, actions: list) -> None:
        """Initialize strategy manager for an information set.
        
        Args:
            info (list): Information set identifier (e.g., [my_claim, opponent_claim])
            actions (list): Available actions at this information set
            
        Raises:
            ValueError: If actions list is empty
            TypeError: If info is not a list
        """
        if not actions:
            raise ValueError("Actions list cannot be empty")
        if not isinstance(info, list):
            raise TypeError("Info must be a list")
            
        self.info = info
        self.actions = actions
        self.num_actions = len(actions)
        self.strategy = np.zeros(self.num_actions)
        self.strategy_sum = np.zeros(self.num_actions)
        # regrets kept for compatibility but not used in current implementation
        self.regrets = np.zeros(self.num_actions)
        self.regret_sum = np.zeros(self.num_actions)
        self.util = 0.0
        self.my_reach = 0.0
        self.opp_reach = 0.0
        # Additional attributes for Kuhn.py compatibility
        self.utils = np.zeros(self.num_actions)

    def normalize(self, array: np.ndarray) -> np.ndarray:
        """Normalize an array to sum to 1, handling zero-sum cases.
        
        Args:
            array (np.ndarray): Array to normalize
            
        Returns:
            np.ndarray: Normalized array (sums to 1) or zero array if input sum is 0
        """
        # Use only positive values for regret matching
        positive_array = np.maximum(0, array)
        norm = positive_array.sum()
        return positive_array / norm if norm != 0 else np.zeros(len(positive_array))
    
    def get_uniform_strategy(self) -> np.ndarray:
        """Get uniform probability distribution over all actions.
        
        Returns:
            np.ndarray: Uniform strategy where each action has equal probability
        """
        return np.full(self.num_actions, 1.0 / self.num_actions)

    def get_strategy(self) -> np.ndarray:
        """Compute current strategy using regret matching.
        
        Uses positive regrets to compute strategy via regret matching algorithm.
        Falls back to uniform strategy if all regrets are non-positive.
        
        Returns:
            np.ndarray: Current strategy (probability distribution over actions)
        """
        new_strategy = self.normalize(self.regret_sum)
        # Use uniform strategy when no regrets are positive (exploration)
        if np.all(new_strategy == 0):
            new_strategy = self.get_uniform_strategy()
        self.strategy = new_strategy
        # Weight strategy by reach probability for average computation
        self.strategy_sum += self.my_reach * self.strategy
        return self.strategy

    def update_and_get_strategy(self, reach_prob: float) -> np.ndarray:
        """Update strategy and return it (for Kuhn.py compatibility).
        
        Args:
            reach_prob (float): Reach probability for current player
            
        Returns:
            np.ndarray: Current strategy (probability distribution over actions)
        """
        self.my_reach = reach_prob
        return self.get_strategy()

    def get_average_strategy(self) -> np.ndarray:
        """Compute average strategy over all training iterations.
        
        The average strategy converges to Nash equilibrium as training progresses.
        
        Returns:
            np.ndarray: Average strategy (probability distribution over actions)
        """
        average_strategy = np.copy(self.strategy_sum)
        normalizing_sum = np.sum(average_strategy)
        
        return average_strategy / normalizing_sum if normalizing_sum > 0 else self.get_uniform_strategy()

    def get_average_util(self) -> float:
        """Get average utility (for Kuhn.py compatibility).
        
        Returns:
            float: Average utility at this node
        """
        return self.util

    def update_regret_sum(self, regret_or_opp_reach) -> None:
        """Update accumulated regrets based on counterfactual values.
        
        Supports two calling patterns:
        1. update_regret_sum(regret_array) - for liar_die.py compatibility
        2. update_regret_sum(opp_reach_prob) - for Kuhn.py compatibility
        
        Args:
            regret_or_opp_reach: Either regret array or opponent reach probability
        """
        if isinstance(regret_or_opp_reach, (int, float)):
            # Kuhn.py pattern: update_regret_sum(opp_reach_prob)
            opp_reach = regret_or_opp_reach
            for a in range(self.num_actions):
                # Regret = action value - node value
                regret_value = self.utils[a] - self.util
                self.regret_sum[a] += opp_reach * regret_value
        else:
            # liar_die.py pattern: update_regret_sum(regret_array)
            regret = regret_or_opp_reach
            for a in range(len(self.strategy)):
                # Regret = action value - node value
                regret[a] -= self.util
                self.regret_sum[a] += self.opp_reach * regret[a]

    def update_reach_probability(self, next_node: 'SequentialStrategyManager', 
                                action_prob: np.ndarray, action_index: int, 
                                my_reach_multiplier: float = 1.0, 
                                opp_reach_multiplier: float = 1.0) -> None:
        """Propagate reach probabilities to child node.
        
        Args:
            next_node (SequentialStrategyManager): Child node to update
            action_prob (np.ndarray): Action probabilities at current node
            action_index (int): Index of action leading to child node
            my_reach_multiplier (float): Multiplier for current player reach. Defaults to 1.0
            opp_reach_multiplier (float): Multiplier for opponent reach. Defaults to 1.0
        """
        next_node.my_reach += action_prob[action_index] * self.my_reach * my_reach_multiplier
        next_node.opp_reach += self.opp_reach * opp_reach_multiplier

    def calculate_utility(self, child_utilities: np.ndarray) -> float:
        """Calculate expected utility based on child utilities and current strategy.
        
        Args:
            child_utilities (np.ndarray): Utilities of child nodes for each action
            
        Returns:
            float: Expected utility at this node
        """
        self.util = 0.0
        for i, child_util in enumerate(child_utilities):
            # Ensure index is within strategy bounds
            if i < len(self.strategy):
                self.util += self.strategy[i] * child_util
        return self.util

    def reset_reach_probabilities(self) -> None:
        """Reset reach probabilities to zero for next iteration."""
        self.my_reach = 0.0
        self.opp_reach = 0.0

    def reset_strategy_sum(self) -> None:
        """Reset accumulated strategy sum to zero."""
        self.strategy_sum.fill(0.0)

    def initialize_reach_probabilities(self, my_reach: float = 1.0, opp_reach: float = 1.0) -> None:
        """Set initial reach probabilities for root node.
        
        Args:
            my_reach (float): Initial reach probability for current player. Defaults to 1.0
            opp_reach (float): Initial reach probability for opponent. Defaults to 1.0
        """
        self.my_reach = my_reach
        self.opp_reach = opp_reach

    def get_strategy_dataframe(self, node_type: str = "node", additional_info: Optional[dict] = None) -> pd.DataFrame:
        """Convert average strategy to formatted DataFrame for display.
        
        Args:
            node_type (str): Type of node ("claim", "response", or "node"). Defaults to "node"
            additional_info (Optional[dict]): Additional information to include (unused). Defaults to None
            
        Returns:
            pd.DataFrame: Formatted strategy with action labels and percentages
        """
        average_strategy = self.get_average_strategy()
        
        # Set info as first column
        data = {'info': [str(self.info)]}
        
        # Add each action as separate column (converted to percentage, 2 decimal places)
        for i, prob in enumerate(average_strategy):
            # Map action indices to game-specific labels
            if node_type == "claim":
                # Claim node: info[0] is current claim, can claim higher values
                current_claim = self.info[0] if len(self.info) > 0 else 0
                actual_action = current_claim + i + 1
                data[f'claim {actual_action}'] = [f"{prob * 100:.{self.PERCENTAGE_DECIMALS}f}%"]
            elif node_type == "response":
                # Response node: action 0 = DOUBT, action 1 = ACCEPT
                if i == 0:
                    data['DOUBT'] = [f"{prob * 100:.{self.PERCENTAGE_DECIMALS}f}%"]
                elif i == 1:
                    data['ACCEPT'] = [f"{prob * 100:.{self.PERCENTAGE_DECIMALS}f}%"]
            else:
                # Default to generic action labels
                data[f'action {i}'] = [f"{prob * 100:.{self.PERCENTAGE_DECIMALS}f}%"]
        
        df = pd.DataFrame(data)
        
        # Replace NaN with empty string for cleaner display
        df = df.fillna('')
        
        return df

    def print_strategy_summary(self, node_type: str = "node", additional_info: Optional[dict] = None) -> pd.DataFrame:
        """Print formatted summary of node's strategy.
        
        Args:
            node_type (str): Type of node for formatting. Defaults to "node"
            additional_info (Optional[dict]): Additional information (unused). Defaults to None
            
        Returns:
            pd.DataFrame: Strategy DataFrame that was printed
        """
        df = self.get_strategy_dataframe(node_type, additional_info)
        print(f"\n{node_type.upper()} Strategy Summary:")
        print(f"Info: {self.info}")
        print(df.to_string(index=False))
        return df


class SequentialStrategyManagerMap:
    """Container for managing multiple SequentialStrategyManager nodes.
    
    Provides a mapping from information sets to their corresponding strategy managers.
    Enables batch operations on all nodes such as resetting strategies or reach probabilities.
    
    Attributes:
        node_map (dict): Mapping from information set tuples to SequentialStrategyManager instances
    """
    
    def __init__(self) -> None:
        """Initialize empty strategy manager map."""
        self.node_map = {}

    def create_node(self, info: list, actions: list) -> SequentialStrategyManager:
        """Create or retrieve strategy manager for an information set.
        
        Args:
            info (list): Information set identifier
            actions (list): Available actions
            
        Returns:
            SequentialStrategyManager: Strategy manager for this information set
        """
        key = tuple(info)
        if key not in self.node_map:
            self.node_map[key] = SequentialStrategyManager(info, actions)
        return self.node_map[key]

    def get_node(self, info: list) -> Optional[SequentialStrategyManager]:
        """Retrieve strategy manager for an information set.
        
        Args:
            info (list): Information set identifier
            
        Returns:
            Optional[SequentialStrategyManager]: Strategy manager, or None if not found
        """
        key = tuple(info)
        return self.node_map.get(key, None)

    def get_strategy_manager(self, info: str, actions: dict) -> SequentialStrategyManager:
        """Get or create strategy manager for Kuhn.py compatibility.
        
        Args:
            info (str): Information set identifier (string format for Kuhn.py)
            actions (dict): Dictionary mapping action indices to action names
            
        Returns:
            SequentialStrategyManager: Strategy manager for this information set
        """
        # Convert string info to list format for compatibility
        info_list = [info]
        actions_list = list(actions.keys())
        return self.create_node(info_list, actions_list)

    def reset_all_strategy_sums(self) -> None:
        """Reset accumulated strategies for all nodes."""
        for node in self.node_map.values():
            if node is not None:
                node.reset_strategy_sum()

    def reset_all_reach_probabilities(self) -> None:
        """Reset reach probabilities for all nodes."""
        for node in self.node_map.values():
            if node is not None:
                node.reset_reach_probabilities()

    def get_all_strategies_dataframe(self, node_type: str = "node", additional_info_func=None) -> pd.DataFrame:
        """Collect strategies from all nodes into single DataFrame.
        
        Args:
            node_type (str): Type of nodes for formatting. Defaults to "node"
            additional_info_func: Function to add info (unused). Defaults to None
            
        Returns:
            pd.DataFrame: Combined strategies from all nodes
        """
        all_dfs = []
        
        for key, node in self.node_map.items():
            if node is not None:
                df = node.get_strategy_dataframe(node_type)
                all_dfs.append(df)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            # Replace NaN with empty string for cleaner display
            combined_df = combined_df.fillna('')
            return combined_df
        else:
            return pd.DataFrame()

    def print_all_strategies(self, node_type: str = "node", additional_info_func=None) -> pd.DataFrame:
        """Print strategies for all nodes.
        
        Args:
            node_type (str): Type of nodes for formatting. Defaults to "node"
            additional_info_func: Function to add info (unused). Defaults to None
            
        Returns:
            pd.DataFrame: Combined strategies DataFrame
        """
        df = self.get_all_strategies_dataframe(node_type, additional_info_func)
        if not df.empty:
            print(f"\n=== ALL {node_type.upper()} STRATEGIES ===")
            print(df.to_string(index=False))
        return df

    def print_average_strategy(self) -> None:
        """Print average strategies for Kuhn.py compatibility.
        
        Displays strategies in a format compatible with Kuhn.py's expected output.
        """
        print("\n=== AVERAGE STRATEGIES ===")
        for key, node in self.node_map.items():
            if node is not None:
                info_str = str(node.info[0]) if len(node.info) > 0 else str(key)
                print(f"\nInformation Set: {info_str}")
                avg_strategy = node.get_average_strategy()
                for i, prob in enumerate(avg_strategy):
                    action_name = f"Action {i}"
                    print(f"  {action_name}: {prob:.3f}")
