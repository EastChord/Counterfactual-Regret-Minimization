"""
Strategy manager for simultaneous-move games using Regret Matching algorithm.

This module provides functionality to manage strategies, regrets, and utilities
for players in simultaneous-move games. It implements the core regret matching
algorithm to learn optimal mixed strategies.
"""
import numpy as np


class SimultaneousStrategyManager:
    """
    Manages strategy learning for simultaneous-move games using regret matching.
    
    This class tracks utilities, regrets, and strategies for a single player
    in a simultaneous-move game. It implements the regret matching algorithm
    to iteratively improve the player's strategy based on counterfactual regrets.
    """
    
    def __init__(self, actions):
        """
        Initialize strategy manager for given set of actions.
        
        Args:
            actions (dict): Dictionary mapping action indices to action names
        """
        self.actions = actions  # Action mapping (index -> name)
        self.num_act = len(actions)  # Number of available actions
        
        # Initialize tracking arrays for regret matching algorithm
        self.utils = np.zeros(self.num_act)  # Current utilities for each action
        self.strategy_sum = np.zeros(self.num_act)  # Cumulative strategy sum for averaging
        self.regrets = np.zeros(self.num_act)  # Current iteration regrets
        self.regret_sum = np.zeros(self.num_act)  # Cumulative regret sum

    def normalize(self, array):
        """
        Normalize array to sum to 1, handling negative values and zero sum.
        
        This function ensures the array represents a valid probability distribution
        by normalizing positive values and handling edge cases where all values
        are non-positive.
        
        Args:
            array (np.ndarray): Array to normalize
            
        Returns:
            np.ndarray: Normalized array (probability distribution)
        """
        # Sum only positive values to avoid negative probabilities
        norm = np.maximum(0, array).sum()
        # Return normalized array or uniform distribution if sum is zero
        return array / norm if norm != 0 else np.zeros(len(array))

    def get_strategy(self):
        """
        Generate current strategy based on accumulated regrets.
        
        This method implements the core regret matching algorithm:
        1. Set strategy proportional to positive regrets
        2. Use uniform distribution if all regrets are non-positive
        3. Update cumulative strategy sum for averaging
        
        Returns:
            np.ndarray: Current strategy (probability distribution over actions)
        """
        # Set strategy proportional to positive regrets
        new_strategy = np.maximum(0, self.regret_sum)
        new_strategy = self.normalize(new_strategy)
        
        # Use uniform strategy if no positive regrets (exploration)
        if np.all(self.regret_sum <= 0):
            new_strategy = np.full(self.num_act, 1.0 / self.num_act)
        
        # Accumulate strategy for averaging
        self.strategy_sum += new_strategy
        return new_strategy

    def get_average_strategy(self):
        """
        Compute average strategy over all iterations.
        
        This method returns the time-averaged strategy, which converges to
        the optimal strategy in regret matching algorithms.
        
        Returns:
            np.ndarray: Average strategy (probability distribution over actions)
        """
        return self.normalize(self.strategy_sum)

    def compute_regrets(self, actual_utility: float):
        """
        Compute regrets for current iteration.
        
        Regret measures how much better each action would have been compared
        to the action that was actually taken.
        
        Args:
            actual_utility (float): Utility received from actual action taken
        """
        # Regret = counterfactual utility - actual utility
        self.regrets = self.utils - actual_utility

    def update_regret_sum(self, actual_utility: float, weight=1.0):
        """
        Update cumulative regret sum with current iteration regrets.
        
        This method computes regrets and adds them to the cumulative sum,
        which is used to generate future strategies.
        
        Args:
            actual_utility (float): Utility received from actual action taken
            weight (float, optional): Weight for regret update. Defaults to 1.0.
        """
        # Compute regrets for current iteration
        self.compute_regrets(actual_utility)
        # Add weighted regrets to cumulative sum
        self.regret_sum += weight * self.regrets

    def print_average_strategy(self):
        """
        Print the learned average strategy in a formatted table.
        
        Displays each action with its corresponding probability percentage
        in a readable format.
        """
        avg_strategy = self.get_average_strategy()
        for i, p in enumerate(avg_strategy):
            print(f"Action: {self.actions[i]:<10} Probability: {p * 100:.2f}%")