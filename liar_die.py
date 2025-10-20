import numpy as np
import pandas as pd
from sequantial_strategy_manager import SequentialStrategyManagerMap as SMap
from typing import Dict, Any

# Fixed seed for reproducibility
np.random.seed(1)
# Default number of training iterations
ITERATION = 100000

class LiarDieTrainer:
    """Trainer for Liar's Dice game using Counterfactual Regret Minimization (CFR).
    
    Implements the CFR algorithm to find Nash equilibrium strategies for a simplified
    version of Liar's Dice where each player has one die.
    
    Attributes:
        sides (int): Number of sides on the dice
        response_map (SMap): Strategy manager for response decisions (doubt/accept)
        claim_map (SMap): Strategy manager for claim decisions
        DOUBT (int): Action constant for doubting opponent's claim
        ACCEPT (int): Action constant for accepting opponent's claim
    """
    
    # Constants
    DOUBT = 0
    ACCEPT = 1
    PROGRESS_REPORT_INTERVAL = 10000
    MAX_SIDES_LIMIT = 20
    
    def __init__(self, sides: int) -> None:
        """Initialize the trainer with game tree nodes for all possible game states.
        
        Args:
            sides (int): Number of sides on the dice (typically 6)
            
        Raises:
            ValueError: If sides is not positive or exceeds reasonable limit
        """
        if sides <= 0:
            raise ValueError("Number of sides must be positive")
        if sides > self.MAX_SIDES_LIMIT:
            raise ValueError(f"Number of sides too large (max: {self.MAX_SIDES_LIMIT})")
            
        self.sides = sides
        self.response_map = SMap()
        self.claim_map = SMap()

        # Create response nodes for all possible claim pairs
        # Response actions depend on claim level: accept only for 0, doubt only for max, both otherwise
        for my_claim in range(sides + 1):
            for opponent_claim in range(my_claim + 1, sides + 1):
                info = [my_claim, opponent_claim]
                actions = ([self.ACCEPT] if opponent_claim == 0 else
                          [self.DOUBT] if opponent_claim == sides else [self.DOUBT, self.ACCEPT])
                self.response_map.create_node(info, actions)

        # Create claim nodes for all possible claim-roll combinations
        # Claim actions are only higher values than current claim (can't claim same or lower)
        for opponent_claim in range(sides):
            for roll in range(1, sides + 1):
                info = [opponent_claim, roll]
                actions = list(range(opponent_claim + 1, sides + 1))
                self.claim_map.create_node(info, actions)

    def initialize(self) -> np.ndarray:
        """Initialize a single training iteration with random dice rolls.
        
        Returns:
            np.ndarray: Array of fixed dice rolls for this iteration, one per claim level
        """
        fixed_roll = np.random.randint(1, self.sides + 1, size=self.sides)
        claim_info = [0, fixed_roll[0]]
        init_node = self.claim_map.get_node(claim_info)
        init_node.initialize_reach_probabilities(1.0, 1.0)
        return fixed_roll

    def forward_accumulate_response(self, opponent_claim: int, fixed_roll: np.ndarray) -> None:
        """Propagate reach probabilities through response nodes in forward pass.
        
        Args:
            opponent_claim (int): Current claim level made by opponent
            fixed_roll (np.ndarray): Fixed dice rolls for this iteration
        """
        # No responses possible when there's no prior claim
        if opponent_claim == 0: return
        
        for my_claim in range(opponent_claim):
            node = self.response_map.get_node([my_claim, opponent_claim])
            action_prob = node.get_strategy()
            
            # Skip if claim level exceeds maximum (terminal state)
            if opponent_claim >= self.sides:
                continue

            next_node = self.claim_map.get_node([opponent_claim, fixed_roll[opponent_claim]])
            node.update_reach_probability(next_node, action_prob, self.ACCEPT)
    
    def forward_accumulate_claim(self, opponent_claim: int, fixed_roll: np.ndarray) -> None:
        """Propagate reach probabilities through claim nodes in forward pass.
        
        Args:
            opponent_claim (int): Current claim level
            fixed_roll (np.ndarray): Fixed dice rolls for this iteration
        """
        if opponent_claim >= self.sides: return

        node = self.claim_map.get_node([opponent_claim, fixed_roll[opponent_claim]])
        action_prob = node.get_strategy()

        for my_claim in range(opponent_claim + 1, self.sides + 1):
            claim_probability = action_prob[my_claim - opponent_claim - 1]

            # Optimization to skip zero-probability branches
            if claim_probability <= 0.0:
                continue

            next_node = self.response_map.get_node([opponent_claim, my_claim])
            # Reach probabilities swap between players due to turn structure
            next_node.my_reach += node.opp_reach
            next_node.opp_reach += claim_probability * node.my_reach

    def forward_accumulation(self, fixed_roll: np.ndarray) -> None:
        """Execute complete forward pass to accumulate reach probabilities.
        
        Args:
            fixed_roll (np.ndarray): Fixed dice rolls for this iteration
        """
        for opponent_claim in range(self.sides + 1):
            self.forward_accumulate_response(opponent_claim, fixed_roll)
            self.forward_accumulate_claim(opponent_claim, fixed_roll)

    def backward_propagate_claim(self, opponent_claim: int, fixed_roll: np.ndarray) -> None:
        """Compute regrets and utilities for claim nodes in backward pass.
        
        Args:
            opponent_claim (int): Current claim level
            fixed_roll (np.ndarray): Fixed dice rolls for this iteration
        """
        if opponent_claim >= self.sides: return

        node = self.claim_map.get_node([opponent_claim, fixed_roll[opponent_claim]])

        regret = np.zeros(node.num_actions)
        for my_claim in range(opponent_claim + 1, self.sides + 1):
            claim_index = my_claim - opponent_claim - 1
            next_node = self.response_map.get_node([opponent_claim, my_claim])
            # Negated because we're evaluating from opponent's perspective
            regret[claim_index] = -next_node.util

        node.calculate_utility(regret)
        node.update_regret_sum(regret)
        node.reset_reach_probabilities()

    def backward_propagate_response(self, opponent_claim: int, fixed_roll: np.ndarray) -> None:
        """Compute regrets and utilities for response nodes in backward pass.
        
        Args:
            opponent_claim (int): Current claim level
            fixed_roll (np.ndarray): Fixed dice rolls for this iteration
        """
        if opponent_claim == 0: return

        for my_claim in range(opponent_claim):
            node = self.response_map.get_node([my_claim, opponent_claim])

            regret = np.zeros(2)
            # Doubt succeeds when opponent's claim exceeds their actual roll
            doubt_util = 1.0 if opponent_claim > fixed_roll[my_claim] else -1.0
            regret[self.DOUBT] = doubt_util
            regret[self.ACCEPT] = (self.claim_map.get_node([opponent_claim, fixed_roll[opponent_claim]]).util
                                   if opponent_claim < self.sides else 0.0)

            node.calculate_utility(regret)
            node.update_regret_sum(regret)
            node.reset_reach_probabilities()

    def backward_propagation(self, fixed_roll: np.ndarray) -> None:
        """Execute complete backward pass to compute regrets and utilities.
        
        Processes nodes in reverse order to propagate utilities correctly.
        
        Args:
            fixed_roll (np.ndarray): Fixed dice rolls for this iteration
        """
        # Reverse order ensures child utilities are computed first
        for opponent_claim in range(self.sides, -1, -1):
            self.backward_propagate_claim(opponent_claim, fixed_roll)
            self.backward_propagate_response(opponent_claim, fixed_roll)

    def reset_strategy_sum(self, iteration: int, iterations: int) -> None:
        """Reset strategy sums at halfway point to discount early exploration.
        
        Args:
            iteration (int): Current iteration number
            iterations (int): Total number of training iterations
        """
        # Reset halfway to focus on converged strategies
        if iteration == iterations // 2:
            self.response_map.reset_all_strategy_sums()
            self.claim_map.reset_all_strategy_sums()

    def print_strategy(self) -> Dict[str, Any]:
        """Display the computed average strategies for all game states.
        
        Returns:
            dict: Dictionary containing DataFrames for 'initial_claim', 'response', and 'claim' strategies
        """
        # Collect response node strategies
        response_df = self.response_map.get_all_strategies_dataframe("response")
        
        # Collect claim node strategies
        claim_df = self.claim_map.get_all_strategies_dataframe("claim")
        
        # Print initial claim policies
        print("\n=== INITIAL CLAIM POLICIES ===")
        initial_claim_dfs = [self.claim_map.get_node([0, roll]).get_strategy_dataframe("initial_claim")
                             for roll in range(1, self.sides + 1)
                             if self.claim_map.get_node([0, roll])]
        initial_claim_df = pd.concat(initial_claim_dfs, ignore_index=True) if initial_claim_dfs else pd.DataFrame()
        
        if not initial_claim_df.empty:
            print(initial_claim_df.to_string(index=False))
        
        for name, df in [("RESPONSE STRATEGIES", response_df), ("CLAIM STRATEGIES", claim_df)]:
            if not df.empty:
                print(f"\n=== {name} ===")
                print(df.to_string(index=False))
        
        return {
            'initial_claim': initial_claim_df,
            'response': response_df,
            'claim': claim_df
        }

    def save_strategies_to_csv(self, filename_prefix: str = "liar_die_strategies") -> Dict[str, Any]:
        """Save computed strategies to CSV files.
        
        Args:
            filename_prefix (str): Prefix for output CSV filenames. Defaults to "liar_die_strategies"
            
        Returns:
            dict: Dictionary containing DataFrames for 'initial_claim', 'response', and 'claim' strategies
        """
        strategies = self.print_strategy()
        
        for strategy_type, df in strategies.items():
            if not df.empty:
                filename = f"{filename_prefix}_{strategy_type}.csv"
                df.to_csv(filename, index=False)
                print(f"\n{strategy_type.upper()} strategies saved to {filename}")
        
        return strategies

    def train(self, iterations: int) -> None:
        """Train the CFR algorithm for specified number of iterations.
        
        Each iteration performs forward and backward passes to update regrets and strategies.
        
        Args:
            iterations (int): Number of training iterations to run
        """
        for iteration in range(iterations):
            if iteration % self.PROGRESS_REPORT_INTERVAL == 0:
                print(f"Iteration {iteration:,} / {iterations:,}")

            fixed_roll = self.initialize()
            self.forward_accumulation(fixed_roll)
            self.backward_propagation(fixed_roll)
            self.reset_strategy_sum(iteration, iterations)

        self.print_strategy()


if __name__ == "__main__":
    trainer = LiarDieTrainer(sides=6)
    trainer.train(iterations=ITERATION)
