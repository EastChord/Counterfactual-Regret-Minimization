"""
Rock-Paper-Scissors (RPS) game implementation using Regret Matching algorithm.

This module implements the classic Rock-Paper-Scissors game with two players
using the Regret Matching algorithm to learn optimal strategies. The algorithm
iteratively updates strategies based on regret values to converge to Nash equilibrium.
"""
import numpy as np
from simultaneous_strategy_manager import SimultaneousStrategyManager


class RPSGameManager:
    """
    Manages Rock-Paper-Scissors game logic and payoffs.
    
    This class handles game state, action selection, and payoff calculation
    for the classic Rock-Paper-Scissors game between two players.
    """
    
    def __init__(self):
        """
        Initialize RPS game manager with players and actions.
        
        Sets up the basic game structure with two players and three possible actions
        (rock, paper, scissors) following standard RPS rules.
        """
        self.players = {0: "player1", 1: "player2"}  # Player identifiers
        self.actions = {0: "rock", 1: "paper", 2: "scissors"}  # Available actions
        self.num_actions = len(self.actions)  # Number of possible actions
        self.num_players = len(self.players)  # Number of players

    def get_action(self, strategy):
        """
        Select an action based on given strategy probability distribution.
        
        Args:
            strategy (np.ndarray): Probability distribution over actions
            
        Returns:
            str: Selected action name (rock, paper, or scissors)
        """
        # Sample action according to strategy probabilities
        a = np.random.choice(self.num_actions, p=strategy)
        return self.actions[a]

    def get_payoff(self, act1, act2):
        """
        Calculate payoff for player 1 given both players' actions.
        
        RPS payoff rules:
        - Same action: tie (0 points)
        - Rock beats scissors, paper beats rock, scissors beats paper (1 point)
        - Otherwise: loss (-1 point)
        
        Args:
            act1 (str): Player 1's action
            act2 (str): Player 2's action
            
        Returns:
            int: Payoff for player 1 (-1, 0, or 1)
        """
        # Tie - both players chose same action
        if act1 == act2:
            return 0
        
        # Player 1 wins - check winning combinations
        if act1 == "rock" and act2 == "scissors":
            return 1
        if act1 == "paper" and act2 == "rock":
            return 1
        if act1 == "scissors" and act2 == "paper":
            return 1
        
        # Player 1 loses - all other cases
        return -1


# Global game manager instance
gm = RPSGameManager()

# Strategy managers for both players
sm1 = SimultaneousStrategyManager(gm.actions)
sm2 = SimultaneousStrategyManager(gm.actions)


def regret_matching():
    """
    Execute one iteration of regret matching algorithm.
    
    This function performs a single round of the regret matching algorithm:
    1. Get current strategies for both players
    2. Sample actions according to strategies
    3. Calculate actual payoffs
    4. Compute counterfactual utilities for all actions
    5. Update regret sums based on actual vs counterfactual payoffs
    """
    # Get current strategies for both players
    st1 = sm1.get_strategy()
    st2 = sm2.get_strategy()
    
    # Sample actions according to current strategies
    act1 = gm.get_action(st1)
    act2 = gm.get_action(st2)
    
    # Calculate actual payoffs for both players
    p1_actual_payoff = gm.get_payoff(act1, act2)
    p2_actual_payoff = gm.get_payoff(act2, act1)
    
    # Calculate counterfactual utilities for all possible actions
    for i, action in enumerate(gm.actions.values()):
        # What would player 1's utility be if they chose this action instead?
        sm1.utils[i] = gm.get_payoff(action, act2)
        # What would player 2's utility be if they chose this action instead?
        sm2.utils[i] = gm.get_payoff(action, act1)
    
    # Update regret sums based on actual vs counterfactual payoffs
    sm1.update_regret_sum(p1_actual_payoff)
    sm2.update_regret_sum(p2_actual_payoff)


def train(iterations):
    """
    Train both players using regret matching algorithm.
    
    This function runs the regret matching algorithm for the specified number
    of iterations to learn optimal strategies for both players.
    
    Args:
        iterations (int): Number of training iterations to perform
    """
    for _ in range(iterations):
        regret_matching()


if __name__ == "__main__":
    # Train both players with 1,000,000 iterations
    train(1000000)
    
    # Display learned optimal strategies
    print("---- Player 1's Optimal Strategy ----")
    sm1.print_average_strategy()
    print("---- Player 2's Optimal Strategy ----")
    sm2.print_average_strategy()
