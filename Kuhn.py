"""
Counterfactual Regret Minimization (CFR) algorithm implementation for Kuhn Poker.

This module provides functionality to learn strategies close to Nash equilibrium
in Kuhn Poker using the CFR algorithm. CFR is an iterative algorithm for finding
optimal strategies in imperfect information games.
"""
import random
from sequantial_strategy_manager import SequentialStrategyManagerMap

# Kuhn Poker game constants
cards = [1, 2, 3]  # Three cards (1, 2, 3)
actions = {0: "PASS", 1: "BET"}  # Possible actions: pass(0) or bet(1)

# Global strategy manager map - tracks strategies for all game states
sm_map = SequentialStrategyManagerMap()


class GameState:
    """
    Represents a specific state in Kuhn Poker game.
    
    Game state consists of current player, opponent player, each player's private cards,
    and the action history up to the current point.
    """
    
    def __init__(self, info, curr_player, opp_player, priv_cards):
        """
        Initialize game state.
        
        Args:
            info (str): Information set identifier for current player
            curr_player (int): Current player to act (0 or 1)
            opp_player (int): Opponent player (0 or 1)
            priv_cards (list): Private cards for each player [player0_card, player1_card]
        """
        self.history = ""  # Action history (p=pass, b=bet)
        self.info = info  # Information set identifier
        self.curr_player = curr_player  # Current player
        self.opp_player = opp_player  # Opponent player
        self.priv_cards = priv_cards  # Private cards

    def get_payoff_if_terminal(self):
        """
        Check if game is in terminal state and return current player's payoff if so.
        
        Kuhn Poker payoff rules:
        - pp (both pass): Player with higher card gets 1 point
        - bb/pbb (both bet): Player with higher card gets 2 points
        - bp/pbp (only one bets): Betting player gets 1 point
        
        Returns:
            int or None: Current player's payoff if game is terminal, 
                        None if game continues
        """
        is_win = self.priv_cards[self.curr_player] > self.priv_cards[self.opp_player]
        
        # Both players passed - winner determined by card comparison
        if self.history == "pp":
            return 1 if is_win else -1
        
        # Both players bet - winner determined by higher bet
        if self.history == "bb" or self.history == "pbb":
            return 2 if is_win else -2
        
        # Only one player bet - betting player wins
        if self.history == "bp" or self.history == "pbp":
            return 1
        
        return None  # Game not yet terminated

    def get_next_state(self, action):
        """
        Generate next game state after performing given action.
        
        Args:
            action (int): Action to perform (0=pass, 1=bet)
            
        Returns:
            GameState: New game state after action is performed
        """
        # Convert action to string (p=pass, b=bet)
        action_str = "p" if action == 0 else "b"
        next_history = self.history + action_str
        
        # Switch to next player
        next_player = 1 - self.curr_player
        
        # Generate information set for next player (card + history)
        next_info = str(self.priv_cards[next_player]) + next_history
        
        # Create new game state
        next_state = GameState(
            next_info, next_player, self.curr_player, self.priv_cards
        )
        next_state.history = next_history
        return next_state


def cfr(state, reach_probs):
    """
    Core recursive function of Counterfactual Regret Minimization (CFR) algorithm.
    
    This function executes CFR algorithm at given game state to:
    1. Update strategy for current state
    2. Calculate expected value for each action
    3. Update regret values
    
    Args:
        state (GameState): Current game state
        reach_probs (list): Probability each player reaches current state [p1_prob, p2_prob]
        
    Returns:
        float: Expected value at current state (from current player's perspective)
    """
    # Return payoff if game is terminated
    if (payoff := state.get_payoff_if_terminal()) is not None:
        return payoff
    
    # Get strategy manager for current state and update strategy
    sm = sm_map.get_strategy_manager(state.info, actions)
    sm.update_and_get_strategy(reach_probs[state.curr_player])
    
    # Calculate expected value for each action
    for action in actions.keys():
        # Calculate new reach probabilities by multiplying current player's reach prob with strategy prob
        new_reach_probs = reach_probs.copy()
        new_reach_probs[state.curr_player] *= sm.strategy[action]
        
        # Move to next state and recursively calculate expected value
        next_state = state.get_next_state(action)
        # Sign flip because it's from opponent's perspective
        sm.utils[action] = -cfr(next_state, new_reach_probs)
    
    # Calculate node's expected value (v_sigma) using current strategy
    node_util = sum(sm.strategy[a] * sm.utils[a] for a in actions.keys())
    sm.util = node_util  # Store node value for regret calculation
    
    # Update regret values using opponent's reach probability
    sm.update_regret_sum(reach_probs[state.opp_player])
    
    # Return the expected value of the current state (v_sigma)
    return sm.util


def train(iterations):
    """
    Learn Kuhn Poker strategy using CFR algorithm.
    
    This function simulates games for given number of iterations to learn
    strategies close to Nash equilibrium.
    
    Args:
        iterations (int): Number of iterations to train
        
    Returns:
        float: Average game value of learned strategy
    """
    util = 0.0
    
    for _ in range(iterations):
        # Shuffle cards randomly for each iteration to start new game
        shuffled_cards = cards.copy()
        random.shuffle(shuffled_cards)
        
        # Create game state starting with first player (index 0)
        state = GameState(str(shuffled_cards[0]), 0, 1, shuffled_cards)
        
        # Initial reach probabilities are 1.0 (certain to reach)
        reach_probs = [1.0, 1.0]
        
        # Execute CFR algorithm and accumulate game value
        util += cfr(state, reach_probs)
    
    # Calculate average game value
    avg_game_value = util / iterations
    return avg_game_value


if __name__ == "__main__":
    # Learn strategy with 1,000,000 iterations
    avg_game_value = train(iterations=1000000)
    print(f"\nAverage game value: {avg_game_value:.3f}")
    
    # Print learned average strategy
    sm_map.print_average_strategy()
