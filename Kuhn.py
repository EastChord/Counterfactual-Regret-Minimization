"""
Counterfactual Regret Minimization (CFR) algorithm implementation for Kuhn Poker.

This module provides functionality to learn strategies close to Nash equilibrium
in Kuhn Poker using the CFR algorithm. CFR is an iterative algorithm for finding
optimal strategies in imperfect information games.
"""
import random
from tqdm import tqdm
from sequantial_strategy_manager import SequentialStrategyManagerMap
from memory_monitor import MemoryMonitor

# Kuhn Poker game constants
cards = [1, 2, 3]  # Three cards (1, 2, 3)
actions = {0: "PASS", 1: "BET"}  # Possible actions: pass(0) or bet(1)

# Global strategy manager map - tracks strategies for all game states
sm_map = SequentialStrategyManagerMap()


class GameState:
    """
    Represents a specific state in Kuhn Poker game.
    
    Game state consists of current player, opponent player, each player's private cards,
    and the betting pot information up to the current point.
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
        self.history = ""  # Action history (p=pass, b=bet) - kept for compatibility
        self.info = info  # Information set identifier
        self.curr_player = curr_player  # Current player
        self.opp_player = opp_player  # Opponent player
        self.priv_cards = priv_cards  # Private cards
        
        # Pot-based betting system
        self.pot = 2  # Initial pot (both players start with 1 chip each)
        self.player_bets = [1, 1]  # Amount each player has bet [player0_bet, player1_bet]
        self.betting_round = 0  # Current betting round (0 = initial, 1+ = after first bet)
        self.actions_taken = 0  # Number of actions taken in current round

    def is_terminal(self):
        """
        Check if game is in terminal state using pot-based logic.
        
        Terminal conditions after at least 2 actions:
        - Bets are equal (pp, bb, pbb) → showdown
        - Bets are unequal AND the last action was pass (bp, pbp) → fold
        Note: pb (pass then bet) is NOT terminal; another action is required.
        
        Returns:
            bool: True if game is terminal, False otherwise
        """
        # Need at least two actions for a hand to end
        if len(self.history) < 2:
            return False

        # Equal bets → showdown
        if self.player_bets[0] == self.player_bets[1]:
            return True

        # Unequal bets: terminal only if the last action was a pass (fold)
        return self.history[-1] == 'p'

    def get_payoff(self):
        """
        Get current player's payoff if game is in terminal state using pot-based logic
        with payouts normalized to the opponent's contribution (matches classic Kuhn).
        
        Payoff rules:
        - Equal bets (showdown): winner gets opponent's contribution
        - Unequal bets (fold): player who bet more wins opponent's contribution
        
        Returns:
            int: Current player's payoff (assumes game is terminal)
        """
        opp = self.opp_player
        opp_contrib = self.player_bets[opp]

        # Fold scenario: unequal bets → the player who bet more wins
        if self.player_bets[0] != self.player_bets[1]:
            if self.player_bets[self.curr_player] > self.player_bets[opp]:
                return opp_contrib
            else:
                return -opp_contrib

        # Showdown: equal bets → higher card wins
        is_win = self.priv_cards[self.curr_player] > self.priv_cards[self.opp_player]
        return opp_contrib if is_win else -opp_contrib

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
        if not self.is_terminal():
            return None
        return self.get_payoff()

    def get_next_state(self, action):
        """
        Generate next game state after performing given action using pot-based logic.
        
        Args:
            action (int): Action to perform (0=pass, 1=bet)
            
        Returns:
            GameState: New game state after action is performed
        """
        # Convert action to string (p=pass, b=bet) for compatibility
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
        
        # Copy pot information
        next_state.pot = self.pot
        next_state.player_bets = self.player_bets.copy()
        next_state.betting_round = self.betting_round
        next_state.actions_taken = self.actions_taken + 1
        
        # Update pot and betting based on action
        if action == 1:  # Bet action
            # Increase current player's bet by 1
            next_state.player_bets[self.curr_player] += 1
            # Increase pot by 1
            next_state.pot += 1
            # Increment betting round
            next_state.betting_round += 1
        # Pass action (action == 0) doesn't change pot or bets
        
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
    if state.is_terminal():
        return state.get_payoff()
    
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
    sm.util = sum(sm.strategy[a] * sm.utils[a] for a in actions.keys())
    
    # Update regret values using opponent's reach probability
    sm.update_regret_sum(reach_probs[state.opp_player])
    
    # Return the expected value of the current state (v_sigma)
    return sm.util


def train(iterations, enable_memory_monitoring=True, memory_check_interval=10000):
    """
    Learn Kuhn Poker strategy using CFR algorithm with optional memory monitoring.
    
    This function simulates games for given number of iterations to learn
    strategies close to Nash equilibrium.
    
    Args:
        iterations (int): Number of iterations to train
        enable_memory_monitoring (bool): Whether to enable memory monitoring
        memory_check_interval (int): How often to check memory usage (every N iterations)
        
    Returns:
        float: Average game value of learned strategy
    """
    util = 0.0
    
    # Initialize memory monitor if enabled
    # memory_monitor = None
    # if enable_memory_monitoring:
    #     memory_monitor = MemoryMonitor(
    #         alert_threshold_mb=2000.0,  # 2GB threshold
    #         gc_threshold_multiplier=1.5,  # 1.5x initial memory triggers GC
    #         enable_auto_gc=True
    #     )
    #     print("Memory monitoring enabled")
    #     memory_monitor.print_memory_stats(0)
    
    # Use tqdm to show progress bar
    for i in tqdm(range(iterations), desc="CFR Training", unit="iter"):
        # Memory monitoring
        # if memory_monitor:
        #     snapshot = memory_monitor.monitor_iteration(
        #         iteration=i,
        #         check_interval=memory_check_interval,
        #         description=f"CFR iteration {i}"
        #     )
            
        #     # Print detailed memory stats at key intervals
        #     if i > 0 and (i % (iterations // 10) == 0 or i == iterations - 1):
        #         memory_monitor.print_memory_stats(i)
        
        # Reset strategy sums at halfway point
        if i == iterations // 2:
            sm_map.reset_all_strategy_sums()
            # if memory_monitor:
            #     print("Strategy sums reset - memory may decrease")
        
        # Shuffle cards randomly for each iteration to start new game
        shuffled_cards = cards.copy()
        random.shuffle(shuffled_cards)
        
        # Create game state starting with first player (index 0)
        state = GameState(str(shuffled_cards[0]), 0, 1, shuffled_cards)
        
        # Initial reach probabilities are 1.0 (certain to reach)
        reach_probs = [1.0, 1.0]
        
        # Execute CFR algorithm and accumulate game value
        util += cfr(state, reach_probs)
    
    # Final memory statistics
    # if memory_monitor:
    #     print("\n" + "="*60)
    #     print("FINAL MEMORY STATISTICS")
    #     print("="*60)
    #     memory_monitor.print_memory_stats(iterations)
        
    #     # Memory trend analysis
    #     trend = memory_monitor.get_memory_trend()
    #     print(f"Memory trend analysis: {trend}")
        
    #     # Check if memory usage is concerning
    #     stats = memory_monitor.get_memory_stats()
    #     if stats['memory_increase_mb'] > 1000:  # More than 1GB increase
    #         print("⚠️  WARNING: Significant memory increase detected!")
    #         print("   Consider reducing batch size or increasing GC frequency")
    #     elif stats['memory_increase_mb'] < 100:  # Less than 100MB increase
    #         print("✅ Memory usage is stable and efficient")
    #     else:
    #         print("ℹ️  Memory usage is within acceptable range")
    
    # Calculate average game value
    avg_game_value = util / iterations
    return avg_game_value


if __name__ == "__main__":
    # Learn strategy with memory monitoring enabled
    print("Starting CFR training with memory monitoring...")
    avg_game_value = train(
        iterations=1000000, 
        enable_memory_monitoring=True,
        memory_check_interval=100000  # Check memory every 50,000 iterations
    )
    print(f"\nAverage game value: {avg_game_value:.3f}")
    
    # Print learned average strategy
    sm_map.print_average_strategy()
