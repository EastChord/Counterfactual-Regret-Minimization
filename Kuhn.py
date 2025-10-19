import random
from sequantial_strategy_manager import SequentialStrategyManagerMap

cards = [1, 2, 3]
actions = {0: "PASS", 1: "BET"}


sm_map = SequentialStrategyManagerMap()


class GameState:
    def __init__(self, info, curr_player, opp_player, priv_cards):
        self.history = ""
        self.info = info
        self.curr_player = curr_player
        self.opp_player = opp_player
        self.priv_cards = priv_cards

    def get_payoff_if_terminal(self):
        is_win = self.priv_cards[self.curr_player] > self.priv_cards[self.opp_player]
        if self.history == "pp":
            return 1 if is_win else -1
        if self.history == "bb" or self.history == "pbb":
            return 2 if is_win else -2
        if self.history == "bp" or self.history == "pbp":
            return 1
        return None

    def get_next_state(self, action):
        action_str = "p" if action == 0 else "b"
        next_history = self.history + action_str
        next_player = 1 - self.curr_player
        next_info = str(self.priv_cards[next_player]) + next_history
        next_state = GameState(
            next_info, next_player, self.curr_player, self.priv_cards
        )
        next_state.history = next_history
        return next_state


def cfr(state, reach_probs):
    # end of game
    if (payoff := state.get_payoff_if_terminal()) is not None:
        return payoff
    # update and get strategy of current state
    sm = sm_map.get_strategy_manager(state.info, actions)
    sm.update_and_get_strategy(reach_probs[state.curr_player])
    # get utility for each action and get game state utility.
    for action in actions.keys():
        new_reach_probs = reach_probs.copy()
        new_reach_probs[state.curr_player] *= sm.strategy[action]
        next_state = state.get_next_state(action)
        sm.utils[action] = -cfr(next_state, new_reach_probs)
    # update regrets
    sm.update_regret_sum(reach_probs[state.opp_player])
    return sm.get_average_util()


def train(iterations):
    util = 0.0
    for _ in range(iterations):
        shuffled_cards = cards.copy()
        random.shuffle(shuffled_cards)
        state = GameState(str(shuffled_cards[0]), 0, 1, shuffled_cards)
        reach_probs = [1.0, 1.0]
        util += cfr(state, reach_probs)
    avg_game_value = util / iterations
    return avg_game_value


if __name__ == "__main__":
    avg_game_value = train(iterations=1000000)
    print(f"\nAverage game value: {avg_game_value:.3f}")
    sm_map.print_average_strategy()
