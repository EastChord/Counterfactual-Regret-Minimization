import numpy as np
import random

# define Kuhn Poker
PASS = 0
BET = 1
ACTIONS = [PASS, BET]
NUM_ACTIONS = len(ACTIONS)
node_map = {}


class Node:
    def __init__(self, info_set):
        self.info_set = info_set
        self.regret_sum = np.zeros(NUM_ACTIONS)
        self.strategy = np.zeros(NUM_ACTIONS)
        self.strategy_sum = np.zeros(NUM_ACTIONS)

    def normalize(self, array):
        norm = np.abs(array).sum()
        return array / norm if norm != 0 else np.zeros(len(array))

    def get_strategy(self, weight):
        # compute new strategy
        new_strategy = np.maximum(0, self.regret_sum)
        new_strategy = self.normalize(new_strategy)

        # update strategy
        if np.all(new_strategy == 0):
            self.strategy = np.ones(NUM_ACTIONS) / NUM_ACTIONS
        else:
            self.strategy = new_strategy

        # update strategy sum to compute average strategy after
        self.strategy_sum += weight * self.strategy
        # self.strategy_sum += self.strategy

        # return strategy
        return self.strategy

    def get_average_strategy(self):
        avg_strategy = self.normalize(self.strategy_sum)
        return avg_strategy

    def __str__(self):
        avg_strategy = self.get_average_strategy()
        return f"{self.info_set:>4}: [PASS: {int(round(avg_strategy[0]*100)):>4}%, BET: {int(round(avg_strategy[1]*100)):>4}%]"


def cfr(cards, history, w0, w1):
    curr_player = len(history) % 2
    opponent = 1 - curr_player

    # return payoff for terminal states
    if history == "pp":
        return 1 if cards[curr_player] > cards[opponent] else -1
    if history == "bb" or history == "pbb":
        return 2 if cards[curr_player] > cards[opponent] else -2
    if history == "bp" or history == "pbp":
        return 1

    info_set = str(cards[curr_player]) + history

    # get node if existent, otherwise create node
    if (node := node_map.get(info_set)) is None:
        node = Node(info_set)
        node_map[info_set] = node

    # get weight for current player
    if curr_player == 0:
        weight = w0
    else:
        weight = w1

    strategy = node.get_strategy(weight)
    util = np.zeros(NUM_ACTIONS)
    node_util = 0.0

    # for each action, recursively call cfr with additional history and probability
    for a in ACTIONS:
        next_history = history + ("p" if a == PASS else "b")

        if curr_player == 0:
            util[a] = -cfr(cards, next_history, w0 * strategy[a], w1)
        else:
            util[a] = -cfr(cards, next_history, w0, w1 * strategy[a])

        node_util += strategy[a] * util[a]

    # for each action, compute and accumulate counterfactual regret
    for a in ACTIONS:
        regret = util[a] - node_util
        node.regret_sum[a] += (w1 if curr_player == 0 else w0) * regret
        # node.regret_sum[a] += regret

    return node_util


def train(iterations):
    cards = [1, 2, 3]
    util = 0.0

    for _ in range(iterations):
        random.shuffle(cards)
        util += cfr(cards, "", w0=1.0, w1=1.0)

    return util / iterations


if __name__ == "__main__":
    average_game_value = train(iterations=1000000)
    print(f"\nAverage game value: {average_game_value}")
    
    node_map_sorted = sorted(node_map.items())
    for _, node in node_map_sorted:
        print(node.__str__())
