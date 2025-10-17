import numpy as np
import random

# define Kuhn Poker
PASS = 0
BET = 1
ACT = [PASS, BET]
NUM_ACT = len(ACT)
node_map = {}


# normalize function
def normalize(array):
    norm = np.maximum(0, array).sum()
    return array / norm if norm != 0 else np.zeros(len(array))


class Node:
    def __init__(self, info):
        self.info = info
        self.reg_sum = np.zeros(NUM_ACT)
        self.strategy = np.zeros(NUM_ACT)
        self.strategy_sum = np.zeros(NUM_ACT)

    def update_strategy(self, weight):
        # negative regret is not considered.
        new_strategy = np.maximum(0, self.reg_sum)
        # summation of probability is 1.
        new_strategy = normalize(new_strategy)

        # corner case: strategy can not be all zero
        # uniform strategy is not necessary solution, but seems general.
        if np.all(new_strategy == 0):
            new_strategy = np.ones(NUM_ACT) / NUM_ACT

        self.strategy = new_strategy
        self.strategy_sum += weight * new_strategy

    def get_average_strategy(self):
        # summation of probability is 1.
        avg_strategy = normalize(self.strategy_sum)
        return avg_strategy

    def print_average_strategy(self):
        avg_strategy = self.get_average_strategy()
        return f"{self.info:>4}: [PASS: {int(round(avg_strategy[0]*100)):>4}%, BET: {int(round(avg_strategy[1]*100)):>4}%]"


def cfr(cards, history, w0, w1):
    # we assume that player 0 is the first player, and player 1 is the second player.
    curr_player = len(history) % 2
    opponent = 1 - curr_player

    # when the node is leaf node in game tree,
    # return payoff for terminal states
    # terminal states:
    #   - all players check (pp)
    #   - player 0 bets and player 1 calls (bb)
    #   - player 0 bets and player 1 folds (bp)
    #   - player 0 passes and player 1 bets (pbp)
    if history == "pp":
        return 1 if cards[curr_player] > cards[opponent] else -1
    if history == "bb" or history == "pbb":
        return 2 if cards[curr_player] > cards[opponent] else -2
    if history == "bp" or history == "pbp":
        return 1

    # cards is already shuffled
    # so we can get the current player's card by cards[curr_player]
    info = str(cards[curr_player]) + history

    # get node if existent, otherwise create node.
    if (node := node_map.get(info)) is None:
        node = Node(info)
        node_map[info] = node

    # this code is to get counterfactual regret
    if curr_player == 0:
        weight = w0
    else:
        weight = w1

    node.update_strategy(weight)

    # initialize
    util = np.zeros(NUM_ACT)
    node_util = 0.0

    # get utility for each action and get node utility.
    for a in ACT:
        next_history = history + ("p" if a == PASS else "b")

        # return utility of cfr is negative utility
        # because we want to maximize the utility of the current player.
        if curr_player == 0:
            util[a] = -cfr(cards, next_history, w0 * node.strategy[a], w1)
        else:
            util[a] = -cfr(cards, next_history, w0, w1 * node.strategy[a])

        node_util += node.strategy[a] * util[a]

    # compute and regret for each action
    for a in ACT:
        regret = util[a] - node_util

        # accumulate counterfactual regret with reach probability
        if curr_player == 0:
            node.reg_sum[a] += w1 * regret
        else:
            node.reg_sum[a] += w0 * regret

    return node_util


def train(iterations):
    # 3 can win 1 and 2, 2 can win 1.
    cards = [1, 2, 3]
    util = 0.0

    # training
    for _ in range(iterations):
        random.shuffle(cards)
        util += cfr(cards, "", w0=1.0, w1=1.0)

    # return average game value
    return util / iterations


if __name__ == "__main__":
    average_game_value = train(iterations=1000000)
    print(f"\nAverage game value: {average_game_value}")

    # print average strategy of all of information sets
    node_map_sorted = sorted(node_map.items())
    for _, node in node_map_sorted:
        print(node.print_average_strategy())
