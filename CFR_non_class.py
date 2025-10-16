import random
import numpy as np
from sklearn.preprocessing import normalize

CARD_NAMES = {1: "Q", 2: "K", 3: "A"}

# define actions as constants
PASS = 0
BET = 1
NUM_ACTIONS = 2

# iterations
ITERATIONS = 500000
ITERATIONS_THRESHOLD = ITERATIONS / 20

# Global variable to store all information sets
node_map = {}


def normalize(values):
    """
    Normalize values

    Example:
        [1, 3, 2] -> [1/6, 3/6, 2/6]
    Args:
        values (list): List of values to normalize.
    Returns:
        list: Normalized values.
    """
    norm = np.abs(values).sum()
    return values / norm if norm != 0 else [0] * len(values)


def create_node(info_set):
    """
    Create a new node for the given information set.
    """
    return {
        "info_set": info_set,
        "regret_sum": [0.0] * NUM_ACTIONS,
        "strategy": [0.0] * NUM_ACTIONS,
        "strategy_sum": [0.0] * NUM_ACTIONS,
    }


def get_strategy(node):
    """
    Get current mixed strategy through regret-matching.
    """

    # get and normalize strategy from summation of regret
    node["strategy"] = np.maximum(0, node["regret_sum"])
    node["strategy"] = normalize(node["strategy"])

    # if all strategies are 0, get uniform strategy
    if sum(node["strategy"]) == 0:
        node["strategy"] = np.ones(NUM_ACTIONS) / NUM_ACTIONS

    # return strategy
    return node["strategy"]


def get_information(cards, history, player):
    return CARD_NAMES[cards[player]] + "-" + history


def cfr(cards, history, weight0, weight1):
    """
    Counterfactual Regret Minimization algorithm.
    This is reculsive function.
    현재 노드에서 플레이어의 전략 효용을 계산.
    """

    # get current player
    curr_player = len(history) % 2  # 0 for first curr_player, 1 for second curr_player
    curr_opponent = 1 - curr_player  # 1 for first curr_player, 0 for second curr_player

    # return payoff for terminal states
    # TODO: 위로 올릴 것.
    # situation 1: opponent calls
    if history == "bb" or history == "pbb":
        return 2 if cards[curr_player] > cards[curr_opponent] else -2
    # situation 2: all players check
    if history == "pp":
        return 1 if cards[curr_player] > cards[curr_opponent] else -1
    # situation 3: opponent folds
    if history == "pbp" or history == "bp":
        return 1

    # get information index of current player
    info_idx = get_information(cards, history, curr_player)

    # create information set node if nonexistent
    if node_map.get(info_idx) is None:
        node_map[info_idx] = create_node(info_idx)

    # get information and strategy
    node = node_map.get(info_idx)
    strategy = get_strategy(node)

    # accumulate the strategy with reach probability for compute average strategy after
    if curr_player == 0:
        node["strategy_sum"] += weight0 * np.array(strategy)
    else:
        node["strategy_sum"] += weight1 * np.array(strategy)

    # initialize utility
    util = [0.0] * NUM_ACTIONS
    node_util = 0.0

    # for each action, compute utility and accumulate node utility
    for a in range(NUM_ACTIONS):
        # compute next action and next history
        next_action = "p" if a == PASS else "b"
        next_history = history + next_action

        # compute utility for each action
        if curr_player == 0:
            next_weight = weight0 * strategy[a]
            util[a] = -cfr(cards, next_history, next_weight, weight1)
        else:
            next_weight = weight1 * strategy[a]
            util[a] = -cfr(cards, next_history, weight0, next_weight)

        # compute node utility
        node_util += strategy[a] * util[a]

    # for each action, compute and accumulate counterfactual regret
    for a in range(NUM_ACTIONS):
        regret = util[a] - node_util
        if curr_player == 0:
            node["regret_sum"][a] += weight1 * regret
        else:
            node["regret_sum"][a] += weight0 * regret
    return node_util


def train(iterations):
    """
    Find Nash Equilibrium of Kuhn Poker game using CFR algorithm.
    Args:
        iterations (int): Number of training iterations.
    Returns:
        float: Average game value.
    """

    cards = [1, 2, 3]
    util = 0.0

    for _ in range(iterations):

        # shuffle cards
        random.shuffle(cards)

        # Get utility by calling cfr function recursively
        util += cfr(cards, "", 1.0, 1.0)

    return util / iterations


if __name__ == "__main__":

    # train CFR
    average_game_value = train(ITERATIONS)

    # print average strategy of all of information sets
    print(f"\nAverage game value: {average_game_value}")
    sorted_nodes = sorted(node_map.items())
    for _, node in sorted_nodes:
        avg = normalize(node["strategy_sum"])
        print(f"{node['info_set']:>4}: [PASS: {avg[0]:.3f}, BET: {avg[1]:.3f}]")
