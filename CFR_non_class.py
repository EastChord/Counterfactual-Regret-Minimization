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

    # get strategy from regret sum and normalize the strategy
    # ? why we get 0 if regret_sum is negative?
    node["strategy"] = np.maximum(0, node["regret_sum"])
    node["strategy"] = normalize(node["strategy"])

    # get uniform strategy if all strategies are 0
    if sum(node["strategy"]) == 0:
        node["strategy"] = np.ones(NUM_ACTIONS) / NUM_ACTIONS

    return node["strategy"]


def node_to_string(node):
    """
    Convert node to string representation.
    """
    avg_strat = normalize(node["strategy_sum"])
    return f"{node['info_set']:>4}: [PASS: {avg_strat[0]:.3f}, BET: {avg_strat[1]:.3f}]"


def is_terminal_state(history):
    return len(history) > 1 and history != "pb"


def get_payoff(cards, history, player, opponent):
    # activate pass flag if final action is pass. (pp, bp, pbp)
    terminal_pass = history[-1] == "p"
    # activate double-bet flag if final actions are bet and bet. (bb, pbb)
    double_bet = history[-2:] == "bb"
    # activate higher flag if player card is higher than opponent card
    is_player_card_higher = cards[player] > cards[opponent]

    if terminal_pass:
        if history == "pp":
            return 1 if is_player_card_higher else -1
        else:
            return 1
    elif double_bet:
        return 2 if is_player_card_higher else -2


def get_information(cards, history, player):
    return CARD_NAMES[cards[player]] + "-" + history


def cfr(cards, history, p0, p1, iteration):
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
    if is_terminal_state(history):
        return get_payoff(cards, history, curr_player, curr_opponent)

    # get information index of current player
    info_idx = get_information(cards, history, curr_player)

    # create information set node if nonexistent
    if node_map.get(info_idx) is None:
        node_map[info_idx] = create_node(info_idx)

    # get information set node
    node = node_map.get(info_idx)

    # for each action, recursively call cfr with additional history and probability
    strategy = get_strategy(node)
    realization_weight = p0 if curr_player == 0 else p1

    # accumulate the current strategy with realization weight
    if iteration > ITERATIONS_THRESHOLD:
        node["strategy_sum"] += realization_weight * np.array(node["strategy"])

    util = [0.0] * NUM_ACTIONS
    node_util = 0.0
    for a in range(NUM_ACTIONS):

        # compute next history
        next_action = "p" if a == PASS else "b"
        next_history = history + next_action

        # compute utility for each action
        if curr_player == 0:
            util[a] = -cfr(cards, next_history, p0 * strategy[a], p1, iteration)
        else:
            util[a] = -cfr(cards, next_history, p0, p1 * strategy[a], iteration)

        # compute node utility
        node_util += strategy[a] * util[a]

    # for each action, compute and accumulate counterfactual regret
    for a in range(NUM_ACTIONS):
        regret = util[a] - node_util
        opponent_reach_prob = p1 if curr_player == 0 else p0
        node["regret_sum"][a] += opponent_reach_prob * regret

    return node_util


def train(iterations):
    """
    Find Nash Equilibrium of Kuhn Poker game using CFR algorithm.
    Args:
        iterations (int): Number of training iterations.
    """
    cards = [1, 2, 3]
    util = 0.0

    # 출력할 노드들 (예: 첫 번째 노드만)
    target_info_sets = set()

    for iteration in range(iterations):
        # shuffle cards
        random.shuffle(cards)

        # Get utility by calling cfr function recursively
        util += cfr(cards, "", 1.0, 1.0, iteration)

    print(f"\nAverage game value: {util / iterations}")

    # 최종 결과 출력
    sorted_nodes = sorted(node_map.items())
    for _, node in sorted_nodes:
        print(node_to_string(node))


if __name__ == "__main__":
    """
    Main execution block: start training.
    """
    # Monte Carlo method, the more iterations, the more accurate the result.
    train(ITERATIONS)
