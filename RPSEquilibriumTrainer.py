import random
import numpy as np

# define constants
ROCK = 0
PAPER = 1
SCISSORS = 2
NUM_ACTIONS = 3
NUM_PLAYERS = 2
PLAYER1 = 0
PLAYER2 = 1

# global variables - initialize state of two players
regret_sum = np.zeros((NUM_PLAYERS, NUM_ACTIONS))
strategy = np.zeros((NUM_PLAYERS, NUM_ACTIONS))
strategy_sum = np.zeros((NUM_PLAYERS, NUM_ACTIONS))


def normalize(array):
    norm = np.abs(array).sum()
    return array / norm if norm != 0 else [0] * len(array)


def get_strategy(player):
    """
    Get player's strategy through regret-matching.
    """
    global regret_sum, strategy, strategy_sum

    # get current player's strategy from regret sum
    # note that we get 0 if regret_sum is negative
    strategy[player] = np.maximum(0, regret_sum[player])

    # normalize current player's strategy
    strategy[player] = normalize(strategy[player])

    # accumulate current player's strategy
    strategy_sum[player] = np.add(strategy_sum[player], strategy[player])

    # return current player's strategy
    return strategy[player]


def get_action(strategy):
    """
    Get random action based on mixed strategy distribution.
    """
    r = random.random()
    a = 0
    cumulative_probability = 0.0
    while a < NUM_ACTIONS - 1:
        cumulative_probability += strategy[a]
        if r < cumulative_probability:
            break
        a += 1
    return a


def get_player1_regret(act1, act2):
    """
    Get summation of regret of player 1's action for player 2's action.

    Args:
        act1: action of player 1
        act2: action of player 2
    Returns:
        summation of regret of player 1's action for player 2's action
    """

    # compute utility
    util = np.zeros(NUM_ACTIONS)
    util[act2] = 0
    util[(act2 + 1) % NUM_ACTIONS] = 1
    util[(act2 - 1 + NUM_ACTIONS) % NUM_ACTIONS] = -1

    # compute regret
    regret_sum = np.zeros(NUM_ACTIONS)
    for a in range(NUM_ACTIONS):
        regret = util[a] - util[act1]
        regret_sum[a] += regret

    # return regret
    return regret_sum


def get_player2_regret(act1, act2):
    """
    Get summation of regret of player 2's action for player 1's action.

    Args:
        act1: action of player 1
        act2: action of player 2
    Returns:
        summation of regret of player 2's action for player 1's action
    """

    # compute utility
    util = np.zeros(NUM_ACTIONS)
    util[act1] = 0
    util[(act1 + 1) % NUM_ACTIONS] = 1
    util[(act1 - 1 + NUM_ACTIONS) % NUM_ACTIONS] = -1

    # compute regret
    regret_sum = np.zeros(NUM_ACTIONS)
    for a in range(NUM_ACTIONS):
        regret = util[a] - util[act2]
        regret_sum[a] += regret

    # return regret
    return regret_sum


def train(iterations):
    """
    Train players.
    """
    global regret_sum

    util = np.zeros(NUM_ACTIONS)

    for _ in range(iterations):
        # calculate each player's strategy and action
        strategy1, strategy2 = get_strategy(PLAYER1), get_strategy(PLAYER2)
        act1, act2 = get_action(strategy1), get_action(strategy2)

        # calculate regret of player 1 (utility of player 1 due to player 2's action)
        regret_sum[PLAYER1] = get_player1_regret(act1, act2)

        # calculate regret of player 2 (utility of player 2 due to player 1's action)
        regret_sum[PLAYER2] = get_player2_regret(act1, act2)


def get_average_strategy(player):
    """
    모든 훈련 반복에 걸친 플레이어의 평균 혼합 전략을 얻습니다.
    """
    global strategy_sum

    avg_strategy = [0.0] * NUM_ACTIONS
    current_strategy_sum = strategy_sum[player]
    normalizing_sum = sum(current_strategy_sum)

    for a in range(NUM_ACTIONS):
        if normalizing_sum > 0:
            avg_strategy[a] = current_strategy_sum[a] / normalizing_sum
        else:
            avg_strategy[a] = 1.0 / NUM_ACTIONS

    return avg_strategy


if __name__ == "__main__":
    train(100000)

    actions = ["ROCK", "PAPER", "SCISSORS"]

    print("-------------------------------------")

    print("---- Player 1's Optimal Strategy ----")
    p1_strategy = get_average_strategy(0)
    for i, p in enumerate(p1_strategy):
        print(f"Action: {actions[i]:<10} Probability: {p:.3f}")

    print("---- Player 2's Optimal Strategy ----")
    p2_strategy = get_average_strategy(1)
    for i, p in enumerate(p2_strategy):
        print(f"Action: {actions[i]:<10} Probability: {p:.3f}")

    print("-------------------------------------")
