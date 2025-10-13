import random
import numpy as np

# 상수 정의
ROCK = 0
PAPER = 1
SCISSORS = 2
NUM_ACTIONS = 3
NUM_PLAYERS = 2

# 전역 변수 - 두 플레이어의 상태 초기화
regret_sum = [[0.0] * NUM_ACTIONS for _ in range(NUM_PLAYERS)]
strategy = [[0.0] * NUM_ACTIONS for _ in range(NUM_PLAYERS)]
strategy_sum = [[0.0] * NUM_ACTIONS for _ in range(NUM_PLAYERS)]


def normalize(array):
    norm = np.abs(array).sum()
    return array / norm if norm != 0 else [0] * len(array)


def get_strategy(player):
    """
    후회 매칭을 통해 플레이어의 현재 혼합 전략을 얻습니다.
    """
    global regret_sum, strategy, strategy_sum

    normalizing_sum = 0
    current_strategy = strategy[player]
    current_regret_sum = regret_sum[player]

    for a in range(NUM_ACTIONS):
        current_strategy[a] = max(0, current_regret_sum[a])

    current_strategy = normalize(current_strategy)
    
    for a in range(NUM_ACTIONS):
        strategy_sum[player][a] += current_strategy[a]

    return current_strategy


def get_action(strategy):
    """
    혼합 전략 분포에 따라 랜덤 액션을 얻습니다.
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


def train(iterations):
    """
    플레이어들을 훈련시킵니다.
    """
    global regret_sum

    action_utility = [0.0] * NUM_ACTIONS

    for _ in range(iterations):
        # 두 플레이어의 전략과 액션 계산
        strategy_p1 = get_strategy(0)
        strategy_p2 = get_strategy(1)
        action_p1 = get_action(strategy_p1)
        action_p2 = get_action(strategy_p2)

        # 플레이어 1의 후회 계산 (플레이어 2의 액션으로 인한 플레이어 1의 유틸리티)
        action_utility[action_p2] = 0
        action_utility[(action_p2 + 1) % NUM_ACTIONS] = 1
        action_utility[(action_p2 - 1 + NUM_ACTIONS) % NUM_ACTIONS] = -1
        for a in range(NUM_ACTIONS):
            regret = action_utility[a] - action_utility[action_p1]
            regret_sum[0][a] += regret

        # 플레이어 2의 후회 계산 (플레이어 1의 액션으로 인한 플레이어 2의 유틸리티)
        action_utility[action_p1] = 0
        action_utility[(action_p1 + 1) % NUM_ACTIONS] = 1
        action_utility[(action_p1 - 1 + NUM_ACTIONS) % NUM_ACTIONS] = -1
        for a in range(NUM_ACTIONS):
            regret = action_utility[a] - action_utility[action_p2]
            regret_sum[1][a] += regret


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

    print("--- Player 1 Final Strategy ---")
    p1_strategy = get_average_strategy(0)
    for i, p in enumerate(p1_strategy):
        print(f"Action: {actions[i]:<10} Probability: {p:.3f}")

    print("\n--- Player 2 Final Strategy ---")
    p2_strategy = get_average_strategy(1)
    for i, p in enumerate(p2_strategy):
        print(f"Action: {actions[i]:<10} Probability: {p:.3f}")
