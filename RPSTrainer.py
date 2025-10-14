import random

# 상수 정의
ROCK = 0
PAPER = 1
SCISSORS = 2
NUM_ACTIONS = 3

# 전역 변수
regret_sum = [0.0] * NUM_ACTIONS  # 반복에서의 후회 합계
strategy = [0.0] * NUM_ACTIONS  # 현재 플레이어의 전략
strategy_sum = [0.0] * NUM_ACTIONS  # 반복에서의 전략 합계
opp_strategy = [0.3, 0.5, 0.2]  # 상대방의 고정된 전략


def get_strategy():
    """
    후회 매칭을 통해 현재 혼합 전략을 얻습니다.
    """
    global regret_sum, strategy, strategy_sum
    
    normalizing_sum = 0
    # 모든 액션에 대해 후회를 전략에 추가
    for a in range(NUM_ACTIONS):
        strategy[a] = max(0, regret_sum[a])
        normalizing_sum += strategy[a]
    
    # 모든 액션에 대해 전략을 정규화하고 전략을 누적
    for a in range(NUM_ACTIONS):
        # 전략을 정규화
        if normalizing_sum > 0:
            strategy[a] /= normalizing_sum
        else:
            strategy[a] = 1.0 / NUM_ACTIONS
        # 전략을 누적
        strategy_sum[a] += strategy[a]
    
    return strategy


def get_action(strategy):
    """
    혼합 전략 분포에 따라 랜덤 액션을 얻습니다.
    """
    r = random.random()  # 0과 1 사이의 랜덤 숫자 생성
    a = 0  # 첫 번째 액션
    cumulative_probability = 0.0
    
    # 액션 얻기
    while a < NUM_ACTIONS - 1:  # 마지막 액션은 포함되지 않지만 문제없음
        cumulative_probability += strategy[a]
        if cumulative_probability > r:
            break
        a += 1  # 다음 액션
    
    return a


def train(iterations):
    """
    플레이어를 훈련시킵니다.
    """
    global regret_sum, strategy_sum
    
    action_utility = [0.0] * NUM_ACTIONS
    
    # 모든 반복에 대해
    for i in range(iterations):
        # 후회 매칭된 혼합 전략 액션을 얻음
        current_strategy = get_strategy()
        my_action = get_action(current_strategy)
        other_action = get_action(opp_strategy)
        
        # 액션 유틸리티 계산
        action_utility[other_action] = 0  # 무승부 케이스
        action_utility[(other_action + 1) % NUM_ACTIONS] = 1  # 승리 케이스
        action_utility[(other_action - 1 + NUM_ACTIONS) % NUM_ACTIONS] = -1  # 패배 케이스
        
        # 모든 액션에 대해 액션 후회를 누적
        for a in range(NUM_ACTIONS):
            regret = action_utility[a] - action_utility[my_action]
            regret_sum[a] += regret


def get_average_strategy():
    """
    모든 훈련 반복에 걸친 평균 혼합 전략을 얻습니다.
    """
    global strategy_sum
    
    avg_strategy = [0.0] * NUM_ACTIONS
    normalizing_sum = sum(strategy_sum)
    
    # 모든 액션에 대해 전략의 합계를 정규화
    for a in range(NUM_ACTIONS):
        if normalizing_sum > 0:
            avg_strategy[a] = strategy_sum[a] / normalizing_sum
        else:
            avg_strategy[a] = 1.0 / NUM_ACTIONS
    
    return avg_strategy


if __name__ == "__main__":
    train(100000)
    avg_strategy = get_average_strategy()
    actions = ["ROCK", "PAPER", "SCISSORS"]

    print("Opponent's Strategy:", opp_strategy)
    print("Calculated Optimal Strategy:", [round(p, 3) for p in avg_strategy])
    print("-" * 30)
    for i, p in enumerate(avg_strategy):
        print(f"Action: {actions[i]}, Probability: {p:.3f}")