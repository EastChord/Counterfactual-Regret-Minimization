import random
import numpy as np
from typing import Dict, Tuple


class Node:
    """
    FSICFR 알고리즘의 정보 집합(Information Set) 노드를 나타냅니다.


    이 클래스는 Java 코드의 내부 Node 클래스에 해당합니다.

    `regretSum`, `strategy`, `strategySum` 등 게임 상태와
    전략을 저장하기 위한 필드들을 포함합니다.
    """

    def __init__(self, num_actions: int):
        """
        노드를 초기화합니다.


        Args:
            num_actions (int): 이 노드에서 가능한 행동의 수.
        """
        self.num_actions = num_actions
        # regretSum: 누적 후회
        self.regret_sum = np.zeros(num_actions)
        # strategy: 현재 반복에서의 전략 (확률 분포)
        self.strategy = np.zeros(num_actions)
        # strategySum: 모든 반복에 걸친 전략의 누적 합계
        self.strategy_sum = np.zeros(num_actions)
        # u: 이 노드의 (예상) 유틸리티 (가치)
        self.u = 0.0
        # pPlayer: 현재 플레이어가 이 노드에 도달할 확률의 합
        self.p_player = 0.0
        # pOpponent: 상대방이 이 노드에 도달할 확률의 합
        self.p_opponent = 0.0

    def get_strategy(self) -> np.ndarray:
        """
        후회 매칭(Regret Matching)을 기반으로 현재 전략을 계산하고,
        전략 합계(strategy_sum)를 업데이트합니다.


        Java의 getStrategy 메서드에 해당합니다.
        """
        # 음수가 아닌 후회(positive regrets)만 사용합니다.
        self.strategy = np.maximum(0, self.regret_sum)
        normalizing_sum = np.sum(self.strategy)

        if normalizing_sum > 0:
            # 확률 분포로 정규화
            self.strategy /= normalizing_sum
        else:
            # 긍정적인 후회가 없으면 균등 분포 사용
            self.strategy = np.full(self.num_actions, 1.0 / self.num_actions)

        # 현재 플레이어의 도달 확률(p_player)로 가중치를 주어 전략 합계 업데이트
        #
        self.strategy_sum += self.p_player * self.strategy
        return self.strategy

    def get_average_strategy(self) -> np.ndarray:
        """
        모든 훈련 반복에 걸친 평균 전략을 반환합니다.


        Java의 getAverageStrategy 메서드에 해당합니다.
        """
        avg_strategy = np.copy(self.strategy_sum)
        normalizing_sum = np.sum(avg_strategy)

        if normalizing_sum > 0:
            # 전체 합계로 정규화하여 평균 확률을 구합니다.
            avg_strategy /= normalizing_sum
        else:
            # 전략이 누적되지 않았다면 균등 분포 반환
            avg_strategy = np.full(self.num_actions, 1.0 / self.num_actions)

        # 참고: 원본 Java 코드는 strategySum 자체를 수정하고 반환하는
        # 잠재적인 버그가 있을 수 있으나, 여기서는 더 안전한 방식(복사본 생성)으로
        # 평균 전략을 계산하여 반환합니다.
        return avg_strategy


DOUBT = 0
ACCEPT = 1
NO_CLAIM = 0


class LiarDieTrainer:
    """
    Liar Die 게임을 위한 FSICFR 훈련기 클래스입니다.


    Java의 LiarDieTrainer 클래스에 해당합니다.
    """

    def __init__(self, sides):
        """
        LiarDieTrainer를 초기화하고 필요한 노드들을 할당합니다.

        Args:
            sides (int): 사용할 주사위의 면 수.
        """

        self.sides = sides
        # 응답 노드를 할당합니다.
        # 응답 노드는 내 주장(0 ~ 6)과 상대 주장(0 ~ 6)으로 결정됩니다.
        self.node_res = np.empty((sides + 1, sides + 1), dtype=object)
        for my_clm in range(sides):
            # 상대의 주장은 내 주장보다 높아야합니다.
            for opp_clm in range(my_clm + 1, sides + 1):
                # 응답 노드에서는 accept, doubt 두 가지 행동이 가능합니다.
                # 아직 상대가 주장하지 않았다면 반드시 accept, 가장 높은 주사위 수를 주장하면 반드시 doubt해야 합니다.
                num_actions = 2 if (opp_clm != NO_CLAIM and opp_clm != sides) else 1
                self.node_res[my_clm, opp_clm] = Node(num_actions)

        # 주장 노드를 할당합니다.
        # 주장 노드는 상대 주장(0 ~ 6)과 굴린 주사위로 나온 숫자(1 ~ 6)로 결정됩니다.
        self.node_clm = np.empty((sides + 1, sides + 1), dtype=object)
        for opp_clm in range(sides):
            for roll in range(1, sides + 1):
                # 내 주장은 상대의 주장보다 높아야 합니다.
                num_actions = sides - opp_clm
                self.node_clm[opp_clm, roll] = Node(num_actions)

    def train(self, iterations: int):
        """
        FSICFR 알고리즘을 사용하여 훈련을 실행합니다.
        """

        for iteration in range(iterations):
            roll_fix = self._initialize_chance_events()
            self._forward_pass(roll_fix)
            self._backward_pass(roll_fix)
            self._reset_strategy_sums_if_needed(iteration, iterations)

    def _initialize_chance_events(self):
        """
        찬스 이벤트를 초기화하고 초기 주사위 굴림 값을 고정합니다.
        """
        roll_fix = np.random.randint(1, self.sides + 1, size=self.sides + 1)
        init_roll = roll_fix[0]
        # 초기 주사위 수를 고정했기 때문에, 맨 처음 주장 노드의 도달 확률은 1입니다.
        self.node_clm[0, init_roll].p_player = 1.0
        self.node_clm[0, init_roll].p_opponent = 1.0
        return roll_fix

    def _forward_pass(self, roll_fix):
        """
        순방향 전파: 도달 확률(실현 가중치) 누적
        """

        # 응답 노드의 순방향 전파
        for opp_clm in range(self.sides + 1):
            # 내가 첫 주장일 때는 도달 확률이 1이므로 계산하지 않습니다.
            if opp_clm == NO_CLAIM:
                continue
            # 내 주장 이후 상대방의 주장에 대해 도달 확률을 누적합니다.
            for my_clm in range(opp_clm):
                node = self.node_res[my_clm, opp_clm]
                # 도달 확률이 0인 노드는 계산하지 않습니다.
                if node.p_player == 0 and node.p_opponent == 0:
                    continue
                # 상대방의 주장이 최대 주장인 경우 accept하지 않으므로 도달 확률을 계산하지 않습니다.
                if opp_clm == sides:
                    continue
                # 다음 노드의 누적 도달 확률을 계산하기 위해 필요한 값을 계산합니다.
                action_prob = node.get_strategy()
                roll = roll_fix[opp_clm]
                next_node = self.node_clm[opp_clm, roll]
                # 누적 확률을 계산합니다.
                next_node.p_player += action_prob[ACCEPT] * node.p_player
                next_node.p_opponent += node.p_opponent

        # 주장 노드 순방향 전파
        for opp_clm in range(self.sides + 1):
            # 상대방의 주장이 최대 주장인 경우 더 이상 주장할 수 없습니다.
            if opp_clm == self.sides:
                continue
            roll = roll_fix[opp_clm]
            node = self.node_clm[opp_clm, roll]
            # 도달 확률이 0인 노드는 계산하지 않습니다.
            if node.p_player == 0 and node.p_opponent == 0:
                continue
            # 다음 노드의 누적 도달 확률을 계산하기 위해 필요한 값을 계산합니다.
            action_prob = node.get_strategy()
            for my_clm in range(opp_clm + 1, self.sides + 1):
                action_index = my_clm - opp_clm - 1
                next_claim_prob = action_prob[action_index]
                # 도달 확률이 0인 노드는 계산하지 않습니다.
                if next_claim_prob == 0:
                    continue
                # 누적 확률을 계산합니다.
                next_node = self.node_res[opp_clm, my_clm]
                next_node.p_player += next_claim_prob * node.p_player
                next_node.p_opponent += node.p_opponent

    def _backward_pass(self, roll_fix: np.ndarray):
        """역방향 전파: 유틸리티 및 후회 계산"""
        for opp_clm in range(self.sides, -1, -1):  # sides 부터 0 까지

            # (Visit claim nodes backward)
            if opp_clm < self.sides:
                roll = roll_fix[opp_clm]
                node = self.node_clm[opp_clm, roll]
                if node.p_player == 0 and node.p_opponent == 0:
                    continue  # 이 노드는 방문되지 않았음

                action_prob = node.strategy  # 순방향 패스에서 계산된 전략 사용
                node.u = 0.0
                num_actions = node.num_actions
                node_regret = np.zeros(num_actions)

                for my_clm in range(opp_clm + 1, self.sides + 1):
                    action_index = my_clm - opp_clm - 1
                    next_node = self.node_res[opp_clm, my_clm]

                    # 중요: 플레이어가 바뀌므로 유틸리티 부호를 뒤집습니다.
                    # (Java 코드는 -를 빠뜨린 것으로 보이며,
                    # 이는 Algorithm 2 및 제로섬 게임 로직과
                    # 모순됩니다. 여기서는 수정된 로직을 적용합니다.)
                    child_util = -next_node.u

                    node_regret[action_index] = child_util
                    node.u += action_prob[action_index] * child_util

                # 후회 계산 및 누적
                for a in range(num_actions):
                    node_regret[a] -= node.u  # (action_util - node_util)
                    # 상대방의 도달 확률(p_opponent)로 가중치 부여
                    node.regret_sum[a] += node.p_opponent * node_regret[a]

                # 다음 반복을 위해 확률 초기화
                node.p_player = 0.0
                node.p_opponent = 0.0

            # (Visit response nodes backward)
            if opp_clm > 0:
                for my_clm in range(opp_clm):
                    node = self.node_res[my_clm, opp_clm]
                    if node.p_player == 0 and node.p_opponent == 0:
                        continue

                    action_prob = node.strategy
                    node.u = 0.0
                    num_actions = node.num_actions
                    node_regret = np.zeros(num_actions)

                    # 'DOUBT' 행동(터미널)의 유틸리티 계산
                    # 'roll_fix[opp_clm]'은 상대방이
                    # 'opp_clm'을 하기 위해 굴린 주사위 값입니다.
                    roll = roll_fix[opp_clm]
                    # 상대가 블러핑했다면(opp_clm > roll) 내가 +1
                    doubt_util = 1.0 if opp_clm > roll else -1.0

                    node_regret[DOUBT] = doubt_util
                    node.u += action_prob[DOUBT] * doubt_util

                    if opp_clm < self.sides:  # 'ACCEPT' 행동 가능
                        roll = roll_fix[opp_clm]
                        next_node = self.node_clm[opp_clm, roll]

                        # 플레이어가 바뀌므로 유틸리티 부호 뒤집기
                        # (Java 코드 수정)
                        accept_util = -next_node.u

                        node_regret[ACCEPT] = accept_util
                        node.u += action_prob[ACCEPT] * accept_util

                    # 후회 계산 및 누적
                    for a in range(num_actions):
                        node_regret[a] -= node.u  #
                        node.regret_sum[a] += node.p_opponent * node_regret[a]  #

                    # 다음 반복을 위해 확률 초기화
                    node.p_player = 0.0
                    node.p_opponent = 0.0

    def _reset_strategy_sums_if_needed(self, iteration: int, total_iterations: int):
        """필요시 전략 합계를 재설정합니다."""
        if iteration == total_iterations // 2:
            print(f"--- Iteration {iteration}: Resetting strategy sums ---")
            for r_node in self.node_res.flat:
                if r_node is not None:
                    r_node.strategy_sum.fill(0.0)
            for c_node in self.node_clm.flat:
                if c_node is not None:
                    c_node.strategy_sum.fill(0.0)

    def print_final_strategy(self):
        """최종 전략을 출력합니다."""
        print("\n--- Liar Die FSICFR Final Strategy ---")

        # 초기 주장 정책 (첫 턴)
        print("\n=== Initial Claim Policy (OppClaim=0) ===")
        print("Roll\tAction Probabilities (Claims 1 to 6)")
        for init_roll in range(1, self.sides + 1):
            node = self.node_clm[0, init_roll]
            avg_strategy = node.get_average_strategy()
            strategy_str = ", ".join([f"{prob:.3f}" for prob in avg_strategy])
            print(f"{init_roll}\t[{strategy_str}]")

        # 응답 노드 전략
        print("\n=== Response Node Strategies ===")
        print("MyClaim\tOppClaim\tAction Probs (DOUBT, ACCEPT)")
        for my_clm in range(self.sides):
            for opp_clm in range(my_clm + 1, self.sides + 1):
                node = self.node_res[my_clm, opp_clm]
                avg_strategy = node.get_average_strategy()
                strategy_str = ", ".join([f"{prob:.3f}" for prob in avg_strategy])
                print(f"{my_clm}\t{opp_clm}\t\t[{strategy_str}]")

        # 주장 노드 전략 (첫 턴 이후)
        print("\n=== Claim Node Strategies (OppClaim > 0) ===")
        print("OppClaim\tRoll\tAction Probabilities (Claims {OppClaim+1} to 6)")
        for opp_clm in range(1, self.sides):  # 0은 이미 위에서 출력
            for roll in range(1, self.sides + 1):
                node = self.node_clm[opp_clm, roll]
                avg_strategy = node.get_average_strategy()
                strategy_str = ", ".join([f"{prob:.3f}" for prob in avg_strategy])
                print(f"{opp_clm}\t\t{roll}\t[{strategy_str}]")


if __name__ == "__main__":
    iterations = 1000000
    sides = 3

    print("Training started:")
    print(f"Training for {iterations} iterations with {sides}-sided die")
    trainer = LiarDieTrainer(sides)
    trainer.train(iterations)

    print("Training complete:")
    print("Final strategy:")
    trainer.print_final_strategy()
