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


class LiarDieTrainer:
    """
    Liar Die 게임을 위한 FSICFR 훈련기 클래스입니다.


    Java의 LiarDieTrainer 클래스에 해당합니다.
    """

    # 행동 상수 정의
    DOUBT = 0
    ACCEPT = 1

    def __init__(self, sides: int):
        """
        LiarDieTrainer를 초기화하고 필요한 노드들을 할당합니다.


        Args:
            sides (int): 사용할 주사위의 면 수.
        """
        self.sides = sides

        # responseNodes[my_claim][opp_claim]:
        # 상대가 'opp_claim'을 주장했을 때 나의(my) 응답(Doubt/Accept) 노드.
        #
        # (sides+1) 크기를 사용하여 0 (주장 없음)부터 sides까지 인덱싱합니다.
        self.response_nodes = np.empty((sides + 1, sides + 1), dtype=object)

        # claimNodes[opp_claim][roll]:
        # 상대가 'opp_claim'을 주장한 것을 내가 'Accept'한 후,
        # 'roll'이 나왔을 때 나의(my) 주장 노드.
        self.claim_nodes = np.empty((sides + 1, sides + 1), dtype=object)

        # 상대가 주장했을 때 나의 응답 노드를 할당합니다.
        for my_claim in range(sides):
            # 상대의 주장은 내 주장보다 높아야 하므로, my_claim + 1 부터 sides + 1 까지의 값을 가질 수 있습니다.
            for opp_claim in range(my_claim + 1, sides + 1):
                # opp_claim이 0이거나(불가능한 시나리오) sides와 같으면(최대치) 더 이상 주장은 불가능합니다.
                # 따라서 이 때의 행동은 응답 한 개(ACCEPT 또는 DOUBT)뿐입니다.
                num_actions = 1 if (opp_claim == 0 or opp_claim == sides) else 2
                self.response_nodes[my_claim, opp_claim] = Node(num_actions)

        # 상대의 주장에 내가 ACCEPT 한 후 roll 값이 나왔을 때의 내 주장 노드를 할당합니다.
        for opp_claim in range(sides):
            for roll in range(1, sides + 1):
                # roll보다 높은 수를 주장해야 하므로, 가능한 주장은 (opp_claim + 1)부터 sides까지입니다.
                num_actions = sides - opp_claim
                self.claim_nodes[opp_claim, roll] = Node(num_actions)

    def train(self, iterations: int):
        """
        FSICFR 알고리즘을 사용하여 훈련을 실행합니다.

        """
        # 각 'opp_claim'을 수락한 후의 주사위 굴림 값을 미리 계산합니다.
        # (찬스 노드의 결과를 미리 결정)
        roll_after_accepting_claim = np.zeros(self.sides + 1, dtype=int)

        print(
            f"Starting training for {iterations} iterations with {self.sides}-sided die..."
        )

        for iter in range(iterations):
            # --- 1. 초기화: 찬스 이벤트 결정 및 시작 확률 설정 ---
            #
            for i in range(self.sides + 1):
                # 1부터 sides 까지의 랜덤 굴림 값
                # ? self.sides + 1이어야 하지 않나요?
                roll_after_accepting_claim[i] = random.randint(1, self.sides)

            # 게임 시작: '주장 0'(없음)을 수락한 후 첫 번째 굴림
            # * 아마 이게 무작위 찬스를 고정한 부분이라고 생각 됨.
            initial_roll = roll_after_accepting_claim[0]
            # 시작 노드의 도달 확률을 1로 설정
            self.claim_nodes[0, initial_roll].p_player = 1.0
            self.claim_nodes[0, initial_roll].p_opponent = 1.0

            # --- 2. 순방향 전파: 도달 확률(실현 가중치) 누적 ---
            #
            # 토폴로지 순서(주장 값이 낮은 것에서 높은 것 순)로 노드 방문
            # 응답 노드와 주장 노드, 두 가지 노드에 대해 도달 확률을 누적합니다.
            for opp_claim in range(self.sides + 1):

                # 응답 노드 순방향 전파
                if opp_claim > 0:
                    for my_claim in range(opp_claim):
                        node = self.response_nodes[my_claim, opp_claim]
                        if node.p_player == 0 and node.p_opponent == 0:
                            continue  # 이 노드는 이번 반복에서 도달 불가

                        # 현재 전략 계산 및 strategy_sum 업데이트
                        action_prob = node.get_strategy()

                        # 상대의 주장(opp_claim)은 주사위 면 수(sides)보다 작거나, sides와 같을 수 있습니다.
                        # 상대의 주장과 주사위 면 수가 같으면, 더 높은 수를 주장할 수 없으므로, ACCEPT가 불가능합니다.
                        # 따라서 ACCEPT 가능한 경우는 opp_claim < self.sides 인 경우입니다.
                        if opp_claim < self.sides:  # ACCEPT 가능
                            roll = roll_after_accepting_claim[opp_claim]
                            next_node = self.claim_nodes[opp_claim, roll]

                            # 'ACCEPT' 행동(인덱스 1)에 대한 확률을 전파합니다.
                            # ? 그런데 왜 += 인거지?
                            next_node.p_player += (
                                action_prob[self.ACCEPT] * node.p_player
                            )
                            next_node.p_opponent += node.p_opponent

                # 주장 노드 순방향 전파
                # 상대의 주장이 sides와 같다면 더 이상 주장할 수 없습니다.
                if opp_claim < self.sides:
                    roll = roll_after_accepting_claim[opp_claim]
                    node = self.claim_nodes[opp_claim, roll]
                    if node.p_player == 0 and node.p_opponent == 0:
                        continue

                    # 현재 전략 계산 및 strategy_sum 업데이트
                    action_prob = node.get_strategy()

                    for my_claim in range(opp_claim + 1, self.sides + 1):
                        action_index = my_claim - opp_claim - 1
                        next_claim_prob = action_prob[action_index]

                        if next_claim_prob == 0:
                            continue

                        next_node = self.response_nodes[opp_claim, my_claim]
                        # 플레이어가 바뀌므로 p_player와 p_opponent를 교체
                        next_node.p_player += next_claim_prob * node.p_player
                        next_node.p_opponent += node.p_opponent

            # --- 3. 역방향 전파: 유틸리티 및 후회 계산 ---
            #
            # 토폴로지 역순(주장 값이 높은 것에서 낮은 것 순)으로 노드 방문
            for opp_claim in range(self.sides, -1, -1):  # sides 부터 0 까지

                # (Visit claim nodes backward)
                if opp_claim < self.sides:
                    roll = roll_after_accepting_claim[opp_claim]
                    node = self.claim_nodes[opp_claim, roll]
                    if node.p_player == 0 and node.p_opponent == 0:
                        continue  # 이 노드는 방문되지 않았음

                    action_prob = node.strategy  # 순방향 패스에서 계산된 전략 사용
                    node.u = 0.0
                    num_actions = node.num_actions
                    node_regret = np.zeros(num_actions)

                    for my_claim in range(opp_claim + 1, self.sides + 1):
                        action_index = my_claim - opp_claim - 1
                        next_node = self.response_nodes[opp_claim, my_claim]

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
                if opp_claim > 0:
                    for my_claim in range(opp_claim):
                        node = self.response_nodes[my_claim, opp_claim]
                        if node.p_player == 0 and node.p_opponent == 0:
                            continue

                        action_prob = node.strategy
                        node.u = 0.0
                        num_actions = node.num_actions
                        node_regret = np.zeros(num_actions)

                        # 'DOUBT' 행동(터미널)의 유틸리티 계산
                        # 'roll_after_accepting_claim[opp_claim]'은 상대방이
                        # 'opp_claim'을 하기 위해 굴린 주사위 값입니다.
                        roll = roll_after_accepting_claim[opp_claim]
                        # 상대가 블러핑했다면(opp_claim > roll) 내가 +1
                        doubt_util = 1.0 if opp_claim > roll else -1.0

                        node_regret[self.DOUBT] = doubt_util
                        node.u += action_prob[self.DOUBT] * doubt_util

                        if opp_claim < self.sides:  # 'ACCEPT' 행동 가능
                            roll = roll_after_accepting_claim[opp_claim]
                            next_node = self.claim_nodes[opp_claim, roll]

                            # 플레이어가 바뀌므로 유틸리티 부호 뒤집기
                            # (Java 코드 수정)
                            accept_util = -next_node.u

                            node_regret[self.ACCEPT] = accept_util
                            node.u += action_prob[self.ACCEPT] * accept_util

                        # 후회 계산 및 누적
                        for a in range(num_actions):
                            node_regret[a] -= node.u  #
                            node.regret_sum[a] += node.p_opponent * node_regret[a]  #

                        # 다음 반복을 위해 확률 초기화
                        node.p_player = 0.0
                        node.p_opponent = 0.0

            # --- 4. (선택적) 훈련 절반 시점에 전략 합계 재설정 ---
            #
            if iter == iterations // 2:
                print(f"--- Iteration {iter}: Resetting strategy sums ---")
                for r_node in self.response_nodes.flat:
                    if r_node is not None:
                        r_node.strategy_sum.fill(0.0)
                for c_node in self.claim_nodes.flat:
                    if c_node is not None:
                        c_node.strategy_sum.fill(0.0)

        print("Training complete.")
        # --- 5. 최종 전략 인쇄 ---
        #
        print("\n--- Liar Die FSICFR Final Strategy ---")

        # 초기 주장 정책 (첫 턴)
        print("\n=== Initial Claim Policy (OppClaim=0) ===")
        print("Roll\tAction Probabilities (Claims 1 to 6)")
        for initial_roll in range(1, self.sides + 1):
            node = self.claim_nodes[0, initial_roll]
            avg_strategy = node.get_average_strategy()
            strategy_str = ", ".join([f"{prob:.3f}" for prob in avg_strategy])
            print(f"{initial_roll}\t[{strategy_str}]")

        # 응답 노드 전략
        print("\n=== Response Node Strategies ===")
        print("MyClaim\tOppClaim\tAction Probs (DOUBT, ACCEPT)")
        for my_claim in range(self.sides):
            for opp_claim in range(my_claim + 1, self.sides + 1):
                node = self.response_nodes[my_claim, opp_claim]
                avg_strategy = node.get_average_strategy()
                strategy_str = ", ".join([f"{prob:.3f}" for prob in avg_strategy])
                print(f"{my_claim}\t{opp_claim}\t\t[{strategy_str}]")

        # 주장 노드 전략 (첫 턴 이후)
        print("\n=== Claim Node Strategies (OppClaim > 0) ===")
        print("OppClaim\tRoll\tAction Probabilities (Claims {OppClaim+1} to 6)")
        for opp_claim in range(1, self.sides):  # 0은 이미 위에서 출력
            for roll in range(1, self.sides + 1):
                node = self.claim_nodes[opp_claim, roll]
                avg_strategy = node.get_average_strategy()
                strategy_str = ", ".join([f"{prob:.3f}" for prob in avg_strategy])
                print(f"{opp_claim}\t\t{roll}\t[{strategy_str}]")


if __name__ == "__main__":
    """
    메인 실행 블록.

    """
    # 6면체 주사위로 트레이너 생성
    trainer = LiarDieTrainer(sides=3)

    # 1,000,000번의 반복으로 훈련 실행
    trainer.train(iterations=1_000_000)
