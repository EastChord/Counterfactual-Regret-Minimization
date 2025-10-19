import random
import numpy as np
from typing import Dict, Tuple


# * 현재 다음 코드를 참고하여 검토하는 중: https://github.com/jwllee/learn-cfr/blob/master/an-intro-to-cfr/liar_die_fsicfr.py
# * 검토 결과는 '#* complete to check'으로 메모하였음.


class Node:
    """
    FSICFR 알고리즘의 정보 집합(Information Set) 노드를 나타냅니다.
    (cfr.pdf 27페이지, Liar Die player decision node)

    이 클래스는 Java 코드의 내부 Node 클래스에 해당합니다.
    `regretSum`, `strategy`, `strategySum` 등 게임 상태와
    전략을 저장하기 위한 필드들을 포함합니다.
    """

    def __init__(self, num_actions: int):
        """
        노드를 초기화합니다.
        (cfr.pdf 27페이지, Liar Die node constructor)

        Args:
            num_actions (int): 이 노드에서 가능한 행동의 수.
        """
        self.num_actions = num_actions

        # regret_sum: 누적 후회 (Java: regretSum)
        # (cfr.pdf 10페이지, Equation 4: R_i^T)
        self.regret_sum = np.zeros(num_actions)

        # strategy: 현재 반복에서의 전략 (확률 분포) (Java: strategy)
        # (cfr.pdf 11페이지, Equation 5: sigma_i^{T+1})
        self.strategy = np.zeros(num_actions)

        # strategy_sum: 모든 반복에 걸친 (도달 확률로 가중된) 전략의 누적 합계 (Java: strategySum)
        # (cfr.pdf 12페이지, Algorithm 1, line 26: s_I[a])
        self.strategy_sum = np.zeros(num_actions)

        # u: 이 노드의 (예상) 유틸리티(가치) (Java: u)
        # (cfr.pdf 25페이지, Algorithm 2, line 28, 31: n.v)
        self.u = 0.0

        # my_reach: 현재 *이 노드에서 행동하는* 플레이어가 이 노드에 도달할 확률의 합 (Java: pPlayer)
        # (cfr.pdf 25페이지, Algorithm 2, line 16, 17: n.pSum_i)
        self.my_reach = 0.0

        # opp_reach: *상대방* 플레이어가 이 노드에 도달할 확률의 합 (Java: pOpponent)
        # (cfr.pdf 25페이지, Algorithm 2, line 34: cfp)
        self.opp_reach = 0.0

    def get_strategy(self) -> np.ndarray:
        """
        후회 매칭(Regret Matching)을 기반으로 현재 전략을 계산하고,
        전략 합계(strategy_sum)를 업데이트합니다.
        (cfr.pdf 27페이지, Get Liar Die node current mixed strategy...)
        (cfr.pdf 11페이지, Equation 5)
        """
        # 1. 음수가 아닌 후회(positive regrets)만 사용합니다. (R_i^{T,+})
        self.strategy = np.maximum(0, self.regret_sum)
        normalizing_sum = np.sum(self.strategy)

        if normalizing_sum > 0:
            # 2. 확률 분포로 정규화 (Equation 5, 분모)
            self.strategy /= normalizing_sum
        else:
            # 3. 긍정적인 후회가 없으면 균등 분포 사용 (Equation 5, otherwise)
            self.strategy = np.full(self.num_actions, 1.0 / self.num_actions)

        # 4. 현재 플레이어의 도달 확률(my_reach)로 가중치를 주어 전략 합계 업데이트
        # (cfr.pdf 25페이지, Algorithm 2, line 12)
        # (cfr.pdf 27페이지, line 874)
        # strategy_sum은 최종 평균 전략을 계산하기 위해 누적됩니다.
        self.strategy_sum += self.my_reach * self.strategy
        return self.strategy

    def get_average_strategy(self) -> np.ndarray:
        """
        모든 훈련 반복에 걸친 평균 전략을 반환합니다.
        (cfr.pdf 28페이지, Get Liar Die node average mixed strategy)
        (cfr.pdf 7페이지, Get average mixed strategy...)

        이것이 CFR/FSICFR의 최종 결과물(내쉬 균형 근사치)입니다.
        """
        avg_strategy = np.copy(self.strategy_sum)
        normalizing_sum = np.sum(avg_strategy)

        if normalizing_sum > 0:
            # 전체 합계로 정규화하여 평균 확률을 구합니다.
            avg_strategy /= normalizing_sum
        else:
            # 전략이 누적되지 않았다면 균등 분포 반환
            avg_strategy = np.full(self.num_actions, 1.0 / self.num_actions)

        return avg_strategy


class LiarDieTrainer:
    """
    Liar Die 게임을 위한 FSICFR 훈련기 클래스입니다.
    (cfr.pdf 33페이지, Liar Die Trainer.java)

    Java의 LiarDieTrainer 클래스에 해당합니다.
    """

    # 행동 상수 정의 (pdf 26페이지, line 827)
    DOUBT = 0
    ACCEPT = 1

    def __init__(self, sides: int):
        """
        트레이너를 생성하고 플레이어 결정 노드(정보 집합)를 할당합니다.
        (cfr.pdf 28페이지, Construct trainer and allocate...)
        (cfr.pdf 26페이지, Liar Die definitions)
        """
        # 주사위 면 수
        self.sides = sides

        # 내 주장(0(주장 없음) ~ 6)에 반응한 상대방의 현재 주장(0 ~ 6)을 듣고, 의심 또는 수락을 결정하는 노드
        self.response_nodes = np.empty((sides + 1, sides + 1), dtype=object)

        # 상대방의 주장(1 ~ 6)과 주사위를 굴려 나온 결과(1 ~ 6)를 보고, 주장을 결정하는 노드
        self.claim_nodes = np.empty((sides, sides + 1), dtype=object)

        # 응답 노드 초기화
        for my_claim in range(sides + 1):
            for opp_claim in range(my_claim + 1, sides + 1):
                # 상대방이 주장하지 않았거나 최댓값을 주장했다면, 수락만 가능
                num_actions = 1 if (opp_claim == 0 or opp_claim == sides) else 2
                self.response_nodes[my_claim, opp_claim] = Node(num_actions)

        # 주장 노드 초기화
        for opp_claim in range(sides):
            for roll in range(1, sides + 1):
                # 내 주장은 상대방의 주장보다 높아야 한다
                num_actions = sides - opp_claim
                self.claim_nodes[opp_claim, roll] = Node(num_actions)

    def initialize(self):
        """
        주사위를 미리 굴리고, 시작 노드의 도달 확률을 1로 설정합니다.
        """
        fixed_roll = np.random.randint(1, self.sides + 1, size=self.sides)

        # 시작 노드: 상대가 주장하지 않았고, 처음으로 주사위를 굴려 나온 결과를 보고 내가 주장을 결정하는 노드
        self.claim_nodes[0, fixed_roll[0]].my_reach = 1.0
        self.claim_nodes[0, fixed_roll[0]].opp_reach = 1.0

        return fixed_roll

    def fwd_acc_response(self, opp_claim, fixed_roll):
        """
        응답 노드의 도달 확률 누적
        """
        if opp_claim == 0:
            return

        for my_claim in range(opp_claim):
            node = self.response_nodes[my_claim, opp_claim]
            action_prob = node.get_strategy()

            if opp_claim >= self.sides:
                continue

            roll = fixed_roll[opp_claim]

            next_node = self.claim_nodes[opp_claim, roll]
            next_node.my_reach += action_prob[self.ACCEPT] * node.my_reach
            next_node.opp_reach += node.opp_reach

    # * complete to check
    def fwd_acc_claim(self, opp_claim, fixed_roll):
        """
        주장 노드의 도달 확률 누적
        """
        if opp_claim >= self.sides:
            return

        roll = fixed_roll[opp_claim]
        node = self.claim_nodes[opp_claim, roll]
        action_prob = node.get_strategy()

        for my_claim in range(opp_claim + 1, self.sides + 1):
            next_claim_prob = action_prob[my_claim - opp_claim - 1]

            if next_claim_prob <= 0.0:
                continue

            next_node = self.response_nodes[opp_claim, my_claim]
            next_node.my_reach += node.opp_reach
            next_node.opp_reach += next_claim_prob * node.my_reach

    def forward_accumulation(self, fixed_roll):
        """
        모든 노드의 도달 확률 누적
        """
        for opp_claim in range(self.sides + 1):
            self.fwd_acc_response(opp_claim, fixed_roll)
            self.fwd_acc_claim(opp_claim, fixed_roll)

    def bwd_prop_claim(self, opp_claim, fixed_roll):
        """
        주장 노드의 효용 계산
        """
        if opp_claim >= self.sides:
            return
        

        roll = fixed_roll[opp_claim]
        node = self.claim_nodes[opp_claim, roll]
    
        regret = np.zeros(node.num_actions)
        
        action_prob = node.strategy
        node.u = 0.0

        for my_claim in range(opp_claim + 1, self.sides + 1):
            action_index = my_claim - opp_claim - 1
            next_node = self.response_nodes[opp_claim, my_claim]

            child_util = -next_node.u
            regret[action_index] = child_util
            node.u += action_prob[action_index] * child_util

        for a in range(len(action_prob)):
            regret[a] -= node.u
            node.regret_sum[a] += node.opp_reach * regret[a]

        node.my_reach = 0.0
        node.opp_reach = 0.0

    # * complete to check
    def bwd_prop_response(self, opp_claim, fixed_roll):
        """
        응답 노드의 효용 계산
        """
        if opp_claim == 0:
            return
        
        regret = np.zeros(2)

        for my_claim in range(opp_claim):
            node = self.response_nodes[my_claim, opp_claim]
            action_prob = node.strategy
            node.u = 0.0

            roll = fixed_roll[my_claim]
            doubt_util = 1.0 if opp_claim > roll else -1.0

            regret[self.DOUBT] = doubt_util
            node.u += action_prob[self.DOUBT] * doubt_util

            if opp_claim < self.sides:
                roll = fixed_roll[opp_claim]
                next_node = self.claim_nodes[opp_claim, roll]
                regret[self.ACCEPT] = next_node.u
                node.u += action_prob[self.ACCEPT] * next_node.u

            for a in range(len(action_prob)):
                regret[a] -= node.u
                node.regret_sum[a] += node.opp_reach * regret[a]

            node.my_reach = 0.0
            node.opp_reach = 0.0

    def backward_propagation(self, fixed_roll):
        """
        모든 노드의 효용 계산
        """
        for opp_claim in range(self.sides, -1, -1):
            self.bwd_prop_claim(opp_claim, fixed_roll)
            self.bwd_prop_response(opp_claim, fixed_roll)

    def reset_strategy_sum(self, iter, iterations):
        """
        훈련의 절반 시점에 전략 합계 초기화
        """
        if iter == iterations // 2:
            for r_node in self.response_nodes.flat:
                if r_node is not None:
                    r_node.strategy_sum.fill(0.0)
            for c_node in self.claim_nodes.flat:
                if c_node is not None:
                    c_node.strategy_sum.fill(0.0)

    def print_strategy(self):
        """
        전략 출력
        """
        for initial_roll in range(1, self.sides + 1):
            print(f"Initial claim policy with roll {initial_roll}: ")
            for prob in self.claim_nodes[0, initial_roll].get_average_strategy():
                print(f"{prob:.3f}")

        print("Old Claim\tNew Claim\tAction Probs")
        for my_claim in range(self.sides + 1):
            for opp_claim in range(my_claim + 1, self.sides + 1):
                res = self.response_nodes[my_claim, opp_claim].get_average_strategy()
                res_rounded = [f"{prob:.3f}" for prob in res]
                print(f"{my_claim}\t{opp_claim}\t\t{res_rounded}")

        print(f"Old Claim\tRoll\tAction Probs")
        for opp_claim in range(1, self.sides):
            for roll in range(1, self.sides + 1):
                clm = self.claim_nodes[opp_claim, roll].get_average_strategy()
                clm_rounded = [f"{prob:.3f}" for prob in clm]
                print(f"{opp_claim}\t{roll}\t{clm_rounded}")

    def train(self, iterations: int):
        """
        FSICFR 알고리즘을 사용하여 훈련을 실행합니다.
        """
        for iter in range(iterations):
            
            if iter % 10000 == 0:
                print(f"Iteration {iter:,} / {iterations:,}")
            
            fixed_roll = self.initialize()
            self.forward_accumulation(fixed_roll)
            self.backward_propagation(fixed_roll)
            self.reset_strategy_sum(iter, iterations)

        self.print_strategy()


if __name__ == "__main__":
    """
    메인 실행 블록.
    (cfr.pdf 33페이지, Liar Die Trainer main method)
    """
    # 6면체 주사위로 트레이너 생성
    trainer = LiarDieTrainer(sides=6)

    # 1,000,000번의 반복으로 훈련 실행
    trainer.train(iterations=1000000)
