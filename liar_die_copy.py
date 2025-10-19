import numpy as np
from sequantial_strategy_manager_copy import SequentialStrategyManager as SM


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
                self.response_nodes[my_claim, opp_claim] = SM(num_actions)

        # 주장 노드 초기화
        for opp_claim in range(sides):
            for roll in range(1, sides + 1):
                # 내 주장은 상대방의 주장보다 높아야 한다
                num_actions = sides - opp_claim
                self.claim_nodes[opp_claim, roll] = SM(num_actions)

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
    trainer.train(iterations=100000)
