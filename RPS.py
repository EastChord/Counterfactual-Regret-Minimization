import numpy as np
from simultaneous_strategy_manager import SimultaneousStrategyManager

class RPSGameManager:
    def __init__(self):
        self.players = {0: "player1", 1: "player2"}
        self.actions = {0: "rock", 1: "paper", 2: "scissors"}
        self.num_actions = len(self.actions)
        self.num_players = len(self.players)

    def get_action(self, strategy):
        a = np.random.choice(self.num_actions, p=strategy)
        return self.actions[a]

    def get_payoff(self, act1, act2):
        if act1 == act2:
            return 0
        if act1 == "rock" and act2 == "scissors":
            return 1
        if act1 == "paper" and act2 == "rock":
            return 1
        if act1 == "scissors" and act2 == "paper":
            return 1
        return -1


gm = RPSGameManager()
sm1 = SimultaneousStrategyManager(gm.actions)
sm2 = SimultaneousStrategyManager(gm.actions)


def regret_matching():
    st1 = sm1.get_strategy()
    st2 = sm2.get_strategy()
    act1 = gm.get_action(st1)
    act2 = gm.get_action(st2)
    p1_actual_payoff = gm.get_payoff(act1, act2)
    p2_actual_payoff = gm.get_payoff(act2, act1)
    for i, action in enumerate(gm.actions.values()):
        sm1.utils[i] = gm.get_payoff(action, act2)
        sm2.utils[i] = gm.get_payoff(action, act1)
    sm1.update_regret_sum(p1_actual_payoff)
    sm2.update_regret_sum(p2_actual_payoff)


def train(iterations):
    for _ in range(iterations):
        regret_matching()


if __name__ == "__main__":
    train(1000000)
    print("---- Player 1's Optimal Strategy ----")
    sm1.print_average_strategy()
    print("---- Player 2's Optimal Strategy ----")
    sm2.print_average_strategy()
