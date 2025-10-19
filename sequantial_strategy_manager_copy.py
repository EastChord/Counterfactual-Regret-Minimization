import numpy as np


class SequentialStrategyManager:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.strategy = np.zeros(self.num_actions)
        self.strategy_sum = np.zeros(self.num_actions)
        self.regrets = np.zeros(self.num_actions)
        self.regret_sum = np.zeros(self.num_actions)
        self.u = 0.0
        self.my_reach = 0.0
        self.opp_reach = 0.0

    def normalize(self, array):
        norm = np.maximum(0, array).sum()
        return array / norm if norm != 0 else np.zeros(len(array))

    def get_strategy(self):
        new_strategy = np.maximum(0, self.regret_sum)
        new_strategy = self.normalize(new_strategy)
        if np.all(self.regret_sum == 0):
            new_strategy = np.full(self.num_actions, 1.0 / self.num_actions)
        self.strategy = new_strategy
        self.strategy_sum += self.my_reach * self.strategy
        return self.strategy

    def get_average_strategy(self):
        avg_strategy = self.normalize(self.strategy_sum)
        return avg_strategy
        