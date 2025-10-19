import numpy as np


class SimultaneousStrategyManager:
    def __init__(self, actions):
        self.actions = actions
        self.num_act = len(actions)
        self.utils = np.zeros(self.num_act)
        self.strategy_sum = np.zeros(self.num_act)
        self.regrets = np.zeros(self.num_act)
        self.regret_sum = np.zeros(self.num_act)

    def normalize(self, array):
        norm = np.maximum(0, array).sum()
        return array / norm if norm != 0 else np.zeros(len(array))

    def get_strategy(self):
        new_strategy = np.maximum(0, self.regret_sum)
        new_strategy = self.normalize(new_strategy)
        if np.all(self.regret_sum <= 0):
            new_strategy = np.full(self.num_act, 1.0 / self.num_act)
        self.strategy_sum += new_strategy
        return new_strategy

    def get_average_strategy(self):
        return self.normalize(self.strategy_sum)

    def compute_regrets(self, actual_utility: float):
        self.regrets = self.utils - actual_utility

    def update_regret_sum(self, actual_utility: float, weight=1.0):
        self.compute_regrets(actual_utility)
        self.regret_sum += weight * self.regrets

    def print_average_strategy(self):
        avg_strategy = self.get_average_strategy()
        for i, p in enumerate(avg_strategy):
            print(f"Action: {self.actions[i]:<10} Probability: {p * 100:.2f}%")