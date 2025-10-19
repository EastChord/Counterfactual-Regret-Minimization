import numpy as np


class SequentialStrategyManager:
    def __init__(self, info, actions):
        self.info = info
        self.actions = actions
        self.num_actions = len(actions)
        self.utils = np.zeros(self.num_actions)
        self.strategy = np.zeros(self.num_actions)
        self.strategy_sum = np.zeros(self.num_actions)
        self.regrets = np.zeros(self.num_actions)
        self.regret_sum = np.zeros(self.num_actions)

    def normalize(self, array):
        norm = np.maximum(0, array).sum()
        return array / norm if norm != 0 else np.zeros(len(array))

    def update_and_get_strategy(self, weight=1.0):
        new_strategy = np.maximum(0, self.regret_sum)
        new_strategy = self.normalize(new_strategy)
        if np.all(self.regret_sum <= 0):
            new_strategy = np.full(self.num_actions, 1.0 / self.num_actions)
        self.strategy = new_strategy
        self.strategy_sum += weight * self.strategy
        return self.strategy

    def get_average_strategy(self):
        avg_strategy = self.normalize(self.strategy_sum)
        return avg_strategy

    def get_average_util(self):
        return np.sum(self.utils * self.strategy)

    def update_regret_sum(self, weight=1.0):
        self.regrets = self.utils - self.get_average_util()
        self.regret_sum += weight * self.regrets

    def print_average_strategy(self):
        avg_strategy = self.get_average_strategy()
        print(f"{self.info:>4}: [", end="")
        for i, p in enumerate(avg_strategy):
            print(f"{self.actions[i]:<5}: {p * 100:.2f}% ", end="")
        print("]")


class SequentialStrategyManagerMap:
    def __init__(self):
        self.sm_map = {}

    def get_strategy_manager(self, info, actions):
        if (sm := self.sm_map.get(info)) is None:
            sm = SequentialStrategyManager(info, actions)
            self.sm_map[info] = sm
        return sm

    def print_average_strategy(self):
        sorted_sm_map = sorted(self.sm_map.items())
        for info, sm in sorted_sm_map:
            sm.print_average_strategy()
