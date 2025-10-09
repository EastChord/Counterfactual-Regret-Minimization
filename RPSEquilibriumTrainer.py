import random


class RPSEquilibriumTrainer:
    """
    Trainer for two players to learning the Nash Equilibrium of Rock-Paper-Scissors
    by using Regret Minimization.

    Read the comments of RSPTrainer.py for more details of the codes.
    """

    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    NUM_ACTIONS = 3
    NUM_PLAYERS = 2

    def __init__(self):
        # initialize states for two players
        self.regret_sum = [[0.0] * self.NUM_ACTIONS for _ in range(self.NUM_PLAYERS)]
        self.strategy = [[0.0] * self.NUM_ACTIONS for _ in range(self.NUM_PLAYERS)]
        self.strategy_sum = [[0.0] * self.NUM_ACTIONS for _ in range(self.NUM_PLAYERS)]

    def get_strategy(self, player):
        """
        Get current mixed strategy for a player through regret-matching.
        """
        normalizing_sum = 0

        current_strategy = self.strategy[player]
        current_regret_sum = self.regret_sum[player]

        for a in range(self.NUM_ACTIONS):
            current_strategy[a] = max(0, current_regret_sum[a])
            normalizing_sum += current_strategy[a]

        for a in range(self.NUM_ACTIONS):
            if normalizing_sum > 0:
                current_strategy[a] /= normalizing_sum
            else:
                current_strategy[a] = 1.0 / self.NUM_ACTIONS
            self.strategy_sum[player][a] += current_strategy[a]

        return current_strategy

    def get_action(self, strategy):
        """
        Get random action according to mixed-strategy distribution
        """
        r = random.random()
        a = 0
        cumulative_probability = 0.0
        while a < self.NUM_ACTIONS - 1:
            cumulative_probability += strategy[a]
            if r < cumulative_probability:
                break
            a += 1
        return a

    def train(self, iterations):
        """
        Train the players.
        """
        action_utility = [0.0] * self.NUM_ACTIONS

        for _ in range(iterations):
            # calculate strategies and actions for two players
            strategy_p1 = self.get_strategy(0)
            strategy_p2 = self.get_strategy(1)
            action_p1 = self.get_action(strategy_p1)
            action_p2 = self.get_action(strategy_p2)

            # calculate regret for player 1 (utility of player 1 due to player 2's action)
            action_utility[action_p2] = 0
            action_utility[(action_p2 + 1) % self.NUM_ACTIONS] = 1
            action_utility[(action_p2 - 1 + self.NUM_ACTIONS) % self.NUM_ACTIONS] = -1
            for a in range(self.NUM_ACTIONS):
                regret = action_utility[a] - action_utility[action_p1]
                self.regret_sum[0][a] += regret

            # calculate regret for player 2 (utility of player 2 due to player 1's action)
            action_utility[action_p1] = 0
            action_utility[(action_p1 + 1) % self.NUM_ACTIONS] = 1
            action_utility[(action_p1 - 1 + self.NUM_ACTIONS) % self.NUM_ACTIONS] = -1
            for a in range(self.NUM_ACTIONS):
                regret = action_utility[a] - action_utility[action_p2]
                self.regret_sum[1][a] += regret

    def get_average_strategy(self, player):
        """
        Get average mixed strategy for a player across all training iterations.
        """
        avg_strategy = [0.0] * self.NUM_ACTIONS
        current_strategy_sum = self.strategy_sum[player]
        normalizing_sum = sum(current_strategy_sum)

        for a in range(self.NUM_ACTIONS):
            if normalizing_sum > 0:
                avg_strategy[a] = current_strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / self.NUM_ACTIONS

        return avg_strategy


if __name__ == "__main__":
    trainer = RPSEquilibriumTrainer()
    trainer.train(100000)

    actions = ["ROCK", "PAPER", "SCISSORS"]

    print("--- Player 1 Final Strategy ---")
    p1_strategy = trainer.get_average_strategy(0)
    for i, p in enumerate(p1_strategy):
        print(f"Action: {actions[i]:<10} Probability: {p:.3f}")

    print("\n--- Player 2 Final Strategy ---")
    p2_strategy = trainer.get_average_strategy(1)
    for i, p in enumerate(p2_strategy):
        print(f"Action: {actions[i]:<10} Probability: {p:.3f}")
