import random


class RPSTrainer:
    """
    To train a Rock-Paper-Scissors player using regret minimization.
    """

    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    NUM_ACTIONS = 3

    def __init__(self):
        self.regret_sum = [0.0] * self.NUM_ACTIONS  # summation of regrets in iterations
        self.strategy = [0.0] * self.NUM_ACTIONS  # current player's strategy
        self.strategy_sum = [
            0.0
        ] * self.NUM_ACTIONS  # summation of strategies in iterations
        self.opp_strategy = [0.3, 0.5, 0.2]  # opponent's fixed strategy

    def get_strategy(self):
        """
        Get current mixed strategy through regret-matching.
        """
        normalizing_sum = 0
        # for all actions, add regret to strategy
        for a in range(self.NUM_ACTIONS):
            self.strategy[a] = max(0, self.regret_sum[a])
            normalizing_sum += self.strategy[a]
        # for all actions, normalize the strategy and accumulate the strategy
        for a in range(self.NUM_ACTIONS):
            # normalize the strategy
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0 / self.NUM_ACTIONS
            # accumulate the strategy
            self.strategy_sum[a] += self.strategy[a]
        return self.strategy

    def get_action(self, strategy):
        """
        Get random action according to mixed-strategy distribution
        """
        r = random.random()  # generate random number between 0 and 1
        a = 0  # first action
        cumulative_probability = 0.0
        # get action
        while (
            a < self.NUM_ACTIONS - 1
        ):  # final action is not included, buts it's not a problem.
            cumulative_probability += strategy[a]
            if cumulative_probability > r:
                break
            a += 1  # next action
        return a

    def train(self, iterations):
        """
        Train the player.
        """
        action_utility = [0.0] * self.NUM_ACTIONS
        # for all iterations
        for i in range(iterations):
            # get regret-matched mixed-strategy actions
            strategy = self.get_strategy()
            my_action = self.get_action(strategy)
            other_action = self.get_action(self.opp_strategy)
            # compute action utilities
            action_utility[other_action] = 0  # draw case
            action_utility[(other_action + 1) % self.NUM_ACTIONS] = 1  # win case
            action_utility[(other_action - 1 + self.NUM_ACTIONS) % self.NUM_ACTIONS] = (
                -1
            )  # lose case
            # for all actions, accumulate action regrets
            for a in range(self.NUM_ACTIONS):
                regret = action_utility[a] - action_utility[my_action]
                self.regret_sum[a] += regret

    def get_average_strategy(self):
        """
        Get average mixed strategy across all training iterations.
        """
        avg_strategy = [0.0] * self.NUM_ACTIONS
        normalizing_sum = sum(self.strategy_sum)
        # for all actions, normalize the summation of strategies
        for a in range(self.NUM_ACTIONS):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / self.NUM_ACTIONS
        return avg_strategy


if __name__ == "__main__":
    trainer = RPSTrainer()
    trainer.train(100000)
    avg_strategy = trainer.get_average_strategy()
    actions = ["ROCK", "PAPER", "SCISSORS"]

    print("Opponent's Strategy:", trainer.opp_strategy)
    print("Calculated Optimal Strategy:", [round(p, 3) for p in avg_strategy])
    print("-" * 30)
    for i, p in enumerate(avg_strategy):
        print(f"Action: {actions[i]}, Probability: {p:.3f}")
