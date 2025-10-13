import random
import numpy as np
from sklearn.preprocessing import normalize


class KuhnTrainer:
    """
    Trainer for Kuhn Poker game using CFR algorithm.
    """

    # define actions as constants
    PASS = 0
    BET = 1
    NUM_ACTIONS = 2

    class Node:
        """
        Expression for information set of game tree.
        """

        def __init__(self, info_set):
            self.info_set = info_set
            self.regret_sum = [0.0] * KuhnTrainer.NUM_ACTIONS
            self.strategy = [0.0] * KuhnTrainer.NUM_ACTIONS
            self.strategy_sum = [0.0] * KuhnTrainer.NUM_ACTIONS

        @staticmethod
        def normalize(values):
            norm = np.abs(values).sum()
            return values / norm if norm != 0 else [0] * len(values)

        def get_strategy(self, realization_weight):
            """
            Get current mixed strategy through regret-matching.
            """
            # get strategy from regret sum
            for i in range(KuhnTrainer.NUM_ACTIONS):
                self.strategy[i] = self.regret_sum[i] if self.regret_sum[i] > 0 else 0

            # normalize the strategy using sklearn L1 normalization
            self.strategy = self.normalize(self.strategy)
            
            # exception handling
            if sum(self.strategy) == 0:
                self.strategy = [1.0 / KuhnTrainer.NUM_ACTIONS] * KuhnTrainer.NUM_ACTIONS

            # accumulate the current strategy with realization weight
            for i in range(KuhnTrainer.NUM_ACTIONS):
                self.strategy_sum[i] += realization_weight * self.strategy[i]

            return self.strategy

        def get_average_strategy(self):
            return self.normalize(self.strategy_sum)

        def __str__(self) -> str:
            avg_strat = self.get_average_strategy()
            return f"{self.info_set:>4}: [PASS: {avg_strat[0]:.3f}, BET: {avg_strat[1]:.2f}]"

    def __init__(self):
        self.node_map = {}  # map to store all of information sets

    def train(self, iterations):
        """
        Find Nash Equilibrium of Kuhn Poker game using CFR algorithm.
        Args:
            iterations (int): Number of training iterations.
        """
        cards = [1, 2, 3]
        util = 0.0
        
        # 출력할 노드들 (예: 첫 번째 노드만)
        target_info_sets = set()

        for iteration in range(iterations):
            # shuffle cards
            random.shuffle(cards)

            # Get utility by calling cfr function recursively
            util += self.cfr(cards, "", 1.0, 1.0)
            
        print(f"\nAverage game value: {util / iterations}")
        
        # 최종 결과 출력
        sorted_nodes = sorted(self.node_map.items())
        for _, node in sorted_nodes:
            print(node)

    def cfr(self, cards, history, p0, p1):
        """
        Counterfactual Regret Minimization algorithm.
        """
        plays = len(history)  # how many plays have been made
        player = plays % 2  # 0 for first player, 1 for second player
        opponent = 1 - player  # 1 for first player, 0 for second player

        # return payoff for terminal states
        if plays > 1:
            # activate pass flag if final action is pass. (pp, bp, pbp)
            terminal_pass = history[-1] == "p"
            # activate double-bet flag if final actions are bet and bet. (bb, pbb)
            double_bet = history[-2:] == "bb"
            # activate higher flag if player card is higher than opponent card
            is_player_card_higher = cards[player] > cards[opponent]

            if terminal_pass:
                if history == "pp":
                    return 1 if is_player_card_higher else -1
                else:
                    return 1
            elif double_bet:
                return 2 if is_player_card_higher else -2

        # e.g. if card is 1 and history is "p" then info set is "1p"
        info_set = str(cards[player]) + history

        # get information set node or creat it if nonexistance
        node = self.node_map.get(info_set)
        if node is None:
            node = self.Node(info_set)
            self.node_map[info_set] = node

        # for each action, recursively call cfr with additional history and probability
        realization_weight = p0 if player == 0 else p1
        strategy = node.get_strategy(realization_weight)

        util = [0.0] * self.NUM_ACTIONS
        node_util = 0.0

        for a in range(self.NUM_ACTIONS):
            next_history = history + ("p" if a == self.PASS else "b")

            if player == 0:
                util[a] = -self.cfr(cards, next_history, p0 * strategy[a], p1)
            else:
                util[a] = -self.cfr(cards, next_history, p0, p1 * strategy[a])

            node_util += strategy[a] * util[a]

        # for each eaction, compute and accumulate counterfactual regret
        for a in range(self.NUM_ACTIONS):
            regret = util[a] - node_util
            opponent_reach_prob = p1 if player == 0 else p0
            node.regret_sum[a] += opponent_reach_prob * regret

        return node_util


if __name__ == "__main__":
    """
    Main execution block: create a trainer object and start training.
    """
    # Monte Carlo method, the more iterations, the more accurate the result.
    iterations = 100000
    trainer = KuhnTrainer()
    trainer.train(iterations)
