import random
from re import A
import numpy as np


# utility function
def normalize(array):
    norm = np.maximum(array, 0).sum()
    return array / norm if norm != 0 else [0.0] * len(array)


# Liar Die definitions
DOUBT = 0
ACCEPT = 1


class node:
    # constructor
    def __init__(self, num_actions):
        self.regret_sum = np.zeros(num_actions)
        self.strategy = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.u = 0.0
        self.p_player = 0.0
        self.p_opponent = 0.0

    # get liar die node current mixed strategy through regret-matching
    def get_strategy(self):
        new_strategy = np.maximum(0, self.regret_sum)
        new_strategy = normalize(new_strategy)

        if np.all(new_strategy == 0):
            new_strategy = np.ones(len(self.strategy)) / len(self.strategy)

        self.strategy_sum = np.add(self.strategy_sum, self.p_player * new_strategy)

        return new_strategy

    # get liar die node average mixed strategy
    def get_average_strategy(self):
        return normalize(self.strategy_sum)


# construct trainer and allocate player decision nodes
class trainer:
    def __init__(self, sides):
        self.sides = sides

        # Initialize response nodes: 2D array [myClaim][oppClaim]
        self.response_nodes = [
            [None for _ in range(sides + 1)] for _ in range(sides + 1)
        ]
        for my_claim in range(sides + 1):
            for opp_claim in range(my_claim + 1, sides + 1):
                # Number of actions: 1 if oppClaim is 0 or sides, otherwise 2
                num_actions = 1 if (opp_claim == 0 or opp_claim == sides) else 2
                self.response_nodes[my_claim][opp_claim] = node(num_actions)

        # Initialize claim nodes: 2D array [oppClaim][roll]
        self.claim_nodes = [[None for _ in range(sides + 1)] for _ in range(sides)]
        for opp_claim in range(sides):
            for roll in range(1, sides + 1):
                # Number of actions: sides - oppClaim
                num_actions = sides - opp_claim
                self.claim_nodes[opp_claim][roll] = node(num_actions)

    # train with FSICFR
    def train(self, iterations):
        regret = np.zeros(self.sides)
        roll_after_accepting_claim = np.zeros(self.sides)

        for _ in range(iterations):

            ### initialize rolls and starting probabilities ###

            for i in range(len(roll_after_accepting_claim)):
                roll_after_accepting_claim[i] = random.nextInt(self.sides) + 1

            self.claim_nodes[0][roll_after_accepting_claim[0]].p_player = 1.0
            self.claim_nodes[0][roll_after_accepting_claim[0]].p_opponent = 1.0

            ### accumulate realization weights forward ###
            
            for opp_claim in range(self.sides):
                
                ### visit response nodes forward ###
                
                if opp_claim <= 0:
                    continue
                
                for my_claim in range(0, opp_claim):
                    node = self.response_nodes[my_claim][opp_claim]
                    action_prob = node.get_strategy()

                
                ### visit claim nodes forword ###
                  
            # backpropagate utilites, adjusting regret and strategies
            # reset strategy sums after half of training

        # print resulting strategy
