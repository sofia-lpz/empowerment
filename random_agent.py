import random
import numpy as np
from env import SimpleEnv

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def choose_action(self):
        # Create a vector of random probabilities for each action in the action space
        action_probabilities = np.random.rand(self.env.action_space.n)
        
        # Select the action with the highest probability
        action = np.argmax(action_probabilities)
        
        return action_probabilities
