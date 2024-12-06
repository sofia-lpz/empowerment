import gymnasium as gym
from gym_simplegrid.envs import SimpleGridEnv
import numpy as np
import random
import matplotlib.pyplot as plt
from env_options import obstacle_map, options

env = gym.make(
    'SimpleGrid-v0', 
    obstacle_map=obstacle_map, 
)

num_rows = len(obstacle_map)
num_cols = len(obstacle_map[0])

q_table = np.loadtxt('qtable.txt').reshape(len(obstacle_map), len(obstacle_map[0]), env.action_space.n)
eq_table = np.loadtxt('eqtable.txt').reshape(len(obstacle_map), len(obstacle_map[0]), env.action_space.n)

def choose_action(state):
    return np.argmax(q_table[state[0], state[1]])

def convert_obs_to_tuple(obs):
    x = obs // num_cols # integer division
    y = obs % num_cols  # modulo operation
    return (x, y)

env_test = gym.make(
    'SimpleGrid-v0', 
    obstacle_map=obstacle_map, 
    render_mode='human'
)

obs, info = env_test.reset(seed=1, options=options)
state = convert_obs_to_tuple(obs)

done = False
total_reward = 0

while not done:
    action = choose_action(state)
    obs, reward, done, _, info = env_test.step(action)

    state = convert_obs_to_tuple(obs)
    total_reward += reward
    
    env_test.render()

print(f"Total Reward: {total_reward:.2f}")
env_test.close()
