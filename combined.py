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

def choose_combined_actions(state):
    q_action = np.argmax(q_table[state[0], state[1]])
    emp_action = np.argmax(eq_table[state[0], state[1]])
    return q_action, emp_action

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
    # Get both actions
    q_action, emp_action = choose_combined_actions(state)
    
    # Take Q-learning action first
    obs, reward1, done1, _, info = env_test.step(q_action)
    if not done1:
        # Take empowerment action second
        obs, reward2, done, _, info = env_test.step(emp_action)
        total_reward += (reward1 + reward2)
    else:
        total_reward += reward1
        done = True

    state = convert_obs_to_tuple(obs)
    env_test.render()

print(f"Total Reward: {total_reward:.2f}")
env_test.close()