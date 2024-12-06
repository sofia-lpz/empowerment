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

gamma = 0.99
alpha = 0.1
epsilon = 1.0
epsilon_decay = 0.95
epsilon_min = 0.01

num_rows = len(obstacle_map)
num_cols = len(obstacle_map[0])

num_episodes = 1000
max_steps = 100

q_table = np.zeros((num_rows, num_cols, env.action_space.n))


def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state[0], state[1]])
    
def convert_obs_to_tuple(obs):
    x = obs // num_cols # integer division
    y = obs % num_cols  # modulo operation
    return (x, y)

#train
for episode in range(num_episodes):
    obs, info = env.reset(seed=1, options=options)
    total_reward = 0
    done = False

    state = convert_obs_to_tuple(obs)

    for step in range(max_steps):
        action = choose_action(state, epsilon)
        obs, reward, done, _, info = env.step(action)

        next_state = convert_obs_to_tuple(obs)
        total_reward += reward

        # Q-learning update
        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state[0], state[1], action] = new_value

        state = next_state

        if done:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

np.savetxt('qtable.txt', q_table.reshape(num_rows * num_cols, env.action_space.n))

# Test
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
    # Use epsilon=0 for testing to always choose best action
    action = choose_action(state, epsilon=0)
    obs, reward, done, _, info = env_test.step(action)
    
    state = convert_obs_to_tuple(obs)
    total_reward += reward
    
    env_test.render()

print(f"Total Reward: {total_reward:.2f}")
env_test.close()