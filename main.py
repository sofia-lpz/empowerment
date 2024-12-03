import gymnasium
import numpy as np
import random

env = gymnasium.make("MiniGrid-DistShift1-v0")

gamma = 0.99
alpha = 0.1
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.005

num_episodes = 100000
max_steps = 100

def get_state_key(obs):
    # disminuir agent view
    view = obs['image'][:,:,0]
    # (3x3 area)
    front_view = tuple(view[0:3, 1:4].flatten())
    return front_view

q_table = {}

def get_q_values(state_key):
    if state_key not in q_table:
        q_table[state_key] = np.random.uniform(low=0, high=0.1, size=env.action_space.n)
    return q_table[state_key]

def choose_action(state_key, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(get_q_values(state_key))

# Train
episode_rewards = []
for episode in range(num_episodes):
    obs, _ = env.reset()
    state_key = get_state_key(obs)
    total_reward = 0
    done = False

    for step in range(max_steps):
        action = choose_action(state_key, epsilon)
        next_obs, reward, done, truncated, info = env.step(action)
        next_state_key = get_state_key(next_obs)
        total_reward += reward

        # q-learning
        old_value = get_q_values(state_key)[action]
        next_max = np.max(get_q_values(next_state_key))
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state_key][action] = new_value

        state_key = next_state_key

        if done or truncated:
            break

    episode_rewards.append(total_reward)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    if episode % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

# run
env = gymnasium.make("MiniGrid-DistShift1-v0", render_mode='human')

for episode in range(5):
    obs, _ = env.reset()
    state_key = get_state_key(obs)
    done = False
    total_reward = 0

    print(f"Starting Episode {episode}")

    for step in range(max_steps):
        env.render()
        
        action = np.argmax(get_q_values(state_key))
        next_obs, reward, done, truncated, info = env.step(action)
        next_state_key = get_state_key(next_obs)
        total_reward += reward
        
        state_key = next_state_key

        if done or truncated:
            print(f"Episode {episode} finished with total reward: {total_reward}")
            break

env.close()