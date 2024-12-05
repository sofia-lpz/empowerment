import gymnasium as gym
from gym_simplegrid.envs import SimpleGridEnv
import numpy as np
import random
import matplotlib.pyplot as plt

options = {
    'start_loc': 12,
    'goal_loc': (2,0)
}

obstacle_map = [
    "10001000",
    "10010000",
    "00000001",
    "01000001",
]

options ={
        'start_loc': 12,
        'goal_loc': (2,0)
    }

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

def get_empowerment(state, max_steps=5):
    """
    Calculate the empowerment using the information-theoretic approach.

    Parameters:
    - state: Tuple (x, y) representing the agent's current position.
    - max_steps: Number of steps for computing empowerment.

    Returns:
    - empowerment: The channel capacity (in bits) of the actuation-sensor loop.
    """
    def simulate_action_sequence(env, state, actions):
        """Simulate an action sequence and return the resulting sensor state."""
        x, y = state
        env.reset(seed=1, options={"start_loc": x * num_cols + y, "goal_loc": options['goal_loc']})
        for action in actions:
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        return convert_obs_to_tuple(obs)

    # 1. Define all possible action sequences (up to max_steps)
    action_space_size = env.action_space.n
    all_action_sequences = [
        list(seq) for seq in np.ndindex((action_space_size,) * max_steps)
    ]

    # 2. Compute conditional probabilities p(s_{t+n}|a^n)
    state_counts = {}
    for actions in all_action_sequences:
        resulting_state = simulate_action_sequence(env, state, actions)
        state_counts[resulting_state] = state_counts.get(resulting_state, 0) + 1

    total_sequences = len(all_action_sequences)
    conditional_probabilities = {
        state: count / total_sequences for state, count in state_counts.items()
    }

    # 3. Compute marginal probabilities p(s_{t+n})
    marginal_probabilities = {}
    for state, prob in conditional_probabilities.items():
        marginal_probabilities[state] = marginal_probabilities.get(state, 0) + prob

    # 4. Compute mutual information I(A^n; S_{t+n})
    mutual_information = 0
    for state, p_s in marginal_probabilities.items():
        for actions in all_action_sequences:
            resulting_state = simulate_action_sequence(env, state, actions)
            if resulting_state == state:
                p_s_given_a = conditional_probabilities[state]
                p_a = 1 / total_sequences  # Uniform distribution over actions
                mutual_information += p_a * p_s_given_a * np.log2(p_s_given_a / p_s)

    # 5. Empowerment is the maximum mutual information
    return mutual_information

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

empowerment_map = np.zeros((num_rows, num_cols))
for i in range(num_rows):
    for j in range(num_cols):
        if obstacle_map[i][j] == '0':
            empowerment_map[i,j] = get_empowerment((i,j), max_steps=2)
        else:
            empowerment_map[i,j] = np.nan

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(empowerment_map, 
               cmap='YlOrRd',
               interpolation='nearest')

cbar = plt.colorbar(im)
cbar.set_label('Empowerment')

ax.set_title('Empowerment Map')
ax.set_xlabel('Column')
ax.set_ylabel('Row')

ax.grid(True, which='major', color='black', linewidth=0.5)
ax.set_xticks(np.arange(-.5, num_cols, 1), minor=True)
ax.set_yticks(np.arange(-.5, num_rows, 1), minor=True)
ax.grid(True, which='minor', color='white', linewidth=2)

plt.show()

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