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
epsilon_decay = 0.99995
epsilon_min = 0.01

num_rows = len(obstacle_map)
num_cols = len(obstacle_map[0])

num_episodes = 5000
max_steps = 1000

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

def get_empowerment(state, max_steps=10):
    """
    Calculate the empowerment for a given state using information theory.
    
    Parameters:
    - state: Tuple (x, y), the agent's current position.
    - max_steps: The number of steps to consider for empowerment.

    Returns:
    - empowerment: The empowerment value (in bits) for the given state.
    """
    def simulate_action_sequence(env, state, actions):
        """simula aciones en el ambiente desde el POI y regresa la posicion"""
        x, y = state
        env.reset(seed=1, options={"start_loc": x * num_cols + y, "goal_loc": options['goal_loc']})
        for action in actions:
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        return convert_obs_to_tuple(obs)

    # 1. Enumerate all possible action sequences
    action_space_size = env.action_space.n
    all_action_sequences = list(np.ndindex((action_space_size,) * max_steps))

    # 2. Simulate resulting states and count occurrences
    state_action_counts = {}
    for actions in all_action_sequences:
        resulting_state = simulate_action_sequence(env, state, actions)
        if resulting_state not in state_action_counts:
            state_action_counts[resulting_state] = {}
        if actions not in state_action_counts[resulting_state]:
            state_action_counts[resulting_state][actions] = 0
        state_action_counts[resulting_state][actions] += 1

    # 3. conditional probabilities p(s_{t+n}|a^n) and marginal p(s_{t+n})
    total_sequences = len(all_action_sequences)
    conditional_probs = {}
    marginal_probs = {}

    for resulting_state, action_counts in state_action_counts.items():
        marginal_probs[resulting_state] = sum(action_counts.values()) / total_sequences
        for actions, count in action_counts.items():
            if resulting_state not in conditional_probs:
                conditional_probs[resulting_state] = {}
            conditional_probs[resulting_state][actions] = count / total_sequences

    # 4. mutual information I(A^n; S_{t+n})
    mutual_information = 0
    uniform_action_prob = 1 / total_sequences  # distribucion uniforme de acciones

    for resulting_state, p_s in marginal_probs.items():
        for actions, p_s_given_a in conditional_probs[resulting_state].items():
            if p_s_given_a > 0:
                mutual_information += p_s_given_a * np.log2(p_s_given_a / (p_s * uniform_action_prob))

    # Return the empowerment value
    return mutual_information

# After calculating empowerment for all states, normalize the values
empowerment_map = np.zeros((num_rows, num_cols))
for i in range(num_rows):
    for j in range(num_cols):
        if obstacle_map[i][j] == '0':
            empowerment_map[i,j] = get_empowerment((i,j), max_steps=2)
        else:
            empowerment_map[i,j] = -1  # Keep obstacles as -1

# Get min/max excluding obstacle values (-1)
valid_empowerment = empowerment_map[empowerment_map != -1]
min_emp = np.min(valid_empowerment)
max_emp = np.max(valid_empowerment)

# Normalize non-obstacle values
mask = empowerment_map != -1
empowerment_map[mask] = (empowerment_map[mask] - min_emp) / (max_emp - min_emp)

# Obstacles remain as -1

#train
for episode in range(num_episodes):
    obs, info = env.reset(seed=1, options=options)
    total_reward = 0
    done = False

    state = convert_obs_to_tuple(obs)

    for step in range(max_steps):
        action = choose_action(state, epsilon)
        obs, _, done, _, info = env.step(action)

        empowered_reward = empowerment_map[state[0], state[1]]

        next_state = convert_obs_to_tuple(obs)
        total_reward += empowered_reward

        # Q-learning update
        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        new_value = (1 - alpha) * old_value + alpha * (empowered_reward + gamma * next_max)
        q_table[state[0], state[1], action] = new_value

        state = next_state

        if done:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Total empowered_reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

np.savetxt('eqtable.txt', q_table.reshape(num_rows * num_cols, env.action_space.n))

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap using imshow
im = ax.imshow(empowerment_map, 
               cmap='YlOrRd',  # Yellow-Orange-Red colormap
               interpolation='nearest')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Empowerment')

# Customize the plot
ax.set_title('Empowerment Map')
ax.set_xlabel('Column')
ax.set_ylabel('Row')

# Add grid
ax.grid(True, which='major', color='black', linewidth=0.5)
ax.set_xticks(np.arange(-.5, num_cols, 1), minor=True)
ax.set_yticks(np.arange(-.5, num_rows, 1), minor=True)
ax.grid(True, which='minor', color='white', linewidth=2)

# Show the plot
plt.show()

# Test with visualization
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
    obs, _, done, _, info = env_test.step(action)

    reward = empowerment_map[state[0], state[1]]
    
    state = convert_obs_to_tuple(obs)
    total_reward += reward
    
    env_test.render()

print(f"Total Reward: {total_reward:.2f}")
env_test.close()
