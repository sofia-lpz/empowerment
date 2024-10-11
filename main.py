from random_agent import RandomAgent
from env import SimpleEnv
import numpy as np

def main():
    env = SimpleEnv(render_mode="human")

    random_agent = RandomAgent(env)
    second_random_agent = RandomAgent(env)

    actions = []

    state = env.reset()
    done = False

    while not done:

        action_probabilities = random_agent.choose_action()
        second_action_probabilities = second_random_agent.choose_action()

        # average the probabilities for each action

        average_action_probabilities = (action_probabilities + second_action_probabilities) / 2
        print("Average Action Probabilities:", average_action_probabilities)

        action = np.argmax(average_action_probabilities)


        actions.append(action)

        # take action
        result = env.step(action)
        state, reward, done = result[:3]  # Unpack only the first three values
        
        # Render the environment to display it
        env.render()

    print("Actions taken by the Random Agent:", actions)

if __name__ == "__main__":
    main()