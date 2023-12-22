import gymnasium as gym
import numpy as np

# Create the Cliff Walking environment
env = gym.make('CliffWalking-v0')

# Set the discount factor
gamma = 0.99

# Set the learning rate
alpha = 0.1

# Set the number of episodes to train for
num_episodes = 1000

# Set the maximum number of steps per episode
max_steps_per_episode = 100

# Set the exploration rate
exploration_rate = 1.0

# Set the exploration decay rate
exploration_decay_rate = 0.001

# Create the Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Train the agent
for episode in range(num_episodes):
    # Reset the environment
    observation_env = env.reset()
    observation = observation_env[0]
    # Set the initial reward
    reward = 0

    # Set the initial done flag
    done = False

    # Set the initial step count
    step = 0

    # Take the given number of steps
    while not done and step < max_steps_per_episode:
        # Choose an action
        if np.random.uniform(0, 1) > exploration_rate:
            action = np.argmax(q_table[observation])
        else:
            action = env.action_space.sample()

        # Take the action and get the next observation, reward, and done flag
        next_observation, reward, done, truncated, info = env.step(action)

        #next_observation = next_observation_env[0]
        # Update the Q-value for the current state and action
        q_table[observation, action] = q_table[observation, action] + alpha * (reward + gamma * np.max(q_table[next_observation]) - q_table[observation, action])

        # Set the current observation to the next observation
        observation = next_observation

        # Increment the step count
        step += 1

    # Decrease the exploration rate
    exploration_rate = exploration_rate * (1 - exploration_decay_rate)

for (i, item) in enumerate(q_table):
    print(i, item)