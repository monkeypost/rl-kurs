import gymnasium as gym
import random

def random_actions():
    return [0 for j in range(4)]

env = gym.make('CliffWalking-v0')

# Initialize two sets of action tables, one for Q1 and one for Q2
action_table_Q1 = [random_actions() for i in range(48)]
action_table_Q2 = [random_actions() for i in range(48)]

alpha = 0.1
epsilon = 1.0
gamma = 0.99

num_steps = 100000
step_counter = 0
observation = env.reset()
observed_state = observation[0]

for i in range(num_steps):
    #env.render()
    
    probability = random.random()
    
    local_list_Q1 = action_table_Q1[observed_state]
    local_list_Q2 = action_table_Q2[observed_state]

    if (probability < epsilon):
        action = random.randint(0, 3)
    else:
        # Select the action with the maximum Q-value from Q1 + Q2
        action = local_list_Q1.index(max(local_list_Q1)) if random.random() < 0.5 else local_list_Q2.index(max(local_list_Q2))

    state_value_of_action = local_list_Q1[action]
    observation, reward, done, truncated, info = env.step(action)

    local_list_Q1 = action_table_Q1[observation]    
    local_list_Q2 = action_table_Q2[observation]

    # Select the best action using Q1 but evaluate using Q2
    max_state_prime_value_Q1 = max(local_list_Q1)
    max_state_prime_value_Q2 = max(local_list_Q2)
    
    if random.random() < 0.5:
        action_table_Q1[observed_state][action] = round(state_value_of_action + alpha * (reward + gamma * max_state_prime_value_Q2 - state_value_of_action), 3)
    else:
        action_table_Q2[observed_state][action] = round(state_value_of_action + alpha * (reward + gamma * max_state_prime_value_Q1 - state_value_of_action), 3)

    observed_state = observation

    if done or step_counter == 100:
        observation = env.reset()
        observed_state = observation[0]
        step_counter = 0
        epsilon -= 0.001

    step_counter += 1

env.close()

for (k, item) in enumerate(action_table_Q1):
    print(k, item)

print(epsilon)

# Evaluate the policy
done = False

env = gym.make('CliffWalking-v0', render_mode="human")

observation = env.reset()
observed_state = observation[0]

while not done:
    env.render()
    
    local_list_Q1 = action_table_Q1[observed_state]
    local_list_Q2 = action_table_Q2[observed_state]

    # Select the action with the maximum Q-value from Q1 + Q2
    action = local_list_Q1.index(max(local_list_Q1)) if random.random() < 0.5 else local_list_Q2.index(max(local_list_Q2))

    observation, reward, done, truncated, info = env.step(action)
    observed_state = observation

env.close()