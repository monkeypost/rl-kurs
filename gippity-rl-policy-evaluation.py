import gymnasium as gym
import random

def random_actions():
    return [0 for j in range(4)]

env = gym.make('CliffWalking-v0')

action_table = [random_actions() for i in range(48)]

alpha = 0.1
epsilon_decay = 0.001
num_steps = 1000
max_epsilon = 0.65
max_gamma = 0.99

best_reward = float('-inf')
best_gamma = 0
best_epsilon = 0

for gamma in [round(x * 0.01, 2) for x in range(int(max_gamma * 100) + 1)]:
    for epsilon in [round(x * 0.01, 2) for x in range(int(max_epsilon * 100) + 1)]:
        action_table = [random_actions() for i in range(48)]
        epsilon_temp = epsilon
        step_counter = 0
        observation = env.reset()
        observed_state = observation[0]
        action = random.randint(0,3)
        total_reward = 0

        for i in range(num_steps):
            local_list = action_table[observed_state]  
            state_value_of_action = local_list[action]
            observation, reward, done, truncated, info = env.step(action)
            previous_action = action

            probability = random.random()
            if probability < epsilon_temp:
                action = random.randint(0, 3)
            else:
                action = local_list.index(max(local_list))

            local_list = action_table[observation]
            state_prime_value = local_list[action]
            action_table[observed_state][previous_action] = round(state_value_of_action + alpha * (reward + gamma * state_prime_value - state_value_of_action), 5)
            
            total_reward += reward
            observed_state = observation
            if done or step_counter == 100:
                observation = env.reset()
                observed_state = observation[0]
                step_counter = 0
                epsilon_temp = epsilon_temp * (1 - epsilon_decay)

            step_counter += 1

        if total_reward > best_reward:
            best_reward = total_reward
            best_gamma = gamma
            best_epsilon = epsilon

print(f"Best Gamma: {best_gamma}, Best Epsilon: {best_epsilon}, Best Reward: {best_reward}")

env.close()
