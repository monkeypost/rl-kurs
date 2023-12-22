import gymnasium as gym
import random

def random_actions():
    return [round(random.uniform(-2.00, 2.00), 2) for j in range(4)]
    #return [-1 for j in range(4)]
    #return [0 for j in range(4)]

env = gym.make('CliffWalking-v0', render_mode="human")

action_table = [random_actions() for i in range(48)]
action_table[35] = [-1, -1, 10, -1]
action_table[47] = [0,0,0,0]

# local_list = action_table[2]
# max_value = local_list.index(max(local_list))
# action = action_table[2][max_value]
# print(action)

#print(action_table)
alpha = 0.1
epsilon = 1.0
gamma = 0.1

num_steps = 10000
step_counter = 0
observation = env.reset()
observed_state = observation[0]
for i in range(num_steps):
    env.render()

    probability = random.random()

    local_list = action_table[observed_state]  
    if (probability < epsilon):
        action = random.randint(0,3)
        print('tar random action')
    else:
        action = local_list.index(max(local_list))

    if (alpha > 0.01 and i % 100 == 0):        
        alpha -= 0.01
        print(f'Alpha nu {alpha}')
      
    state_value_of_action = local_list[action]
    observation, reward, done, truncated, info = env.step(action)

    print(f'Step {i}: observation={observation}, \
           reward={reward}, done={done}, info={info}, action={action}')#, värde={round(state_value_of_action, 2)}, av: {local_list}')
    
    if (observation == 35):
        print(action_table[35])

    local_list = action_table[observation]
    max_state_prime_value = max(local_list)
    action_table[observed_state][action] = state_value_of_action + alpha * (reward + gamma * max_state_prime_value - state_value_of_action)

    observed_state = observation
    if done or step_counter == 100:
        observation = env.reset()
        observed_state = observation[0]
        step_counter = 0
    
    step_counter += 1
    epsilon -= 0.001

env.close()


for (k, item) in enumerate(action_table):
    print(k, item)

# new_observation_value = 1.7
# local_list = action_table[new_observation_value]
# max_value = 1.7
# val = round(1.5 + alpha * (-1 + gamma * 1.7 - 1.5), 2)

# print(val)