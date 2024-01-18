import gymnasium as gym
import random

def random_actions():
    return [0 for j in range(4)]
    #return [-1 for j in range(4)]
    #return [0 for j in range(4)]

env = gym.make('CliffWalking-v0')

action_table = [random_actions() for i in range(48)]
# action_table[35] = [-1, -1, 10, -1]
# action_table[47] = [0,0,0,0]

# local_list = action_table[2]
# max_value = local_list.index(max(local_list))
# action = action_table[2][max_value]
# print(action)

#print(action_table)
alpha = 0.1
epsilon = 0.65
gamma = 0.99
epsilon_decay = 0.001
num_steps = 200000
step_counter = 0
observation = env.reset()
observed_state = observation[0]
action = random.randint(0,3)
for i in range(num_steps):
    env.render()

    local_list = action_table[observed_state]  
    state_value_of_action = local_list[action]
    observation, reward, done, truncated, info = env.step(action)

    previous_action = action
    probability = random.random()
    if (probability < epsilon):
        action = random.randint(0,3)
    else:
        action = local_list.index(max(local_list))      

    # print(f'Step {i}: observation={observation}, \
    #        reward={reward}, done={done}, info={info}, action={action}')#, värde={round(state_value_of_action, 2)}, av: {local_list}')
    
    # if (observation == 35):
    #     print(action_table[35])

    local_list = action_table[observation]
    state_prime_value = local_list[action]
    action_table[observed_state][previous_action] = round(state_value_of_action + alpha * (reward + gamma * state_prime_value - state_value_of_action), 5)

    observed_state = observation
    if done or step_counter == 100:
        observation = env.reset()
        observed_state = observation[0]
        step_counter = 0
        epsilon = epsilon * (1 - epsilon_decay)
    
    step_counter += 1

env.close()


for (k, item) in enumerate(action_table):
    print(k, item)

print(epsilon)

# För att utvärdra policyn
done = False

env = gym.make('CliffWalking-v0', render_mode="human")

observation = env.reset()
observed_state = observation[0]

while done != True:
    env.render()
    
    local_list = action_table[observed_state]  
    action = local_list.index(max(local_list))
    observation, reward, done, truncated, info = env.step(action)
    observed_state = observation

env.close()
