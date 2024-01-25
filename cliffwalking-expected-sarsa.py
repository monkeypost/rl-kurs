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
alpha = 0.5
epsilon = 1.0
gamma = 0.99

num_steps = 40000
step_counter = 0
observation = env.reset()
observed_state = observation[0]
for i in range(num_steps):
    env.render()

    probability = random.random()

    local_list = action_table[observed_state]  
    if (probability < epsilon):
        action = random.randint(0,3)
    else:
        action = local_list.index(max(local_list))
      
    state_value_of_action = local_list[action]
    observation, reward, done, truncated, info = env.step(action)

    local_list = action_table[observation]
    expected_value = 0.0
    for j in range(len(local_list)):
        expected_value += 0.25 * local_list[j] #needed help from ChatGPT to calculate this.
    action_table[observed_state][action] = round(state_value_of_action + alpha * (reward + gamma * expected_value - state_value_of_action), 3)

    observed_state = observation
    if done or step_counter == 100:
        observation = env.reset()
        observed_state = observation[0]
        step_counter = 0
        epsilon -= 0.001
    
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
