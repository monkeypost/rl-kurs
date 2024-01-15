import gymnasium as gym
import numpy as np

np.set_printoptions(linewidth=np.inf)
env = gym.make('CliffWalking-v0')

env.reset()

states = []
for i in range(37):
    states.append(0)

for i in range(37, 47):
    states.append(1.11111) #Just to make print even
states.append(0)

gamma = 0.9

do_not_caluclate = list(range(37, 48))
for e in range(100):
    threshold = 0
    for s in range(len(states)):       
        is_in_range = do_not_caluclate.count(s)
        if (is_in_range == 0): 
            expected_value = 0
            for a in (range(4)):
                current_value = states[s]
                s_prime = env.unwrapped.P[s][a][0][1]
                r = env.unwrapped.P[s][a][0][2]
                if (r == -100): #Without this the agent gets trapped in left corner
                    r = -1 
                expected_value += 0.25 * (r + gamma * states[s_prime])
            current_value
                
            states[s] = round(expected_value, 5)


print(np.reshape(states, (4,12)))
# För att utvärdra policyn
done = False

env = gym.make('CliffWalking-v0', render_mode="human")

observation = env.reset()
observed_state = observation[0]

while done != True:
    env.render()
    
    action = 0
    current_value = states[observed_state]
    for a in (range(4)):
        s_prime = env.unwrapped.P[observed_state][a][0][1]
        r = env.unwrapped.P[observed_state][a][0][2]
        if (states[s_prime] > current_value):
            action = a
            current_value = states[s_prime]
    observation, reward, done, truncated, info = env.step(action)
    observed_state = observation

env.close()