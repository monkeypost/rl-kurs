import gymnasium as gym
import numpy as np

np.set_printoptions(linewidth=np.inf)
env = gym.make('CliffWalking-v0')

env.reset()

states = []
for i in range(48):
    states.append(0)

gamma = 0.8
e = 0

while True:
    threshold = 0.00001 #Taken from 100 iterations
    delta = 0 #With help from ChatGpt 
    for s in range(48):       
        expected_value = 0
        current_value = states[s]
        for a in (range(4)):
            s_prime = env.unwrapped.P[s][a][0][1]
            r = env.unwrapped.P[s][a][0][2]
            terminal = env.unwrapped.P[s][a][0][3]
            
            if (r == -100): #Without this the agent gets trapped in left corner
                r = -1 

            if (terminal):
                r = 10
            expected_value += 0.25 * (r + gamma * states[s_prime])        
        delta = max(delta, abs(current_value - expected_value)) #With help from ChatGpt  
        states[s] = round(expected_value, 5)
    e += 1
    print(f"Iteration: {e}")        
    print(np.reshape(states, (4,12)))
    if (delta < threshold): #With help from ChatGpt 
        break

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